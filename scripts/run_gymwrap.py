import os
import math
import time
from time import perf_counter

import numpy as np

from core.map import load_ros_map, build_distance_field_m
from core.sim import SimParams
from gymwrap.env import MiniSimGymEnv

import rmppi_cuda


def sample_free_state(
    free_mask: np.ndarray, meta: dict, v0: float = 0.0, seed: int | None = None
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    H, W = free_mask.shape
    res = float(meta["resolution"])
    x0, y0, _ = meta["origin"]

    free_idx = np.argwhere(free_mask > 0)
    if free_idx.size == 0:
        raise RuntimeError("free_mask has no free cells")

    v, u = free_idx[rng.integers(0, len(free_idx))]
    x = x0 + (u * res)
    y = y0 + ((H - 1 - v) * res)
    yaw = float(rng.uniform(-np.pi, np.pi))
    return np.array([x, y, yaw, v0], dtype=np.float32)


def pick_carrot_goal_from_scan(
    state: np.ndarray,
    scan: np.ndarray,
    angles_body: np.ndarray,
    max_range_m: float,
    lookahead_m: float,
    v_goal: float,
) -> np.ndarray:
    # state = [x, y, yaw, v]
    x, y, yaw, v = map(float, state)
    scan = np.asarray(scan, dtype=np.float32)

    rng = np.clip(scan, 0.0, float(max_range_m))
    score = rng - 0.30 * np.abs(angles_body) * float(max_range_m)

    i = int(np.argmax(score))
    a = float(angles_body[i])
    d = float(min(lookahead_m, rng[i]))

    goal_yaw = yaw + a
    xg = x + d * math.cos(goal_yaw)
    yg = y + d * math.sin(goal_yaw)
    return np.array([xg, yg, goal_yaw, float(v_goal)], dtype=np.float32)


def main():
    map_yaml = os.environ.get("MINISIM_MAP_YAML", "").strip()
    if not map_yaml:
        raise RuntimeError(
            "Set MINISIM_MAP_YAML, e.g.\n"
            "export MINISIM_MAP_YAML=~/src/f1tenth_gym/gym/f110_gym/envs/maps/vegas.yaml"
        )

    occ, free, meta = load_ros_map(map_yaml)
    dist_m = build_distance_field_m(free, meta)

    # --- Defaults (you can override with env vars)
    dt = float(os.environ.get("MINISIM_DT", "0.02"))
    n_beams = int(os.environ.get("MINISIM_N_BEAMS", "91"))
    v_max = float(os.environ.get("MINISIM_VMAX", "6.0"))
    tau_speed = float(os.environ.get("MINISIM_TAU_SPEED", "0.12"))

    render_every = int(os.environ.get("MINISIM_RENDER_EVERY", "15"))
    rays_stride = int(os.environ.get("MINISIM_RAY_STRIDE", "30"))

    p = SimParams(dt=dt, n_beams=n_beams)
    if hasattr(p, "v_max"):
        p.v_max = v_max

    env = MiniSimGymEnv(
        occ,
        meta,
        dist_m=dist_m,
        params=p,
        render=True, #was true
        render_every=render_every,
        render_rays=True, #was true
        rays_stride=rays_stride,
        action_mode="speed_cmd",
        tau_speed=tau_speed,
        accel_limit=5.0,
    )

    # --- Reset
    s0 = sample_free_state(free, meta, v0=0.0, seed=0)
    obs, _ = env.reset(options={"state": s0})

    # --- MPPI controller
    wheelbase = float(getattr(p, "wheelbase", 0.33))
    ctrl = rmppi_cuda.F1TenthKinematicMPPI(
        float(dt),
        float(wheelbase),
        float(tau_speed),
        float(getattr(p, "v_max", v_max)),
        2.0,  # lambda
        1.0,  # alpha
        1,    # max_iter
    )

    # Quadratic weights for [x, y, yaw, v]
    q = np.array([30.0, 30.0, 8.0, 6.0], dtype=np.float32)

    fov = float(getattr(p, "fov_rad", math.radians(270.0)))
    angles_body = (-0.5 * fov) + np.linspace(
        0.0, fov, int(n_beams), endpoint=False, dtype=np.float32
    )

    v_goal = float(os.environ.get("MINISIM_VGOAL", "3.5"))
    v_min = float(getattr(p, "v_min", 0.0))
    v_max_eff = float(getattr(p, "v_max", v_max))
    steer_lim = float(getattr(p, "steer_limit", 0.418))
    max_range_m = float(getattr(p, "max_range_m", 10.0))

    # --- Timing accumulators
    t_wall = time.time()
    goal_ms_acc = 0.0
    mppi_ms_acc = 0.0
    step_ms_acc = 0.0

    for k in range(1, 200001):
        state = np.asarray(obs["state"], dtype=np.float32)
        scan = np.asarray(obs["scan"], dtype=np.float32)

        v = float(state[3])
        lookahead = float(np.clip(1.5 + 0.8 * v, 1.5, 6.0))

        t0 = perf_counter()
        goal = pick_carrot_goal_from_scan(
            state, scan, angles_body, max_range_m, lookahead, v_goal
        )
        t1 = perf_counter()

        ctrl.set_goal(goal, q)
        u = np.asarray(ctrl.compute_control(state), dtype=np.float32)  # [steer, v_cmd]
        ctrl.slide(1)
        t2 = perf_counter()

        # clamp action for safety
        u[0] = np.clip(u[0], -steer_lim, steer_lim)
        u[1] = np.clip(u[1], v_min, v_max_eff)

        obs, r, terminated, truncated, info = env.step(u)
        t3 = perf_counter()

        goal_ms_acc += (t1 - t0) * 1000.0
        mppi_ms_acc += (t2 - t1) * 1000.0
        step_ms_acc += (t3 - t2) * 1000.0

        if (k % 100) == 0:
            dt_wall = time.time() - t_wall
            fps = 100.0 / max(dt_wall, 1e-9)

            print(
                f"k={k:6d} fps={fps:6.1f} v={state[3]:.2f} steer={u[0]:+.3f} v_cmd={u[1]:.2f} | "
                f"goal_ms={goal_ms_acc/100.0:6.2f} mppi_ms={mppi_ms_acc/100.0:6.2f} step_ms={step_ms_acc/100.0:6.2f}"
            )
            t_wall = time.time()
            goal_ms_acc = mppi_ms_acc = step_ms_acc = 0.0

        if terminated or truncated:
            s0 = sample_free_state(free, meta, v0=0.0, seed=k + 1)
            obs, _ = env.reset(options={"state": s0})


if __name__ == "__main__":
    main()

