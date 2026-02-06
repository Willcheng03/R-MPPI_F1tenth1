#!/usr/bin/env python3
import os, sys, json, time
import numpy as np
import zmq
import yaml
from argparse import Namespace

# ----------------------------
# Ensure we import f110_gym from your f1tenth_gym clone (NOT vendor)
# ----------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))

F1TENTH_GYM_REPO = "/home/f1tenthteam1/src/f1tenth_gym"
GYM_DIR = os.path.join(F1TENTH_GYM_REPO, "gym")

# strip vendor if it appears in sys.path
sys.path = [p for p in sys.path if "/src/vendor" not in p]

# ensure f1tenth_gym gym/ is first
if GYM_DIR not in sys.path:
    sys.path.insert(0, GYM_DIR)

import gym
from f110_gym.envs.base_classes import Integrator


def normalize_reset(out):
    if isinstance(out, tuple):
        if len(out) == 2 and isinstance(out[1], dict):
            return out[0], out[1]
        if len(out) >= 1 and isinstance(out[0], tuple):
            inner = out[0]
            if len(inner) == 2 and isinstance(inner[1], dict):
                return inner[0], inner[1]
            return inner[0], {}
        return out[0], {}
    return out, {}


def normalize_step(out):
    if not isinstance(out, tuple):
        raise RuntimeError(f"Unexpected env.step() output type: {type(out)}")

    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return obs, float(reward), done, info

    if len(out) == 4:
        obs, reward, done, info = out
        return obs, float(reward), bool(done), info

    raise RuntimeError(f"Unexpected env.step() tuple length: {len(out)}")


def map_yaml_to_gym_args(map_yaml_path: str):
    """
    ROS map yaml contains: image: <file>, etc.
    f110_gym wants: map=<STEM>, map_ext=<'.png'>
    where STEM is the full path WITHOUT extension.
    """
    map_yaml_path = os.path.abspath(map_yaml_path)
    with open(map_yaml_path, "r") as f:
        y = yaml.safe_load(f)

    img_rel = y["image"]
    base_dir = os.path.dirname(map_yaml_path)
    img_path = img_rel if os.path.isabs(img_rel) else os.path.join(base_dir, img_rel)
    img_path = os.path.abspath(img_path)

    stem, ext = os.path.splitext(img_path)
    if not ext:
        raise ValueError(f"Map image has no extension in yaml: {img_path}")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Map image not found: {img_path}")

    return stem, ext


def main():
    # ----------------------------
    # ZMQ REQ client -> connect to server
    # ----------------------------
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.connect("tcp://127.0.0.1:5555")

    sock.send_string(json.dumps({"type": "ping"}))
    print("[gym_client] ping ->", sock.recv_string())

    # ----------------------------
    # Choose map (built-in) via env var
    # ----------------------------
    # Default: vegas map yaml
    map_yaml = os.environ.get(
        "F1TENTH_GYM_MAP_YAML",
        "/home/f1tenthteam1/src/f1tenth_gym/gym/f110_gym/envs/maps/vegas.yaml",
    )

    map_stem, map_ext = map_yaml_to_gym_args(map_yaml)
    print("[gym_client] using map_yaml:", map_yaml)
    print("[gym_client] map stem:", map_stem)
    print("[gym_client] map ext:", map_ext)

    # ----------------------------
    # Start pose
    # ----------------------------
    # If you want vegas-specific spawn, set env vars.
    sx = float(os.environ.get("F1TENTH_SX", "0.0"))
    sy = float(os.environ.get("F1TENTH_SY", "0.0"))
    stheta = float(os.environ.get("F1TENTH_STHETA", "0.0"))

    # NOTE: your old example_map spawn came from config_example_map.yaml.
    # On vegas, (0,0,0) might be in a wall; if the car immediately crashes,
    # set F1TENTH_SX/F1TENTH_SY/F1TENTH_STHETA to a valid free-space pose.

    # ----------------------------
    # Create env
    # ----------------------------
    dt = 0.01
    env = gym.make(
        "f110_gym:f110-v0",
        map=map_stem,        # full path stem (no ext)
        map_ext=map_ext,     # ".png" or ".pgm"
        num_agents=1,
        timestep=dt,
        integrator=Integrator.RK4,
        disable_env_checker=True,
    )

    poses0 = np.array([[sx, sy, stheta]], dtype=np.float32)
    obs, info = normalize_reset(env.reset(poses=poses0))

    if not isinstance(obs, dict):
        raise RuntimeError(f"Expected dict obs from reset(), got {type(obs)}: {obs}")

    print("[gym_client] reset OK; obs keys:", list(obs.keys()))
    print("[gym_client] start pose:", poses0.tolist())

    # LiDAR metadata (assume symmetric FOV)
    default_angle_min = -2.35619449  # -135 deg
    default_angle_max =  2.35619449  # +135 deg
    default_max_range = 10.0

    # ----------------------------
    # Loop
    # ----------------------------
    for k in range(300):
        x = float(obs["poses_x"][0])
        y = float(obs["poses_y"][0])
        yaw = float(obs["poses_theta"][0])
        v = float(obs["linear_vels_x"][0]) if "linear_vels_x" in obs else 0.0

        scan = None
        if "scans" in obs and obs["scans"] is not None:
            try:
                scan = obs["scans"][0].astype(np.float32).tolist()
            except Exception:
                scan = None

        req = {
            "type": "step",
            "pose": [x, y, yaw],
            "speed": v,
            "scan": scan,
            "scan_angle_min": default_angle_min,
            "scan_angle_max": default_angle_max,
            "scan_max_range": default_max_range,
        }

        sock.send_string(json.dumps(req))
        resp = json.loads(sock.recv_string())

        if not resp.get("ok", False):
            raise RuntimeError(f"Server error: {resp}")

        steer, speed_cmd = resp["action"]
        action = np.array([[float(steer), float(speed_cmd)]], dtype=np.float32)

        obs, reward, done, info = normalize_step(env.step(action))

        if k % 25 == 0:
            dt_ms = resp.get("dt_ms", -1.0)
            dbg = resp.get("debug", {})
            mode = dbg.get("mode", "unknown")
            print(f"[gym_client] k={k} mode={mode} steer={steer:.3f} speed={speed_cmd:.3f} done={done} dt_ms={dt_ms:.2f}")

        if done:
            obs, info = normalize_reset(env.reset(poses=poses0))

        time.sleep(dt)

    env.close()
    print("[gym_client] finished")


if __name__ == "__main__":
    main()
