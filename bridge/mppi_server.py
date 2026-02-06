#!/usr/bin/env python3
import os
import json
import time
import math
import zmq
import numpy as np

# Optional deps for map loading + distance transform
import yaml
from PIL import Image

try:
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

try:
    import rmppi_cuda
    HAVE_RMPPI = True
except Exception as e:
    HAVE_RMPPI = False
    RMPPI_ERR = str(e)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def contiguous_regions(mask: np.ndarray):
    if mask.size == 0:
        return []
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[splits + 1]]
    ends = np.r_[idx[splits], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def ftg_action(
    scan_list,
    angle_min=-2.35619449,
    angle_max=2.35619449,
    max_range=10.0,
    max_steer=0.418,
):
    scan = np.asarray(scan_list, dtype=np.float32).copy()
    n = scan.size
    if n < 10:
        return 0.0, 1.0, {"note": "scan too short", "n": int(n)}

    scan[~np.isfinite(scan)] = max_range
    scan = np.clip(scan, 0.0, max_range)

    angle_min = float(angle_min)
    angle_max = float(angle_max)
    ang_inc = (angle_max - angle_min) / float(max(n - 1, 1))

    mid = n // 2
    half_window = int(0.5 * n * (math.radians(90) / math.radians(135)))
    half_window = int(clamp(half_window, int(0.2 * n), int(0.45 * n)))
    i0 = max(0, mid - half_window)
    i1 = min(n, mid + half_window + 1)

    front = scan[i0:i1]
    if front.size < 10:
        return 0.0, 1.0, {"note": "front window too short", "n": int(n)}

    closest_i = int(np.argmin(front))
    closest_d = float(front[closest_i])

    bubble_m = 0.6
    bubble_idx = int(max(3, bubble_m / max(closest_d, 0.05)))
    bubbled = front.copy()
    b0 = max(0, closest_i - bubble_idx)
    b1 = min(front.size, closest_i + bubble_idx + 1)
    bubbled[b0:b1] = 0.0

    free_thresh = 1.2
    free_mask = bubbled > free_thresh
    gaps = contiguous_regions(free_mask)

    if not gaps:
        steer = 0.25 if closest_i < (front.size // 2) else -0.25
        steer = float(clamp(steer, -max_steer, max_steer))
        speed = 0.6
        return steer, speed, {
            "note": "no gap; evade",
            "closest_d": closest_d,
            "closest_i": int(closest_i),
        }

    gap = max(gaps, key=lambda ab: (ab[1] - ab[0]))
    g0, g1 = gap
    gap_slice = bubbled[g0:g1 + 1]
    aim_local = int(np.argmax(gap_slice))
    aim_i = g0 + aim_local

    global_i = i0 + aim_i
    aim_angle = angle_min + global_i * ang_inc

    steer = float(clamp(1.1 * aim_angle, -max_steer, max_steer))

    center_band = front[(front.size // 2 - 10):(front.size // 2 + 11)]
    center_d = float(np.min(center_band)) if center_band.size else float(np.min(front))

    if center_d > 4.0:
        speed = 2.5
    elif center_d > 3.0:
        speed = 2.0
    elif center_d > 2.0:
        speed = 1.5
    elif center_d > 1.5:
        speed = 1.0
    else:
        speed = 0.6

    return steer, float(speed), {
        "closest_d": closest_d,
        "bubble_idx": int(bubble_idx),
        "gap": [int(g0), int(g1)],
        "aim_angle": float(aim_angle),
        "center_d": float(center_d),
        "n": int(n),
    }


# -----------------------------
# Map loading (ROS map_server style YAML + PNG)
# -----------------------------
def load_ros_map(map_yaml_path: str):
    """
    Reads ROS-style map yaml:
      image: <png>
      resolution: <m/pixel>
      origin: [x, y, yaw]
      negate: 0/1
      occupied_thresh, free_thresh
    Returns:
      occ (H,W) uint8 where 1=occupied,0=free
      dist_m (H,W) float32 distance to nearest obstacle in meters
      meta dict
    """
    map_yaml_path = os.path.abspath(map_yaml_path)
    with open(map_yaml_path, "r") as f:
        y = yaml.safe_load(f)

    img_rel = y["image"]
    res = float(y["resolution"])
    origin = y.get("origin", [0.0, 0.0, 0.0])
    negate = int(y.get("negate", 0))
    occ_th = float(y.get("occupied_thresh", 0.65))
    free_th = float(y.get("free_thresh", 0.196))

    base_dir = os.path.dirname(map_yaml_path)
    img_path = img_rel if os.path.isabs(img_rel) else os.path.join(base_dir, img_rel)
    img_path = os.path.abspath(img_path)

    im = Image.open(img_path).convert("L")
    img = np.array(im, dtype=np.uint8)  # 0..255

    # Convert to probability-like value in [0,1]
    # ROS map convention:
    #   If negate==0: 0=free (white), 1=occupied (black) after inversion
    #   If negate==1: 0=occupied, 1=free (rare)
    if negate == 0:
        p_occ = 1.0 - (img.astype(np.float32) / 255.0)
    else:
        p_occ = (img.astype(np.float32) / 255.0)

    # Decide occupancy. Unknown handling omitted; treat mid-gray as free-ish.
    occ = (p_occ >= occ_th).astype(np.uint8)  # 1 occupied
    free = (p_occ <= free_th).astype(np.uint8)

    # For distance transform we want free space mask (1=free)
    # Use free if provided; otherwise use ~occ
    free_mask = free if np.any(free) else (1 - occ)

    # Distance to obstacles: distanceTransform expects 0=obstacle, >0=free
    dt_input = (free_mask > 0).astype(np.uint8)
    if not HAVE_CV2:
        raise RuntimeError("cv2 not available; install python3-opencv to use distance transform")
    dist_px = cv2.distanceTransform(dt_input, distanceType=cv2.DIST_L2, maskSize=5)
    dist_m = (dist_px * res).astype(np.float32)

    meta = {
        "map_yaml": map_yaml_path,
        "map_image": img_path,
        "resolution": res,
        "origin": origin,  # [x0,y0,yaw0]
        "H": int(img.shape[0]),
        "W": int(img.shape[1]),
    }
    return occ, dist_m, meta


def world_to_pixel(x, y, meta):
    """
    ROS map_server origin: origin=[x0,y0,yaw] is the world coordinate of the map's bottom-left pixel.
    Image array row 0 is top, so y needs inversion.
    """
    res = meta["resolution"]
    x0, y0, _ = meta["origin"]
    W = meta["W"]
    H = meta["H"]

    px = (x - x0) / res
    py = (y - y0) / res

    # Convert to image coordinates (row-major, top-left origin)
    u = int(round(px))
    v = int(round((H - 1) - py))  # invert y
    return u, v


def sample_dist(dist_m, u, v):
    H, W = dist_m.shape
    if u < 0 or u >= W or v < 0 or v >= H:
        return 0.0
    return float(dist_m[v, u])


def pick_map_goal(x, y, yaw, dist_m, meta):
    """
    2A: pick a local goal point by maximizing clearance (distance transform),
    with a bias for forward progress.
    Returns: (xg, yg, yaw_ref, v_ref, dbg)
    """
    # Search fan
    angles = np.deg2rad(np.linspace(-70.0, 70.0, 29)).astype(np.float32)
    radii = np.array([0.8, 1.2, 1.6, 2.0], dtype=np.float32)

    best = None
    best_score = -1e9
    best_clear = 0.0

    for r in radii:
        for a in angles:
            th = yaw + float(a)
            cx = x + float(r * math.cos(th))
            cy = y + float(r * math.sin(th))

            u, v = world_to_pixel(cx, cy, meta)
            clear = sample_dist(dist_m, u, v)  # meters

            # Score: prioritize clearance; add forward projection
            forward = float(r * math.cos(a))  # along current heading
            score = 3.0 * clear + 0.3 * forward

            if score > best_score:
                best_score = score
                best = (cx, cy)
                best_clear = clear

    if best is None:
        # fallback: straight ahead small step
        cx = x + 0.8 * math.cos(yaw)
        cy = y + 0.8 * math.sin(yaw)
        best = (cx, cy)
        best_clear = 0.0

    xg, yg = best
    yaw_ref = math.atan2(yg - y, xg - x)

    # Speed target based on clearance (more clearance -> faster)
    # Tune these for your map scale.
    if best_clear > 1.2:
        v_ref = 2.5
    elif best_clear > 0.8:
        v_ref = 2.0
    elif best_clear > 0.5:
        v_ref = 1.5
    elif best_clear > 0.35:
        v_ref = 1.0
    else:
        v_ref = 0.6

    dbg = {
        "xg": float(xg),
        "yg": float(yg),
        "yaw_ref": float(yaw_ref),
        "v_ref": float(v_ref),
        "clear_m": float(best_clear),
        "score": float(best_score),
    }
    return xg, yg, yaw_ref, v_ref, dbg


def main():
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind("tcp://127.0.0.1:5555")
    print("[mppi_server] REP bound on tcp://127.0.0.1:5555")

    if HAVE_RMPPI:
        exports = [x for x in dir(rmppi_cuda) if not x.startswith("_")]
        print("[mppi_server] rmppi_cuda import OK; exports:", exports)
    else:
        print("[mppi_server] rmppi_cuda import FAILED:", RMPPI_ERR)

    max_steer = 0.418
    max_speed = 3.0
    min_speed = 0.0

    # --- Load map (2A) ---
    # You can point this at example_map.yaml (ROS map yaml), NOT config_example_map.yaml.
    map_yaml = os.environ.get("F1TENTH_MAP_YAML", "")
    occ = dist_m = meta = None

    if map_yaml:
        try:
            occ, dist_m, meta = load_ros_map(map_yaml)
            print("[mppi_server] Loaded map:", meta)
        except Exception as e:
            print("[mppi_server] Map load failed; will still run (FTG/MPPI without map-goal). Error:", str(e))
            occ = dist_m = meta = None
    else:
        print("[mppi_server] F1TENTH_MAP_YAML not set. For 2A, export it to your map yaml file.")

    # --- MPPI init (kinematic bicycle wrapper from your new binding) ---
    use_mppi = HAVE_RMPPI and hasattr(rmppi_cuda, "F1TenthKinematicMPPI")
    mppi = None
    q_coeffs = np.array([25.0, 25.0, 6.0, 2.0], dtype=np.float32)  # [x,y,yaw,v]

    if use_mppi:
        dt = 0.01  # matches your gym_client timestep :contentReference[oaicite:2]{index=2}
        lam = 2.0
        alpha = 1.0
        max_iter = 1
        wheelbase = 0.33
        tau_speed = 0.25
        mppi = rmppi_cuda.F1TenthKinematicMPPI(dt, lam, alpha, max_iter, wheelbase, tau_speed, max_speed)
        print("[mppi_server] Using MPPI: F1TenthKinematicMPPI")
    else:
        print("[mppi_server] MPPI not available; using FTG fallback only")

    while True:
        msg = sock.recv()
        t0 = time.time()

        try:
            req = json.loads(msg.decode("utf-8"))

            if req.get("type") == "ping":
                sock.send_string(json.dumps({
                    "type": "pong",
                    "ok": True,
                    "have_rmppi": HAVE_RMPPI,
                    "have_map": bool(dist_m is not None),
                }))
                continue

            if req.get("type") != "step":
                sock.send_string(json.dumps({"ok": False, "err": "unknown message type"}))
                continue

            scan = req.get("scan", None)
            angle_min = req.get("scan_angle_min", -2.35619449)
            angle_max = req.get("scan_angle_max", 2.35619449)
            max_range = req.get("scan_max_range", 10.0)

            pose = req.get("pose", None)   # [x,y,yaw] :contentReference[oaicite:3]{index=3}
            speed = float(req.get("speed", 0.0))

            steer = 0.0
            speed_cmd = 1.0
            dbg = {}

            # -----------------------------
            # 2A: Map-goal + MPPI
            # -----------------------------
            if mppi is not None and dist_m is not None and meta is not None and pose is not None and len(pose) >= 3:
                x = float(pose[0])
                y = float(pose[1])
                yaw = float(pose[2])

                xg, yg, yaw_ref, v_ref, goal_dbg = pick_map_goal(x, y, yaw, dist_m, meta)

                goal = np.array([xg, yg, yaw_ref, v_ref], dtype=np.float32)
                state = np.array([x, y, yaw, speed], dtype=np.float32)

                mppi.slide(1)
                mppi.set_goal(goal, q_coeffs)
                u = mppi.compute_control(state)  # [steer, speed_cmd]

                steer = float(clamp(float(u[0]), -max_steer, max_steer))
                speed_cmd = float(clamp(float(u[1]), min_speed, max_speed))

                dbg = {"mode": "mppi_map_goal", **goal_dbg, "baseline_cost": float(mppi.baseline_cost())}

            # -----------------------------
            # Fallback: FTG (LiDAR)
            # -----------------------------
            else:
                if scan is None:
                    steer, speed_cmd, dbg = 0.0, 1.0, {"mode": "fallback", "note": "no scan"}
                else:
                    steer, speed_cmd, dbg = ftg_action(
                        scan,
                        angle_min=angle_min,
                        angle_max=angle_max,
                        max_range=max_range,
                        max_steer=max_steer,
                    )
                steer = float(clamp(steer, -max_steer, max_steer))
                speed_cmd = float(clamp(speed_cmd, min_speed, max_speed))

            resp = {
                "ok": True,
                "action": [steer, speed_cmd],
                "dt_ms": (time.time() - t0) * 1000.0,
                "debug": dbg,
            }
            sock.send_string(json.dumps(resp))

        except Exception as e:
            sock.send_string(json.dumps({"ok": False, "err": str(e)}))


if __name__ == "__main__":
    main()
