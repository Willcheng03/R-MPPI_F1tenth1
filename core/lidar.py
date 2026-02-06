import math
import numpy as np

# Optional numba acceleration
try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):  # no-op decorator fallback
        def wrap(f):
            return f
        return wrap


@njit(cache=True, fastmath=True)
def _raycast_numba(
    x: float, y: float, yaw: float,
    occ: np.ndarray,
    res: float, x0: float, y0: float,
    H: int, W: int,
    cos_body: np.ndarray, sin_body: np.ndarray,
    max_range_m: float,
    step_m: float,
) -> np.ndarray:
    """
    Numba-accelerated raymarch against occupancy grid.
    occ is (H,W) uint8 where 1=occupied, 0=free.
    """
    n_beams = cos_body.shape[0]
    ranges = np.empty((n_beams,), dtype=np.float32)

    cy = math.cos(yaw)
    sy = math.sin(yaw)

    # helper: world->pixel with rounding, inlined for numba
    # u = round((x-x0)/res)
    # v = round((H-1) - (y-y0)/res)
    for i in range(n_beams):
        # rotate body ray direction into world
        ca = cos_body[i]
        sa = sin_body[i]
        dx = cy * ca - sy * sa
        dy = sy * ca + cy * sa

        r = 0.0
        hit = 0

        while r < max_range_m:
            px = x + r * dx
            py = y + r * dy

            fx = (px - x0) / res
            fy = (py - y0) / res

            # round() in numba returns float, so implement nearest-int
            u = int(fx + 0.5) if fx >= 0.0 else int(fx - 0.5)
            v = int((H - 1) - (fy + 0.5)) if fy >= 0.0 else int((H - 1) - (fy - 0.5))

            # bounds / occupancy
            if u < 0 or u >= W or v < 0 or v >= H:
                hit = 1
                break
            if occ[v, u] == 1:
                hit = 1
                break

            r += step_m

        ranges[i] = np.float32(r if hit == 1 else max_range_m)

    return ranges


def raycast_grid(
    state,
    occ,
    meta,
    n_beams=360,
    fov_rad=math.radians(270.0),
    max_range_m=10.0,
    step_m=0.05,
):
    """
    Simple 2D LiDAR raycasting into an occupancy grid.
    Returns ranges (n_beams,) float32.

    Uses numba acceleration if available; falls back to pure Python otherwise.
    """
    x, y, yaw, _ = map(float, state)

    res = float(meta["resolution"])
    x0, y0, _ = meta["origin"]
    H = int(meta["H"])
    W = int(meta["W"])

    # precompute body-frame beam directions (only depends on fov + n_beams)
    angles = (-0.5 * float(fov_rad)) + np.linspace(0.0, float(fov_rad), int(n_beams), endpoint=False, dtype=np.float32)
    cos_body = np.cos(angles).astype(np.float32)
    sin_body = np.sin(angles).astype(np.float32)

    # Ensure occ is contiguous uint8 for numba speed
    occ_u8 = np.ascontiguousarray(occ.astype(np.uint8, copy=False))

    if _HAVE_NUMBA:
        return _raycast_numba(
            float(x), float(y), float(yaw),
            occ_u8,
            float(res), float(x0), float(y0),
            int(H), int(W),
            cos_body, sin_body,
            float(max_range_m),
            float(step_m),
        )

    # Fallback pure Python (should rarely be used once numba is installed)
    ranges = np.empty((int(n_beams),), dtype=np.float32)
    cy = math.cos(float(yaw))
    sy = math.sin(float(yaw))
    for i in range(int(n_beams)):
        dx = cy * float(cos_body[i]) - sy * float(sin_body[i])
        dy = sy * float(cos_body[i]) + cy * float(sin_body[i])
        r = 0.0
        hit = False
        while r < float(max_range_m):
            px = x + r * dx
            py = y + r * dy
            fx = (px - x0) / res
            fy = (py - y0) / res
            u = int(round(fx))
            v = int(round((H - 1) - fy))
            if u < 0 or u >= W or v < 0 or v >= H or occ_u8[v, u] == 1:
                hit = True
                break
            r += float(step_m)
        ranges[i] = np.float32(r if hit else float(max_range_m))
    return ranges
