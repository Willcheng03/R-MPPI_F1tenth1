import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from core.map import world_to_pixel

# --- Optional Numba acceleration -------------------------------------------------
# Numba helps for the *math* portion (ray endpoints + world->pixel). Rendering
# FPS is still often dominated by Matplotlib's draw time, so we also use a
# LineCollection to update all rays in one artist instead of N individual lines.
try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):
        def _wrap(f):
            return f
        return _wrap


@njit(cache=True)
def _world_to_pixel_xy(x: float, y: float, origin_x: float, origin_y: float, res: float, height: int):
    """Fast world->pixel conversion.

    Assumes a standard ROS-style occupancy grid meta:
      - origin is (origin_x, origin_y) in meters
      - resolution is meters/pixel
      - image is displayed with origin='upper'

    Returns (u, v) in *pixel coordinates* (floats).
    """
    u = (x - origin_x) / res
    v = (y - origin_y) / res
    # Convert to 'upper' image coords.
    vv = (height - 1) - v
    return u, vv


@njit(cache=True)
def _compute_ray_segments(
    x: float,
    y: float,
    yaw: float,
    scan: np.ndarray,
    scan_angles_body: np.ndarray,
    stride: int,
    origin_x: float,
    origin_y: float,
    res: float,
    height: int,
):
    """Compute ray line segments in pixel coords for a LineCollection."""
    n = scan.shape[0]
    m = (n + stride - 1) // stride
    segs = np.empty((m, 2, 2), dtype=np.float32)

    u0, v0 = _world_to_pixel_xy(x, y, origin_x, origin_y, res, height)
    k = 0
    for i in range(0, n, stride):
        ang = yaw + scan_angles_body[i]
        r = scan[i]
        px = x + r * math.cos(ang)
        py = y + r * math.sin(ang)
        u1, v1 = _world_to_pixel_xy(px, py, origin_x, origin_y, res, height)
        segs[k, 0, 0] = u0
        segs[k, 0, 1] = v0
        segs[k, 1, 0] = u1
        segs[k, 1, 1] = v1
        k += 1
    return segs


class MatplotlibRenderer:
    def __init__(self, occ, meta, car_radius_m=0.20, fov_rad=math.radians(270.0), n_beams=360, zoom_px=300):
        self.occ = occ
        self.meta = meta
        self.car_radius_m = float(car_radius_m)
        self.fov_rad = float(fov_rad)
        self.n_beams = int(n_beams)
        self.zoom_px = int(zoom_px)

        # interactive mode speeds up updates
        plt.ion()

        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect("equal", adjustable="box")

        # Draw map once
        self.map_im = self.ax.imshow(1 - self.occ, cmap="gray", origin="upper")

        # Precompute scan angles in the car frame once (len = n_beams)
        # We'll shift by yaw each render.
        self._scan_angles_body = (-0.5 * self.fov_rad) + np.linspace(
            0.0, self.fov_rad, self.n_beams, endpoint=False, dtype=np.float32
        )

        # Artists we update each frame
        self.car_circle = None
        self.heading_line = None

        # Rays: one bulk artist (fast) instead of N individual Line2D objects.
        self.rays_lc = LineCollection([], linewidths=0.5)
        self.ax.add_collection(self.rays_lc)

        # Cache common meta fields for fast world->pixel. If meta doesn't look
        # like a standard occupancy grid, we fall back to core.map.world_to_pixel.
        self._meta_res = float(self.meta.get("resolution", 1.0))
        origin = self.meta.get("origin", None)
        if isinstance(origin, (list, tuple, np.ndarray)) and len(origin) >= 2 and ("height" in self.meta):
            self._meta_origin_x = float(origin[0])
            self._meta_origin_y = float(origin[1])
            self._meta_height = int(self.meta["height"])
            self._fast_w2p = True
        else:
            self._meta_origin_x = 0.0
            self._meta_origin_y = 0.0
            self._meta_height = int(self.meta.get("height", 0) or 0)
            self._fast_w2p = False

        # Create initial artists with dummy state
        self._init_artists(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))

    def _init_artists(self, state):
        x, y, yaw, v = map(float, state)
        u, vv = world_to_pixel(x, y, self.meta)

        car_r_px = self.car_radius_m / float(self.meta["resolution"])

        self.car_circle = plt.Circle((u, vv), car_r_px, fill=False)
        self.ax.add_patch(self.car_circle)

        hx = u + car_r_px * math.cos(yaw)
        hy = vv + car_r_px * math.sin(yaw)
        (self.heading_line,) = self.ax.plot([u, hx], [vv, hy], linewidth=2)

        # No rays allocated yet; created on demand
        self.ax.set_xlim(u - self.zoom_px, u + self.zoom_px)
        self.ax.set_ylim(vv + self.zoom_px, vv - self.zoom_px)

    def render(self, state, scan=None, title="", show_rays=True, rays_stride=10):
        self.ax.set_title(title)

        x, y, yaw, v = map(float, state)
        u, vv = world_to_pixel(x, y, self.meta)

        car_r_px = self.car_radius_m / float(self.meta["resolution"])

        # Update car circle
        self.car_circle.center = (u, vv)
        self.car_circle.radius = car_r_px

        # Update heading line
        hx = u + car_r_px * math.cos(yaw)
        hy = vv + car_r_px * math.sin(yaw)
        self.heading_line.set_data([u, hx], [vv, hy])

        # Update rays
        self._update_rays(x, y, yaw, u, vv, scan, show_rays, rays_stride)

        # Zoom around car
        self.ax.set_xlim(u - self.zoom_px, u + self.zoom_px)
        self.ax.set_ylim(vv + self.zoom_px, vv - self.zoom_px)

        # HUD text (single text object is fastest; but keep it simple)
        # remove old text objects by reusing one:
        if not hasattr(self, "_hud"):
            self._hud = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes, va="top")
        self._hud.set_text(f"x={x:.2f} y={y:.2f} yaw={yaw:.2f} v={v:.2f}")

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def _update_rays(self, x, y, yaw, u, vv, scan, show_rays, rays_stride):
        # Hide rays if disabled
        if (scan is None) or (not show_rays):
            self.rays_lc.set_segments([])
            return

        scan = np.asarray(scan, dtype=np.float32)
        n = scan.shape[0]
        stride = max(1, int(rays_stride))

        # If scan beam count differs, recompute angles buffer
        if n != self.n_beams:
            self.n_beams = int(n)
            self._scan_angles_body = (-0.5 * self.fov_rad) + np.linspace(
                0.0, self.fov_rad, self.n_beams, endpoint=False, dtype=np.float32
            )

        # Compute ray segments (pixel coords) and update the LineCollection.
        if self._fast_w2p:
            segs = _compute_ray_segments(
                float(x), float(y), float(yaw), scan, self._scan_angles_body, stride,
                float(self._meta_origin_x), float(self._meta_origin_y), float(self._meta_res), int(self._meta_height)
            )
            # LineCollection expects float array segments; keep as numpy.
            self.rays_lc.set_segments(segs)
        else:
            # Conservative fallback: compute endpoints using the existing world_to_pixel.
            idx = np.arange(0, n, stride, dtype=np.int32)
            angles = yaw + self._scan_angles_body[idx]
            r = scan[idx]
            px = x + r * np.cos(angles)
            py = y + r * np.sin(angles)

            segs = []
            for k in range(idx.size):
                uu, v2 = world_to_pixel(float(px[k]), float(py[k]), self.meta)
                segs.append(((u, vv), (uu, v2)))
            self.rays_lc.set_segments(segs)
