import argparse
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image
import yaml


def draw_disk(img, cx, cy, r, value):
    h, w = img.shape
    y0 = max(0, int(cy - r))
    y1 = min(h, int(cy + r + 1))
    x0 = max(0, int(cx - r))
    x1 = min(w, int(cx + r + 1))
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    img[y0:y1, x0:x1][mask] = value


def place_blocks_border_corner(
    img,
    cx,
    cy,
    rin_free_px,
    rout_free_px,
    n_blocks,
    block_px_min,
    block_px_max,
    seed=0,
    which_border="both",   # "inner" | "outer" | "both"
    corner_bias=True,
    corner_jitter_deg=12.0,
    border_margin_px=8.0,
):
    rng = random.Random(seed)
    h, w = img.shape

    # 4 diagonal "corners"
    if corner_bias:
        base_angles = [math.radians(a) for a in (45, 135, 225, 315)]
        anchors = [base_angles[i % 4] for i in range(n_blocks)]
        rng.shuffle(anchors)
        jitter = math.radians(corner_jitter_deg)
        anchors = [a + rng.uniform(-jitter, jitter) for a in anchors]
    else:
        anchors = [rng.uniform(0, 2 * math.pi) for _ in range(n_blocks)]

    def pick_border_radius():
        opts = []
        if which_border in ("inner", "both"):
            opts.append("inner")
        if which_border in ("outer", "both"):
            opts.append("outer")
        side = rng.choice(opts) if opts else "outer"
        if side == "inner":
            return rin_free_px + border_margin_px
        else:
            return rout_free_px - border_margin_px

    placed = 0
    tries = 0
    max_tries = max(400, n_blocks * 500)

    while placed < n_blocks and tries < max_tries:
        tries += 1
        ang = anchors[placed] if placed < len(anchors) else rng.uniform(0, 2 * math.pi)
        r = pick_border_radius()

        x = int(cx + r * math.cos(ang))
        y = int(cy + r * math.sin(ang))
        if not (0 <= x < w and 0 <= y < h):
            continue

        # pick big sizes
        bw = rng.randint(max(block_px_min, int(0.7 * block_px_max)), block_px_max)
        bh = rng.randint(max(block_px_min, int(0.7 * block_px_max)), block_px_max)

        x0 = max(0, x - bw // 2)
        x1 = min(w, x0 + bw)
        y0 = max(0, y - bh // 2)
        y1 = min(h, y0 + bh)

        patch = img[y0:y1, x0:x1]
        if patch.size == 0:
            continue

        # Mostly free pixels (white)
        if np.mean(patch) < 220:
            continue

        # ensure block sits in drivable ring (avoid crossing walls too much)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        # require at least some pixels inside free ring
        inside_free = np.any((rr >= rin_free_px) & (rr <= rout_free_px))
        if not inside_free:
            continue

        img[y0:y1, x0:x1] = 0
        placed += 1

    return placed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_stem", type=str, default="maps/donut_track",
                    help="Output path without extension, e.g. maps/donut_track")
    ap.add_argument("--size_px", type=int, default=900, help="Image size in pixels (square)")
    ap.add_argument("--resolution", type=float, default=0.05, help="meters per pixel")

    # Make the ring fill more of the image by increasing r_outer_m
    ap.add_argument("--r_inner_m", type=float, default=12.0, help="Inner radius of the donut (meters)")
    ap.add_argument("--r_outer_m", type=float, default=20.0, help="Outer radius of the donut (meters)")
    ap.add_argument("--wall_thickness_m", type=float, default=0.45, help="Wall thickness (meters)")

    # Fewer, bigger blocks that hug the borders and sit near 'corners'
    ap.add_argument("--n_blocks", type=int, default=8)
    ap.add_argument("--block_min_m", type=float, default=1.0)
    ap.add_argument("--block_max_m", type=float, default=2.0)
    ap.add_argument("--which_border", type=str, default="both", choices=["inner", "outer", "both"])
    ap.add_argument("--corner_bias", action="store_true")
    ap.add_argument("--no_corner_bias", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    corner_bias = True
    if args.no_corner_bias:
        corner_bias = False
    if args.corner_bias:
        corner_bias = True

    out_stem = Path(args.out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    N = int(args.size_px)
    res = float(args.resolution)

    img = np.zeros((N, N), dtype=np.uint8)
    cx = cy = (N - 1) / 2.0

    rin_px = args.r_inner_m / res
    rout_px = args.r_outer_m / res
    wall_px = max(1.0, args.wall_thickness_m / res)

    # Define the FREE region as a ring between (rin+wall) and (rout-wall)
    rin_free_px = rin_px + wall_px
    rout_free_px = rout_px - wall_px

    # carve the free ring
    draw_disk(img, cx, cy, rout_free_px, 255)
    draw_disk(img, cx, cy, rin_free_px, 0)

    # Place blocks (obstacles) on the free ring, near borders
    block_min_px = max(4, int(args.block_min_m / res))
    block_max_px = max(block_min_px, int(args.block_max_m / res))

    placed = place_blocks_border_corner(
        img,
        cx,
        cy,
        rin_free_px,
        rout_free_px,
        args.n_blocks,
        block_min_px,
        block_max_px,
        seed=args.seed,
        which_border=args.which_border,
        corner_bias=corner_bias,
        corner_jitter_deg=10.0,
        border_margin_px=max(10.0, wall_px + 4.0),
    )
    print(f"Placed {placed}/{args.n_blocks} border blocks")

    png_path = out_stem.with_suffix(".png")
    Image.fromarray(img).save(png_path)
    print("Wrote", png_path)

    map_size_m = N * res
    origin = [-map_size_m / 2.0, -map_size_m / 2.0, 0.0]
    meta = {
        "image": str(png_path.name),
        "resolution": res,
        "origin": origin,
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
    }
    yaml_path = out_stem.with_suffix(".yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    print("Wrote", yaml_path)


if __name__ == "__main__":
    main()
