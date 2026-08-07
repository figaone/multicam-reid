"""
Turn a MapAnything reconstruction (per-camera .npz from
run_mapanything_reconstruct.py) into a shared ground frame our pipeline can use.

Method:
  1. Pool the reconstructed 3D points from all cameras (they share ONE metric
     world frame) and RANSAC-fit the dominant ground plane.
  2. Define a single 2D coordinate system ON that plane (origin + two in-plane
     axes) -- this is the shared bird's-eye frame.
  3. For each camera, fit a homography  full-res image pixel -> plane (x, y)
     directly from the pixel<->3D-point correspondences it already provides.

Output: a homography-type ground_frame JSON (same shape the team homographies
use), consumable by scripts/compare_shared_frames.py and the matcher.

Run from Code/ (project .venv; needs only numpy + opencv):
    python scripts/mapanything_to_ground_frame.py --recon-dir /path/to/recon \
        --out ground_frame_mapanything.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np


def fit_plane_ransac(pts, threshold, iters=1000, seed=0):
    """Return (normal unit vec, point_on_plane, inlier_mask) for the dominant plane."""
    rng = np.random.default_rng(seed)
    n = len(pts)
    best_inliers = np.zeros(n, dtype=bool)
    best = (np.array([0, 0, 1.0]), pts.mean(axis=0))
    for _ in range(iters):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = pts[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        dist = np.abs((pts - p0) @ normal)
        inliers = dist < threshold
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best = (normal, p0)
    # refit on inliers via SVD (total least squares)
    inl = pts[best_inliers]
    centroid = inl.mean(axis=0)
    _, _, Vt = np.linalg.svd(inl - centroid, full_matrices=False)
    normal = Vt[-1]
    normal = normal / np.linalg.norm(normal)
    return normal, centroid, best_inliers


def plane_basis(normal):
    """Two orthonormal in-plane axes for a plane with the given normal."""
    a = np.array([1.0, 0.0, 0.0])
    if abs(normal @ a) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = a - (a @ normal) * normal
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def load_recon(path):
    d = np.load(path, allow_pickle=True)
    return {
        "pts3d": d["pts3d"].astype(np.float64),
        "mask": d["mask"].astype(bool),
        "recon_hw": d["recon_hw"],
        "full_wh": d["full_wh"],
        "cam_num": int(d["cam_num"][0]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plane-thr-frac", type=float, default=0.01,
                    help="Plane inlier threshold as a fraction of scene extent.")
    ap.add_argument("--lower-frac", type=float, default=0.45,
                    help="Only use pixels in the lower this-fraction of each image "
                         "for the ground plane (road is usually low in frame).")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.recon_dir, "camera_*_recon.npz")))
    if not files:
        raise SystemExit(f"No camera_*_recon.npz in {args.recon_dir}")
    recons = [load_recon(f) for f in files]

    # pooled ground-candidate points (lower part of each image, valid only)
    pooled = []
    for r in recons:
        hr, wr = int(r["recon_hw"][0]), int(r["recon_hw"][1])
        m = r["mask"].copy()
        row_cut = int((1.0 - args.lower_frac) * hr)
        m[:row_cut, :] = False
        pooled.append(r["pts3d"][m])
    allpts = np.vstack(pooled)
    extent = np.linalg.norm(allpts.max(axis=0) - allpts.min(axis=0))
    thr = args.plane_thr_frac * extent
    print(f"Pooled {len(allpts)} ground-candidate points, scene extent {extent:.2f}, "
          f"plane threshold {thr:.3f}")

    normal, origin, _ = fit_plane_ransac(allpts, thr)
    u, v = plane_basis(normal)
    print(f"Ground plane normal {normal.round(3)}")

    cameras = {}
    for r, f in zip(recons, files):
        hr, wr = int(r["recon_hw"][0]), int(r["recon_hw"][1])
        full_w, full_h = int(r["full_wh"][0]), int(r["full_wh"][1])
        m = r["mask"].copy()
        row_cut = int((1.0 - args.lower_frac) * hr)
        m[:row_cut, :] = False
        # near-plane inliers for THIS camera
        pts = r["pts3d"][m]
        d = np.abs((pts - origin) @ normal)
        keep = d < thr
        pts = pts[keep]
        rows, cols = np.where(m)
        rows, cols = rows[keep], cols[keep]
        if len(pts) < 20:
            print(f"  camera_{r['cam_num']}: too few ground inliers ({len(pts)}), skipping")
            continue

        # plane (x, y) coords
        rel = pts - origin
        plane_xy = np.stack([rel @ u, rel @ v], axis=1)
        # full-res pixel coords
        sx, sy = full_w / wr, full_h / hr
        px = np.stack([cols * sx, rows * sy], axis=1)

        H, inl = cv2.findHomography(px.astype(np.float64), plane_xy.astype(np.float64),
                                    method=cv2.RANSAC, ransacReprojThreshold=thr)
        if H is None:
            print(f"  camera_{r['cam_num']}: homography fit failed, skipping")
            continue
        inl_n = int(inl.sum()) if inl is not None else 0
        cameras[str(r["cam_num"])] = {
            "H": H.tolist(),
            "width": full_w,
            "height": full_h,
            "inliers": inl_n,
        }
        print(f"  camera_{r['cam_num']}: fit homography from {len(pts)} pts, {inl_n} inliers")

    payload = {
        "version": 1,
        "type": "homography",
        "reference": "mapanything_ground_plane",
        "cameras": cameras,
        "meta": {"source": "mapanything", "recon_dir": str(args.recon_dir)},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(payload, fp, indent=2)
    print(f"Wrote {args.out} ({len(cameras)} cameras)")


if __name__ == "__main__":
    main()
