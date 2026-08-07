"""
Visualize a MapAnything reconstruction (per-camera .npz from
run_mapanything_reconstruct.py) as a colored 3D point cloud.

Outputs (into the recon dir):
  * reconstruction.ply         -- open in any 3D viewer (MeshLab, Blender, VS Code)
  * recon_view_top.png         -- top-down (bird's-eye)
  * recon_view_oblique.png     -- angled 3D view
  * recon_view_side.png        -- side view
Camera positions are marked as large dots.

Run from Code/ (project .venv):
    python scripts/visualize_reconstruction.py --recon-dir <dir>
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mapanything_to_ground_frame import fit_plane_ransac, plane_basis  # noqa: E402

CAM_COLORS = {1: (60, 120, 240), 2: (60, 200, 120), 3: (240, 90, 90)}  # RGB


def load_recon(path):
    d = np.load(path, allow_pickle=True)
    pts = d["pts3d"].astype(np.float64)          # (H, W, 3)
    mask = d["mask"].astype(bool)
    hr, wr = pts.shape[:2]
    img = cv2.imread(str(d["source_image"]))     # BGR full-res
    img = cv2.resize(img, (wr, hr))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pose = d["camera_pose"].astype(np.float64) if "camera_pose" in d else None
    cam_pos = pose[:3, 3] if pose is not None and pose.shape == (4, 4) else None
    return pts[mask], rgb[mask], int(d["cam_num"][0]), cam_pos


def write_ply(path, xyz, rgb):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(xyz, rgb):
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")


def render(xyz, rgb, cams, out_path, elev, azim, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb / 255.0, s=0.6, depthshade=False)
    for cam_num, pos in cams.items():
        if pos is None:
            continue
        col = np.array(CAM_COLORS[cam_num]) / 255.0
        ax.scatter([pos[0]], [pos[1]], [pos[2]], color=col, s=180, marker="^",
                   edgecolors="k", linewidths=1.2)
        ax.text(pos[0], pos[1], pos[2], f"  cam{cam_num}", color="k", fontsize=11, weight="bold")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)
    # equal aspect
    lo, hi = xyz.min(0), xyz.max(0)
    c = (lo + hi) / 2
    r = (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)
    print("wrote", out_path)


def render_birdseye(uvw, rgb, cams_uvw, out_path):
    """True top-down 2D scatter in ground-plane coords (u, v)."""
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.scatter(uvw[:, 0], uvw[:, 1], c=rgb / 255.0, s=0.8)
    for cam_num, pos in cams_uvw.items():
        if pos is None:
            continue
        col = np.array(CAM_COLORS[cam_num]) / 255.0
        ax.scatter([pos[0]], [pos[1]], color=col, s=220, marker="^",
                   edgecolors="k", linewidths=1.4, zorder=5)
        ax.annotate(f"cam{cam_num}", (pos[0], pos[1]), textcoords="offset points",
                    xytext=(8, 6), fontsize=12, weight="bold")
    ax.set_aspect("equal")
    ax.set_title("MapAnything reconstruction - true top-down (ground-plane frame)")
    ax.set_xlabel("ground X"); ax.set_ylabel("ground Y")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)
    print("wrote", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon-dir", required=True)
    ap.add_argument("--max-points", type=int, default=180000)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.recon_dir, "camera_*_recon.npz")))
    if not files:
        raise SystemExit(f"No camera_*_recon.npz in {args.recon_dir}")

    XYZ, RGB, cams = [], [], {}
    for f in files:
        xyz, rgb, cam_num, cam_pos = load_recon(f)
        XYZ.append(xyz); RGB.append(rgb); cams[cam_num] = cam_pos
        print(f"camera_{cam_num}: {len(xyz)} points")
    xyz = np.vstack(XYZ); rgb = np.vstack(RGB)

    # clip far outliers for a clean view
    lo = np.percentile(xyz, 1, axis=0); hi = np.percentile(xyz, 99, axis=0)
    keep = np.all((xyz >= lo) & (xyz <= hi), axis=1)
    xyz, rgb = xyz[keep], rgb[keep]

    # downsample
    if len(xyz) > args.max_points:
        idx = np.random.default_rng(0).choice(len(xyz), args.max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    print(f"visualizing {len(xyz)} points")

    d = Path(args.recon_dir)
    write_ply(d / "reconstruction.ply", xyz, rgb)
    print("wrote", d / "reconstruction.ply")

    # rotate into ground-plane frame so "up" is +Z (true bird's-eye)
    extent = np.linalg.norm(xyz.max(0) - xyz.min(0))
    normal, origin, _ = fit_plane_ransac(xyz, 0.01 * extent)
    u, v = plane_basis(normal)
    R = np.stack([u, v, normal], axis=0)          # rows = new axes
    uvw = (xyz - origin) @ R.T
    cams_uvw = {k: ((p - origin) @ R.T if p is not None else None) for k, p in cams.items()}

    render_birdseye(uvw, rgb, cams_uvw, d / "recon_view_top.png")
    render(uvw, rgb, cams_uvw, d / "recon_view_oblique.png", elev=28, azim=-70, title="MapAnything reconstruction - oblique 3D")
    render(uvw, rgb, cams_uvw, d / "recon_view_side.png", elev=6, azim=-90, title="MapAnything reconstruction - side (shows flat road + buildings)")


if __name__ == "__main__":
    main()
