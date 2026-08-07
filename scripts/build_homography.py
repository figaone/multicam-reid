"""
Validate the derived camera calibration.

For each camera it:
  1. grabs a sample frame from the segment video,
  2. undistorts it with the single-term radial model,
  3. warps the undistorted frame into its own bird's-eye ground plane,
  4. writes a side-by-side  original | undistorted | bird's-eye  panel.

Also saves the derived calibration (K, distortion, homographies) to
``<segment>/.reid/calibration.json``.

Run:
    python scripts/build_homography.py <segment_folder> <intrinsics.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multicam_reid.core.project import Project
from multicam_reid.core.calibration import (
    load_intrinsics,
    match_camera_id,
)

BEV_SIZE = 900  # output bird's-eye canvas (pixels, square)


def sample_frame(video: Path, index_ratio: float = 0.4) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * index_ratio))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise IOError(f"Could not read frame from {video}")
    return frame


def draw_ground_grid(img, calib, w, h, n_lines=11):
    """
    Project a regular ground-plane grid back onto the (undistorted) image.
    If the calibration is correct, one family of lines runs along the road
    (parallel to lane markings) and the other crosses it perpendicularly.
    """
    calib.compute_ground_homography()
    Hi2g = calib.H_img2ground
    H_g2i = calib.H_ground2img

    bc = Hi2g @ np.array([w / 2, h - 1, 1.0])
    near_sign = np.sign(bc[2]) or 1.0

    # Visible near-field ground extent.
    ys = np.linspace(0, h - 1, 60)
    xs = np.linspace(0, w - 1, 60)
    grid = np.array([(x, y) for y in ys for x in xs], dtype=np.float64)
    hom = np.hstack([grid, np.ones((len(grid), 1))])
    proj = (Hi2g @ hom.T).T
    keep = (np.sign(proj[:, 2]) == near_sign) & (np.abs(proj[:, 2]) > 0.02 * abs(bc[2]))
    g = proj[keep, :2] / proj[keep, 2, None]
    gmin = np.percentile(g, 3, axis=0)
    gmax = np.percentile(g, 90, axis=0)

    out = img.copy()

    def project_polyline(pts_ground, color):
        ph = np.hstack([pts_ground, np.ones((len(pts_ground), 1))])
        pi = (H_g2i @ ph.T).T
        good = pi[:, 2] * near_sign > 1e-9
        pix = pi[good, :2] / pi[good, 2, None]
        prev = None
        for (x, y) in pix:
            if 0 <= x < w and 0 <= y < h:
                cur = (int(x), int(y))
                if prev is not None:
                    cv2.line(out, prev, cur, color, 2, cv2.LINE_AA)
                prev = cur
            else:
                prev = None

    xr = np.linspace(gmin[0], gmax[0], n_lines)
    yr = np.linspace(gmin[1], gmax[1], n_lines)
    ys_fine = np.linspace(gmin[1], gmax[1], 200)
    xs_fine = np.linspace(gmin[0], gmax[0], 200)
    for xc in xr:  # lines of constant X (run along the Y/forward axis)
        project_polyline(np.column_stack([np.full_like(ys_fine, xc), ys_fine]), (0, 255, 0))
    for yc in yr:  # lines of constant Y (run along the X/lateral axis)
        project_polyline(np.column_stack([xs_fine, np.full_like(xs_fine, yc)]), (255, 200, 0))
    return out


def fit_bev_homography(calib, w: int, h: int, out: int) -> np.ndarray:
    """
    Scale/translate the (up-to-scale) image->ground homography so the visible
    *near-field road* lands inside an ``out x out`` canvas.

    Only pixels that sit on the ground side of the horizon are used; pixels near
    or above the horizon back-project to (near-)infinite ground distance and must
    be excluded or they smear the whole warp.
    """
    calib.compute_ground_homography()
    Hi2g = calib.H_img2ground

    # Dense sample over the whole frame.
    ys = np.linspace(0, h - 1, 60)
    xs = np.linspace(0, w - 1, 60)
    grid = np.array([(x, y) for y in ys for x in xs], dtype=np.float64)
    hom = np.hstack([grid, np.ones((len(grid), 1))])
    proj = (Hi2g @ hom.T).T                     # (N,3) ground homogeneous
    wden = proj[:, 2]

    # Near-field side = sign of the denominator at the bottom-centre pixel.
    bc = Hi2g @ np.array([w / 2, h - 1, 1.0])
    near_sign = np.sign(bc[2]) or 1.0
    # Keep ground-side pixels, excluding a margin around the horizon (small |w|).
    keep = (np.sign(wden) == near_sign) & (np.abs(wden) > 0.02 * np.abs(bc[2]))
    g = proj[keep, :2] / wden[keep, None]
    if len(g) < 8:
        raise RuntimeError(f"{calib.cam_id}: ground mapping degenerate")

    # Robust near-field extent (drop the long far-field tail).
    gmin = np.percentile(g, 3, axis=0)
    gmax = np.percentile(g, 85, axis=0)
    span = np.maximum(gmax - gmin, 1e-6)
    scale = (out * 0.9) / span.max()
    tx = -gmin[0] * scale + out * 0.05
    ty = -gmin[1] * scale + out * 0.05
    S = np.array([[scale, 0, tx], [0, scale, ty], [0, 0, 1]], dtype=np.float64)
    return S @ Hi2g


def main(folder: str, intrinsics_path: str):
    project = Project(folder)
    project.load()
    intrinsics = load_intrinsics(intrinsics_path)

    out_dir = Path(folder) / ".reid"
    out_dir.mkdir(exist_ok=True)
    panel_dir = out_dir / "calib_preview"
    panel_dir.mkdir(exist_ok=True)

    saved = {}
    for cam in project.cameras:
        cid = match_camera_id(cam.name, intrinsics)
        if cid is None:
            print(f"  [skip] no intrinsics matched for {cam.name}")
            continue
        calib = intrinsics[cid]
        frame = sample_frame(project.video_path(cam))
        h, w = frame.shape[:2]

        # 1. undistort
        map_x, map_y = calib.undistort_maps()
        undist = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

        # 2. bird's-eye
        Hbev = fit_bev_homography(calib, w, h, BEV_SIZE)
        # Mask out everything above the horizon so it does not smear the warp.
        Hi2g = calib.H_img2ground
        bc = Hi2g @ np.array([w / 2, h - 1, 1.0])
        near_sign = np.sign(bc[2]) or 1.0
        ys, xs = np.mgrid[0:h, 0:w]
        wden = Hi2g[2, 0] * xs + Hi2g[2, 1] * ys + Hi2g[2, 2]
        ground_mask = (np.sign(wden) == near_sign) & (np.abs(wden) > 0.02 * abs(bc[2]))
        undist_masked = undist.copy()
        undist_masked[~ground_mask] = 0
        bev = cv2.warpPerspective(undist_masked, Hbev, (BEV_SIZE, BEV_SIZE))

        # 2b. ground-grid overlay (the decisive calibration check)
        grid_overlay = draw_ground_grid(undist, calib, w, h)

        # 3. side-by-side panel (resize each to common height)
        ph = 500
        def fit(img):
            return cv2.resize(img, (int(img.shape[1] * ph / img.shape[0]), ph))
        tiles = [frame, grid_overlay, bev]
        labels = ["original", "undistorted + ground grid", "bird's-eye (own frame)"]
        panel = np.hstack([fit(t) for t in tiles])
        for i, txt in enumerate(labels):
            x = sum(fit(tiles[j]).shape[1] for j in range(i)) + 10
            cv2.putText(panel, txt, (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2, cv2.LINE_AA)
        out_path = panel_dir / f"{cid}.png"
        cv2.imwrite(str(out_path), panel)
        print(f"  {cid}: k1={calib.k1:+.4f}  panel -> {out_path}")
        saved[cid] = calib.to_dict()

    calib_json = out_dir / "calibration.json"
    with open(calib_json, "w") as f:
        json.dump({"cameras": saved}, f, indent=2)
    print(f"\n  calibration saved -> {calib_json}")
    print(f"  preview panels     -> {panel_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python scripts/build_homography.py <segment_folder> <intrinsics.json>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
