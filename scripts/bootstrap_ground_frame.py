"""
Bootstrap a shared ground frame directly from cross-camera track matches.

Idea
----
When two cameras are matched to the SAME vehicle, that vehicle's foot point
(bottom-centre of its box) is the SAME physical ground location in both views.
At every co-visible frame this yields one correspondence between the two
cameras' own ground frames. Collect many of them across all matched vehicles
and RANSAC-fit a similarity that lands every camera on one shared frame — no
manual clicking required.

Robustness
----------
Matches may be noisy (auto ReID is imperfect), so a wrong pair produces
inconsistent correspondences that RANSAC rejects. We also undistort foot points
first (boxes are stored in the original distorted pixels) and drop points above
the horizon.

Run:
    python scripts/bootstrap_ground_frame.py <segment> [--matches auto|manual|<path>] [--map <png>]
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multicam_reid.core.project import Project
from multicam_reid.core import tracks_io
from multicam_reid.core.calibration import load_intrinsics, match_camera_id
from multicam_reid.core.ground_align import estimate_similarity_ransac

DEFAULT_INTRINSICS = "/home/kojo/Downloads/intersection_cameras_intrinsics.json"
CAM_COLORS = [(60, 220, 60), (60, 160, 255), (255, 120, 60), (200, 60, 255)]


def foot_point(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) * 0.5, float(y2))


def ground_pts_for_track(track, calib, stride=3):
    """Undistort + project all foot points of a track to its own ground frame."""
    frames = track["frames"]
    boxes = track["boxes"]
    feet, fr = [], []
    for i in range(0, len(frames), stride):
        feet.append(foot_point(boxes[i]))
        fr.append(frames[i])
    if not feet:
        return {}
    feet = np.array(feet, dtype=np.float64)
    und = calib.undistort_points(feet)
    g = calib.image_to_ground(und)
    return {f: g[k] for k, f in enumerate(fr) if np.isfinite(g[k]).all()}


def collect_correspondences(project, tracks, calib_by_cam, matches, stride=3):
    """
    Return corr[(cam_a, cam_b)] = (Nx2 ground pts in a, Nx2 ground pts in b),
    built from co-visible frames of matched track pairs.
    """
    # Pre-project every relevant track once.
    proj_cache: dict[tuple, dict] = {}

    def get_proj(cam, tid):
        key = (cam, tid)
        if key not in proj_cache:
            proj_cache[key] = ground_pts_for_track(
                tracks[cam][tid], calib_by_cam[cam], stride
            )
        return proj_cache[key]

    corr: dict[tuple, tuple[list, list]] = {}
    for m in matches:
        present = [(cam, tid) for cam, tid in m["tracks"].items() if tid is not None]
        for (cam_a, tid_a), (cam_b, tid_b) in combinations(present, 2):
            if int(tid_a) not in tracks[cam_a] or int(tid_b) not in tracks[cam_b]:
                continue
            ga = get_proj(cam_a, int(tid_a))
            gb = get_proj(cam_b, int(tid_b))
            shared = set(ga) & set(gb)
            if not shared:
                continue
            key = (cam_a, cam_b)
            corr.setdefault(key, ([], []))
            for f in shared:
                corr[key][0].append(ga[f])
                corr[key][1].append(gb[f])
    return corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("segment")
    ap.add_argument("--matches", default="manual",
                    help="'auto' (matches.auto.json), 'manual' (matches.manual.backup.json), or a path")
    ap.add_argument("--intrinsics", default=DEFAULT_INTRINSICS)
    ap.add_argument("--map", default=None, help="optional aerial map PNG for the backdrop")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--ransac-thr", type=float, default=0.05,
                    help="inlier threshold in reference-ground units")
    args = ap.parse_args()

    project = Project(args.segment)
    project.load()
    intrinsics = load_intrinsics(args.intrinsics)

    # Map project cameras -> calibrations, compute ground homographies.
    calib_by_cam = {}
    for cam in project.cameras:
        cid = match_camera_id(cam.name, intrinsics)
        if cid is None:
            print(f"  [skip] no intrinsics for {cam.name}")
            continue
        calib_by_cam[cam.name] = intrinsics[cid]
        intrinsics[cid].compute_ground_homography()

    # Load tracks + matches.
    tracks = {c.name: tracks_io.load_tracks(project.tracks_path(c)) for c in project.cameras}
    reid_dir = project.matches_path.parent
    if args.matches == "auto":
        mpath = reid_dir / "matches.auto.json"
    elif args.matches == "manual":
        mpath = reid_dir / "matches.manual.backup.json"
    else:
        mpath = Path(args.matches)
    matches = tracks_io.load_matches(mpath)
    print(f"  matches source: {mpath.name}  ({len(matches)} clusters)")

    corr = collect_correspondences(project, tracks, calib_by_cam, matches, args.stride)
    if not corr:
        print("  no correspondences found (no co-visible matched frames).")
        return

    # Pick reference = camera appearing in the most correspondence points.
    cam_weight: dict[str, int] = {}
    for (ca, cb), (pa, pb) in corr.items():
        cam_weight[ca] = cam_weight.get(ca, 0) + len(pa)
        cam_weight[cb] = cam_weight.get(cb, 0) + len(pb)
    reference = max(cam_weight, key=cam_weight.get)
    print(f"  reference camera: {reference}")

    # Solve similarity of every other camera onto the reference frame.
    sim = {reference: np.eye(3)}
    for cam in calib_by_cam:
        if cam == reference:
            continue
        src, dst = [], []
        # direct correspondences reference<->cam (either ordering)
        if (reference, cam) in corr:
            d, s = corr[(reference, cam)]
            dst += d; src += s
        if (cam, reference) in corr:
            s, d = corr[(cam, reference)]
            dst += d; src += s
        if len(src) < 3:
            print(f"  [warn] {cam}: only {len(src)} direct correspondences with reference")
            if not src:
                continue
        src = np.array(src); dst = np.array(dst)
        M, inl = estimate_similarity_ransac(src, dst, threshold=args.ransac_thr, iterations=800)
        proj = (M @ np.hstack([src, np.ones((len(src), 1))]).T).T[:, :2]
        res = np.linalg.norm(proj[inl] - dst[inl], axis=1)
        sim[cam] = M
        print(f"  {cam:45s} pts={len(src):4d}  inliers={int(inl.sum()):4d}  "
              f"RMS={np.sqrt((res**2).mean()):.4f}  (ref-ground units)")

    # ---- Render shared-frame scatter of all track foot points ----
    all_world = []
    for cam in calib_by_cam:
        S = sim.get(cam)
        if S is None:
            continue
        for tid, tr in tracks[cam].items():
            g = ground_pts_for_track(tr, calib_by_cam[cam], stride=max(1, args.stride))
            if not g:
                continue
            pts = np.array(list(g.values()))
            h = np.hstack([pts, np.ones((len(pts), 1))])
            w = (S @ h.T).T
            w = w[:, :2] / w[:, 2:3]
            for p in w:
                all_world.append((cam, p[0], p[1]))

    if all_world:
        xs = np.array([p[1] for p in all_world])
        ys = np.array([p[2] for p in all_world])
        xlo, xhi = np.percentile(xs, 1), np.percentile(xs, 99)
        ylo, yhi = np.percentile(ys, 1), np.percentile(ys, 99)
        W = H = 900
        canvas = np.full((H, W, 3), 20, np.uint8)

        def to_px(x, y):
            u = int((x - xlo) / max(xhi - xlo, 1e-9) * (W - 40) + 20)
            v = int((y - ylo) / max(yhi - ylo, 1e-9) * (H - 40) + 20)
            return u, np.clip(v, 0, H - 1)

        cam_list = list(calib_by_cam)
        for cam, x, y in all_world:
            if not (xlo <= x <= xhi and ylo <= y <= yhi):
                continue
            u, v = to_px(x, y)
            col = CAM_COLORS[cam_list.index(cam) % len(CAM_COLORS)]
            cv2.circle(canvas, (u, v), 2, col, -1)
        for i, cam in enumerate(cam_list):
            col = CAM_COLORS[i % len(CAM_COLORS)]
            cv2.putText(canvas, cam.split("_")[0], (20, 30 + 26 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
        out = reid_dir / f"shared_frame_{args.matches}.png"
        cv2.imwrite(str(out), canvas)
        print(f"  shared-frame scatter -> {out}")


if __name__ == "__main__":
    main()
