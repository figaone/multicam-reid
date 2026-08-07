"""
Bootstrap a shared ground frame from cross-camera track matches.

Reusable core logic (used by both the diagnostic script and the calibration
persistence step). See ``scripts/bootstrap_ground_frame.py`` for the CLI and the
scatter visualisation, and ``ground_align`` for the geometry.

When two cameras are matched to the same vehicle, its foot point (bottom-centre
of the box) is the same physical ground location in both views. At each
co-visible frame this yields one correspondence between the cameras' own ground
frames; RANSAC-fitting many of them lands every camera on one shared frame.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .calibration import CameraCalibration, load_intrinsics, match_camera_id
from .ground_align import GroundFrame, estimate_similarity_ransac


def foot_point(box) -> tuple[float, float]:
    """Bottom-centre of a box — the vehicle's contact point with the ground."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) * 0.5, float(y2))


def build_calib_by_cam(project, intrinsics_path) -> dict[str, CameraCalibration]:
    """Map each project camera name to its calibration (ground homography ready)."""
    intrinsics = load_intrinsics(intrinsics_path)
    out: dict[str, CameraCalibration] = {}
    for cam in project.cameras:
        cid = match_camera_id(cam.name, intrinsics)
        if cid is None:
            continue
        calib = intrinsics[cid]
        calib.compute_ground_homography()
        out[cam.name] = calib
    return out


def project_track_feet(track: dict, calib: CameraCalibration, stride: int = 3) -> dict[int, np.ndarray]:
    """
    Foot points of a track, undistorted and projected to the camera's own ground
    frame. Boxes are stored in ORIGINAL (distorted) pixels, so we undistort first.

    Returns {frame_index: ground_xy}.
    """
    frames = track["frames"]
    boxes = track["boxes"]
    feet, fr = [], []
    for i in range(0, len(frames), stride):
        feet.append(foot_point(boxes[i]))
        fr.append(frames[i])
    if not feet:
        return {}
    und = calib.undistort_points(np.array(feet, dtype=np.float64))
    g = calib.image_to_ground(und)
    return {f: g[k] for k, f in enumerate(fr) if np.isfinite(g[k]).all()}


def collect_correspondences(tracks, calib_by_cam, matches, stride: int = 3):
    """
    corr[(cam_a, cam_b)] = (list of ground pts in a, list of ground pts in b)
    built from co-visible frames of matched track pairs.
    """
    proj_cache: dict[tuple, dict] = {}

    def get_proj(cam, tid):
        key = (cam, tid)
        if key not in proj_cache:
            proj_cache[key] = project_track_feet(tracks[cam][tid], calib_by_cam[cam], stride)
        return proj_cache[key]

    corr: dict[tuple, tuple[list, list]] = {}
    for m in matches:
        present = [(cam, tid) for cam, tid in m["tracks"].items()
                   if tid is not None and cam in calib_by_cam]
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


def solve_shared_frame(calib_by_cam, corr, ransac_threshold: float = 0.05):
    """
    Solve the similarity of every camera onto a reference frame.

    Returns (GroundFrame, stats) where stats[cam] = {pts, inliers, rms}.
    Reference = the camera present in the most correspondence points.
    """
    cam_weight: dict[str, int] = {}
    for (ca, cb), (pa, pb) in corr.items():
        cam_weight[ca] = cam_weight.get(ca, 0) + len(pa)
        cam_weight[cb] = cam_weight.get(cb, 0) + len(pb)
    if not cam_weight:
        raise ValueError("No correspondences to solve a shared frame.")
    reference = max(cam_weight, key=cam_weight.get)

    sim = {reference: np.eye(3)}
    stats: dict[str, dict] = {reference: {"pts": 0, "inliers": 0, "rms": 0.0}}
    for cam in calib_by_cam:
        if cam == reference:
            continue
        src, dst = [], []
        if (reference, cam) in corr:
            d, s = corr[(reference, cam)]
            dst += d; src += s
        if (cam, reference) in corr:
            s, d = corr[(cam, reference)]
            dst += d; src += s
        if len(src) < 2:
            continue
        src = np.array(src); dst = np.array(dst)
        M, inl = estimate_similarity_ransac(src, dst, threshold=ransac_threshold, iterations=800)
        proj = (M @ np.hstack([src, np.ones((len(src), 1))]).T).T[:, :2]
        res = np.linalg.norm(proj[inl] - dst[inl], axis=1) if inl.any() else np.array([0.0])
        sim[cam] = M
        stats[cam] = {
            "pts": int(len(src)),
            "inliers": int(inl.sum()),
            "rms": float(np.sqrt((res ** 2).mean())),
        }
    frame = GroundFrame(reference=reference, calib=calib_by_cam, sim=sim)
    return frame, stats
