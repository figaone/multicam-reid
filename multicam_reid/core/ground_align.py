"""
Tie the per-camera ground planes into ONE shared world frame.

Background
----------
``calibration.py`` rectifies each camera into its *own* bird's-eye ground frame,
defined only up to an independent similarity (rotation, uniform scale, and
translation). Intrinsics + vanishing points cannot recover where the cameras sit
relative to one another, so we need a few **correspondences** — the same physical
ground point observed in two or more cameras — to solve for the similarity that
maps every camera's ground frame onto a common reference frame.

Once solved, a track's foot point in any camera can be mapped to a single shared
(x, y) position and matches can be gated by real-world distance.

Correspondence sources (in order of reliability)
-------------------------------------------------
1. Manual anchor points  — click the same landmark (crosswalk corner, pole base,
   lane-marking intersection) in each camera. A handful (>=3 per camera pair) is
   enough and is what the AI City calibration pipelines used.
2. GPS / surveyed points — if available, gives a true metric frame.
3. Bootstrap from confident ReID matches — foot points of mutually-matched tracks
   at co-visible frames. Cheap but noisy; use only many, robustly (RANSAC).

This module provides the math (similarity fitting + frame composition). The
manual-anchor collector is a thin UI on top (see ``scripts``/UI, added separately).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .calibration import CameraCalibration


def estimate_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Least-squares similarity transform (rotation + uniform scale + translation)
    mapping ``src`` points onto ``dst`` points (Umeyama, no reflection).

    ``src`` / ``dst`` are (N, 2) arrays with N >= 2 (>= 3 recommended).
    Returns a 3x3 homogeneous matrix.
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if len(src) != len(dst) or len(src) < 2:
        raise ValueError("Need matching src/dst with at least 2 points.")

    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    s0 = src - mu_s
    d0 = dst - mu_d
    cov = (d0.T @ s0) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1.0
    R = U @ S @ Vt
    var_s = (s0 ** 2).sum() / len(src)
    scale = np.trace(np.diag(D) @ S) / var_s
    t = mu_d - scale * R @ mu_s

    M = np.eye(3)
    M[:2, :2] = scale * R
    M[:2, 2] = t
    return M


def estimate_similarity_ransac(
    src: np.ndarray,
    dst: np.ndarray,
    threshold: float = 1.0,
    iterations: int = 500,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    RANSAC similarity fit for noisy correspondences (e.g. ReID-bootstrapped).
    Returns (3x3 matrix, inlier_mask).
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    n = len(src)
    if n < 2:
        raise ValueError("Need at least 2 correspondences.")
    rng = np.random.default_rng(seed)
    best_inliers = np.zeros(n, dtype=bool)
    best_M = estimate_similarity(src, dst)
    for _ in range(iterations):
        idx = rng.choice(n, size=2, replace=False)
        try:
            M = estimate_similarity(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        proj = (M @ np.hstack([src, np.ones((n, 1))]).T).T[:, :2]
        err = np.linalg.norm(proj - dst, axis=1)
        inliers = err < threshold
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_M = M
    if best_inliers.sum() >= 2:
        best_M = estimate_similarity(src[best_inliers], dst[best_inliers])
    return best_M, best_inliers


@dataclass
class GroundFrame:
    """
    A common ground frame shared by several cameras.

    ``sim[cam_id]`` is the 3x3 similarity mapping that camera's own ground
    coordinates into the shared frame. The reference camera has identity.
    """

    reference: str
    calib: dict[str, CameraCalibration]
    sim: dict[str, np.ndarray] = field(default_factory=dict)

    def image_to_world(self, cam_id: str, pts: np.ndarray) -> np.ndarray:
        """Map (N,2) image pixels in ``cam_id`` to shared-frame coordinates."""
        g = self.calib[cam_id].image_to_ground(pts)          # own ground frame
        S = self.sim.get(cam_id, np.eye(3))
        h = np.hstack([g, np.ones((len(g), 1))])
        w = (S @ h.T).T
        return w[:, :2] / w[:, 2:3]

    def has(self, cam_name: str) -> bool:
        return cam_name in self.calib

    def project_feet(self, cam_name: str, pts: np.ndarray) -> np.ndarray:
        """Foot pixels -> shared frame (undistort -> ground homography -> similarity)."""
        calib = self.calib[cam_name]
        und = calib.undistort_points(np.asarray(pts, dtype=np.float64))
        g = calib.image_to_ground(und)
        S = self.sim.get(cam_name, np.eye(3))
        h = np.hstack([g, np.ones((len(g), 1))])
        w = (S @ h.T).T
        return w[:, :2] / w[:, 2:3]

    def world_distance(self, cam_a: str, pt_a, cam_b: str, pt_b) -> float:
        """Shared-frame distance between a point in cam_a and a point in cam_b."""
        wa = self.image_to_world(cam_a, np.asarray(pt_a).reshape(1, 2))[0]
        wb = self.image_to_world(cam_b, np.asarray(pt_b).reshape(1, 2))[0]
        return float(np.linalg.norm(wa - wb))


def _cam_number(name: str) -> int | None:
    import re
    m = re.findall(r"[Cc]amera[-_]?(\d+)", name)
    return int(m[0]) if m else None


@dataclass
class HomographyGroundFrame:
    """
    A shared frame defined by a direct per-camera homography (image pixel ->
    shared ground plane). Used by automatic-calibration sources (MapAnything) and
    the team-supplied homographies. Keyed by camera number so it works regardless
    of the project's full camera names.
    """

    homographies: dict[int, np.ndarray]
    reference: str = "ground_plane"

    def _H(self, cam_name: str):
        n = _cam_number(cam_name)
        return None if n is None else self.homographies.get(n)

    def has(self, cam_name: str) -> bool:
        return self._H(cam_name) is not None

    def project_feet(self, cam_name: str, pts: np.ndarray) -> np.ndarray:
        """Foot pixels -> shared frame via the camera's homography."""
        H = self._H(cam_name)
        p = np.hstack([np.asarray(pts, dtype=np.float64), np.ones((len(pts), 1))])
        w = (H @ p.T).T
        return w[:, :2] / w[:, 2:3]



def save_ground_frame(path, frame: "GroundFrame") -> None:
    """Persist a shared ground frame (calibration + similarities) to JSON."""
    import json
    from pathlib import Path

    cams = {}
    for cam_key, calib in frame.calib.items():
        d = calib.to_dict()
        d["sim"] = frame.sim.get(cam_key, np.eye(3)).tolist()
        cams[cam_key] = d
    payload = {"version": 1, "reference": frame.reference, "cameras": cams}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_ground_frame(path) -> "GroundFrame":
    """Load a shared ground frame previously saved with ``save_ground_frame``."""
    import json

    with open(path) as f:
        data = json.load(f)

    # Homography-type frame (automatic calibration: MapAnything / team homographies)
    if data.get("type") == "homography":
        H = {}
        for key, entry in data["cameras"].items():
            n = _cam_number(key)
            if n is None:
                try:
                    n = int(key)
                except (TypeError, ValueError):
                    continue
            H[n] = np.array(entry["H"], dtype=np.float64)
        return HomographyGroundFrame(homographies=H, reference=data.get("reference", "ground_plane"))

    calib: dict[str, CameraCalibration] = {}
    sim: dict[str, np.ndarray] = {}
    for cam_key, d in data["cameras"].items():
        c = CameraCalibration(
            cam_id=d["cam_id"],
            width=int(d["width"]),
            height=int(d["height"]),
            K=np.array(d["K"], dtype=np.float64),
            k1=float(d["k1"]),
            halfdiag=float(d["halfdiag"]),
            road_vp=None if d.get("road_vp") is None else np.array(d["road_vp"], dtype=np.float64),
            vertical_vp=None if d.get("vertical_vp") is None else np.array(d["vertical_vp"], dtype=np.float64),
        )
        c.compute_ground_homography()
        calib[cam_key] = c
        sim[cam_key] = np.array(d.get("sim", np.eye(3).tolist()), dtype=np.float64)
    return GroundFrame(reference=data["reference"], calib=calib, sim=sim)


def build_ground_frame(
    calib: dict[str, CameraCalibration],
    reference: str,
    correspondences: dict[str, list[tuple[float, float]]],
    ransac: bool = False,
    ransac_threshold: float = 1.0,
) -> GroundFrame:
    """
    Solve for the similarity of every camera relative to ``reference``.

    ``correspondences[cam_id]`` is a list of image pixel points, one per shared
    anchor, in the SAME order across all cameras (use NaN for anchors a camera
    cannot see). Each camera's own-ground positions of its anchors are aligned to
    the reference camera's own-ground positions of the same anchors.
    """
    for cid in calib:
        calib[cid].compute_ground_homography()

    ref_pts_img = np.asarray(correspondences[reference], dtype=np.float64)
    ref_ground = calib[reference].image_to_ground(ref_pts_img)

    frame = GroundFrame(reference=reference, calib=calib)
    frame.sim[reference] = np.eye(3)

    for cid, pts in correspondences.items():
        if cid == reference:
            continue
        pts_img = np.asarray(pts, dtype=np.float64)
        valid = np.isfinite(pts_img).all(axis=1) & np.isfinite(ref_pts_img).all(axis=1)
        if valid.sum() < 2:
            raise ValueError(
                f"Camera {cid}: needs >=2 shared anchors with the reference "
                f"(got {int(valid.sum())})."
            )
        own_ground = calib[cid].image_to_ground(pts_img[valid])
        dst = ref_ground[valid]
        if ransac:
            M, _ = estimate_similarity_ransac(own_ground, dst, ransac_threshold)
        else:
            M = estimate_similarity(own_ground, dst)
        frame.sim[cid] = M
    return frame
