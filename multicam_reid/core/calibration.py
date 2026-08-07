"""
Camera calibration & ground-plane geometry.

Consumes the intrinsics file supplied by the calibration team
(``intersection_cameras_intrinsics.json``) and derives, per camera:

  * a lens-undistortion mapping (single-term radial model), and
  * a ground-plane homography (image <-> metric-up-to-scale bird's-eye view)
    built from the intrinsic matrix K and the two orthogonal vanishing points
    (road direction + vertical).

What this gives us
------------------
Each camera is rectified into ITS OWN top-down ground frame. Straight roads
become straight and parallel, and distances on the road plane become linear.
This is everything needed to reason about a vehicle's position *within a single
camera*.

What this does NOT give us
--------------------------
A common world frame shared by all cameras. Intrinsics + vanishing points fix
each camera's internal orientation, but NOT where the cameras sit relative to
one another (their positions / relative scale). Tying the three ground frames
into one map requires a handful of correspondences between overlapping views
(see ``align.py`` / the correspondence step). Until then each camera's bird's-eye
frame is only defined up to an independent similarity (rotation, scale, offset).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CameraCalibration:
    """Intrinsic calibration + derived ground-plane geometry for one camera."""

    cam_id: str
    width: int
    height: int
    K: np.ndarray                 # 3x3 intrinsic matrix
    k1: float                     # single-term radial distortion coefficient
    halfdiag: float               # normalisation radius for the distortion model
    road_vp: np.ndarray | None    # road vanishing point (pixels), or None
    vertical_vp: np.ndarray | None  # vertical vanishing point (pixels), or None

    # Lazily computed / assigned:
    H_img2ground: np.ndarray | None = None   # 3x3 image -> ground (up to scale)
    H_ground2img: np.ndarray | None = None   # 3x3 ground -> image

    # ---- distortion -------------------------------------------------------

    def undistort_maps(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Build ``cv2.remap`` lookup maps that turn the distorted stream into an
        undistorted image (same K / resolution).

        The team's model is a single-term, pixel-normalised radial model:

            r_undistorted = r_distorted * (1 + k1 * (r_distorted / halfdiag)^2)

        To fill an undistorted output pixel we need the *source* (distorted)
        location, i.e. the inverse mapping. The 1-D radial function is monotonic,
        so we invert it with a few Newton iterations (vectorised).
        """
        cx = float(self.K[0, 2])
        cy = float(self.K[1, 2])
        ys, xs = np.mgrid[0:self.height, 0:self.width].astype(np.float64)
        dx = xs - cx
        dy = ys - cy
        r_u = np.sqrt(dx * dx + dy * dy)          # undistorted radius (output)

        hd = self.halfdiag
        k1 = self.k1
        # Solve r_d * (1 + k1 (r_d/hd)^2) = r_u  for r_d.
        r_d = r_u.copy()
        for _ in range(12):
            f = r_d * (1.0 + k1 * (r_d / hd) ** 2) - r_u
            df = 1.0 + 3.0 * k1 * (r_d / hd) ** 2
            r_d -= f / np.maximum(df, 1e-6)
        scale = np.where(r_u > 1e-6, r_d / np.maximum(r_u, 1e-6), 1.0)

        map_x = (cx + dx * scale).astype(np.float32)
        map_y = (cy + dy * scale).astype(np.float32)
        return map_x, map_y

    def undistort_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Map distorted pixel coordinates to undistorted ones (forward model).

        ``pts`` is an (N, 2) array of (x, y) pixel coordinates.
        """
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        c = np.array([self.K[0, 2], self.K[1, 2]])
        d = pts - c
        r = np.linalg.norm(d, axis=1, keepdims=True)
        factor = 1.0 + self.k1 * (r / self.halfdiag) ** 2
        return c + d * factor

    # ---- ground-plane homography -----------------------------------------

    def compute_ground_homography(self) -> None:
        """
        Derive the image<->ground homography from K and the two vanishing points.

        Geometry: a vanishing point ``v`` back-projects to a 3-D direction
        ``K^-1 v`` in the camera frame. The road VP gives an in-plane (horizontal)
        road direction; the vertical VP gives the ground-plane normal. Together
        with their cross product they form the world axes expressed in the camera
        frame, i.e. the camera rotation R. The ground plane is world Z = 0, so

            H_ground->image = K [ e_x | e_y | t ]

        with t chosen as a unit camera height (scale is unknown without a metric
        reference, hence "up to scale").
        """
        if self.road_vp is None or self.vertical_vp is None:
            raise ValueError(
                f"Camera {self.cam_id}: need both road and vertical vanishing "
                f"points to build a ground homography."
            )

        Kinv = np.linalg.inv(self.K)
        vp_road = np.array([self.road_vp[0], self.road_vp[1], 1.0])
        vp_vert = np.array([self.vertical_vp[0], self.vertical_vp[1], 1.0])

        e_x = Kinv @ vp_road
        e_z = Kinv @ vp_vert
        e_x /= np.linalg.norm(e_x)
        e_z /= np.linalg.norm(e_z)

        # Make the vertical the plane normal point "up" out of the ground toward
        # the camera (positive depth is +Z in the camera frame -> ensure e_z has
        # a consistent sign so the ground sits in front of the camera).
        if e_z[2] < 0:
            e_z = -e_z

        # Orthogonalise the road direction against the plane normal.
        e_x = e_x - np.dot(e_x, e_z) * e_z
        e_x /= np.linalg.norm(e_x)
        e_y = np.cross(e_z, e_x)

        t = -e_z  # unit camera height along the plane normal
        H_g2i = self.K @ np.column_stack([e_x, e_y, t])

        # Normalise and orient so that image points map to a right-side-up
        # bird's-eye view.
        H_i2g = np.linalg.inv(H_g2i)
        H_i2g /= H_i2g[2, 2]

        self.H_ground2img = H_g2i
        self.H_img2ground = H_i2g

    def image_to_ground(self, pts: np.ndarray) -> np.ndarray:
        """Map (N,2) image pixels to ground-plane coordinates (up to scale)."""
        if self.H_img2ground is None:
            self.compute_ground_homography()
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        h = np.hstack([pts, np.ones((len(pts), 1))])
        g = (self.H_img2ground @ h.T).T
        return g[:, :2] / g[:, 2:3]

    def to_dict(self) -> dict:
        return {
            "cam_id": self.cam_id,
            "width": self.width,
            "height": self.height,
            "K": self.K.tolist(),
            "k1": self.k1,
            "halfdiag": self.halfdiag,
            "road_vp": None if self.road_vp is None else list(map(float, self.road_vp)),
            "vertical_vp": None if self.vertical_vp is None else list(map(float, self.vertical_vp)),
            "H_img2ground": None if self.H_img2ground is None else self.H_img2ground.tolist(),
            "H_ground2img": None if self.H_ground2img is None else self.H_ground2img.tolist(),
        }


def load_intrinsics(json_path: str | Path) -> dict[str, CameraCalibration]:
    """
    Load the calibration-team intrinsics file into per-camera calibrations
    (keyed by the camera id, e.g. ``intersection_camera_1``).
    """
    with open(json_path) as f:
        data = json.load(f)

    out: dict[str, CameraCalibration] = {}
    for cam in data["cameras"]:
        res = cam["resolution"]
        intr = cam["intrinsics"]
        dist = cam.get("distortion", {})
        diag = cam.get("calibration_diagnostics", {})

        road_vp = diag.get("road_vanishing_point")
        vert_vp = diag.get("vertical_vanishing_point")

        calib = CameraCalibration(
            cam_id=cam["id"],
            width=int(res["width"]),
            height=int(res["height"]),
            K=np.array(intr["K"], dtype=np.float64),
            k1=float(dist.get("k1", 0.0)),
            halfdiag=float(dist.get("halfdiag", 0.5 * np.hypot(res["width"], res["height"]))),
            road_vp=None if road_vp is None else np.array(road_vp, dtype=np.float64),
            vertical_vp=None if vert_vp is None else np.array(vert_vp, dtype=np.float64),
        )
        out[calib.cam_id] = calib
    return out


def match_camera_id(cam_name: str, intrinsics: dict[str, CameraCalibration]) -> str | None:
    """
    Map a project camera name (e.g. ``Intersection-Camera-3_2026-...``) to an
    intrinsics id (e.g. ``intersection_camera_3``) by matching the trailing
    camera number.
    """
    import re

    m = re.search(r"camera[^0-9]*([0-9]+)", cam_name, flags=re.IGNORECASE)
    if not m:
        return None
    num = m.group(1)
    for cid in intrinsics:
        cm = re.search(r"([0-9]+)\s*$", cid)
        if cm and cm.group(1) == num:
            return cid
    return None
