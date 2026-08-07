"""
Compare shared-ground-frame sources head-to-head on the SAME data.

For each source we project the manually-matched track foot points into its shared
frame and measure how well the real-world distance separates TRUE cross-camera
pairs (same car) from RANDOM cross-camera pairs. Bigger separation = better gate.

Sources supported:
  * bootstrap    -> .reid/ground_frame.json  (our calibrated frame; needs manual
                    anchors, the current best: ~18x separation)
  * team         -> Fw_ Homography/camera_<n>_H.npy  (team-supplied homographies)
  * mapanything  -> a homography-type JSON written by
                    scripts/mapanything_to_ground_frame.py  (automatic, no manual
                    anchors) -- the thing we want to validate.

Run from Code/:
    python scripts/compare_shared_frames.py <segment_folder> \
        [--team "Fw_ Homography"] [--mapanything ground_frame_mapanything.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multicam_reid.core.project import Project
from multicam_reid.core import tracks_io
from multicam_reid.core.ground_align import load_ground_frame


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def cam_number(name: str) -> int | None:
    m = re.findall(r"[Cc]amera[-_]?(\d+)", name)
    return int(m[0]) if m else None


def foot(box):
    x1, y1, x2, y2 = box
    return [(x1 + x2) * 0.5, float(y2)]


def apply_H(pts, H):
    p = np.hstack([np.asarray(pts, float), np.ones((len(pts), 1))])
    w = (H @ p.T).T
    return w[:, :2] / w[:, 2:3]


# --------------------------------------------------------------------------- #
# projectors: (cam_name, Nx2 image px) -> Nx2 shared-frame coords
# --------------------------------------------------------------------------- #
class BootstrapProjector:
    """Calibrated frame: undistort -> per-camera ground homography -> similarity."""

    def __init__(self, ground_frame):
        self.gf = ground_frame

    def has(self, cam_name):
        return cam_name in self.gf.calib

    def project(self, cam_name, pts):
        calib = self.gf.calib[cam_name]
        sim = self.gf.sim.get(cam_name, np.eye(3))
        und = calib.undistort_points(np.asarray(pts, float))
        g = calib.image_to_ground(und)
        return apply_H(g, sim)


class HomographyProjector:
    """Direct per-camera image-pixel -> shared-plane homography (team / MapAnything)."""

    def __init__(self, cam_H: dict[str, np.ndarray]):
        self.cam_H = cam_H

    def has(self, cam_name):
        return cam_name in self.cam_H

    def project(self, cam_name, pts):
        return apply_H(pts, self.cam_H[cam_name])


def load_team(project, hdir):
    H = {}
    for f in sorted(glob.glob(os.path.join(hdir, "*.npy"))):
        m = re.search(r"camera_(\d+)_H", os.path.basename(f))
        if m:
            H[int(m.group(1))] = np.load(f)
    cam_H = {c.name: H[cam_number(c.name)] for c in project.cameras
             if cam_number(c.name) in H}
    return HomographyProjector(cam_H) if cam_H else None


def load_homography_json(project, path):
    with open(path) as f:
        data = json.load(f)
    by_key = data["cameras"]
    cam_H = {}
    for c in project.cameras:
        key_num = str(cam_number(c.name))
        entry = by_key.get(c.name) or by_key.get(key_num)
        if entry is not None:
            cam_H[c.name] = np.array(entry["H"], dtype=np.float64)
    return HomographyProjector(cam_H) if cam_H else None


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def evaluate(project, projector, tracks, manual, stride=3, n_neg=500, seed=0):
    def getT(cam, tid):
        d = tracks[cam]
        return d.get(str(tid)) or d.get(int(tid))

    cache = {}

    def world(cam, tid):
        key = (cam, tid)
        if key in cache:
            return cache[key]
        tr = getT(cam, tid)
        if tr is None or not projector.has(cam):
            cache[key] = {}
            return {}
        fr, bx = tr["frames"], tr["boxes"]
        idx = range(0, len(fr), stride)
        feet = [foot(bx[i]) for i in idx]
        frames = [fr[i] for i in idx]
        w = projector.project(cam, feet)
        out = {int(frames[k]): w[k] for k in range(len(frames)) if np.isfinite(w[k]).all()}
        cache[key] = out
        return out

    def pair_dist(a, b):
        wa, wb = world(*a), world(*b)
        shared = set(wa) & set(wb)
        if not shared:
            return None
        ds = [float(np.linalg.norm(wa[f] - wb[f])) for f in shared]
        ds.sort()
        return ds[len(ds) // 2]

    # positives: manual cross-camera matched pairs
    pos = []
    present_keys = []
    for m in manual:
        present = [(c, t) for c, t in m["tracks"].items()
                   if t is not None and projector.has(c)]
        present_keys += present
        for a, b in combinations(present, 2):
            d = pair_dist(a, b)
            if d is not None:
                pos.append(d)

    # negatives: random cross-camera pairs
    random.seed(seed)
    all_keys = [(c.name, tid) for c in project.cameras if projector.has(c.name)
                for tid in list(tracks[c.name].keys())[:250]]
    neg, tries = [], 0
    while len(neg) < n_neg and tries < 40000:
        tries += 1
        a, b = random.sample(all_keys, 2)
        if a[0] == b[0]:
            continue
        d = pair_dist(a, b)
        if d is not None:
            neg.append(d)

    return np.array(pos), np.array(neg)


def report(name, pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        print(f"  {name:12s}: insufficient data (pos={len(pos)}, neg={len(neg)})")
        return
    pm, nm = np.median(pos), np.median(neg)
    ratio = nm / pm if pm > 0 else float("inf")
    # threshold that keeps 90% of positives; what fraction of negatives passes?
    thr = np.percentile(pos, 90)
    neg_leak = float((neg <= thr).mean())
    print(f"  {name:12s}: pos_med={pm:8.3f}  neg_med={nm:8.3f}  "
          f"separation={ratio:5.2f}x  |  keep90%pos thr={thr:8.3f} -> "
          f"neg_leak={neg_leak*100:4.1f}%   (pos n={len(pos)}, neg n={len(neg)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--team", default=str(Path(__file__).resolve().parents[1] / "Fw_ Homography"))
    ap.add_argument("--mapanything", default=None,
                    help="Path to a homography-type ground_frame JSON from the adapter.")
    args = ap.parse_args()

    project = Project(args.folder)
    project.load()
    tracks = {c.name: tracks_io.load_tracks(project.tracks_path(c)) for c in project.cameras}
    manual = tracks_io.load_matches(project.matches_path.parent / "matches.manual.backup.json")

    projectors = {}
    gf_path = project.matches_path.parent / "ground_frame.json"
    if gf_path.exists():
        projectors["bootstrap"] = BootstrapProjector(load_ground_frame(gf_path))
    if args.team and Path(args.team).exists():
        p = load_team(project, args.team)
        if p:
            projectors["team"] = p
    if args.mapanything and Path(args.mapanything).exists():
        p = load_homography_json(project, args.mapanything)
        if p:
            projectors["mapanything"] = p

    if not projectors:
        raise SystemExit("No shared-frame sources found to compare.")

    print(f"\nShared-frame comparison on {Path(args.folder).name}")
    print(f"(positives = manual cross-camera matches, negatives = random cross-camera pairs)\n")
    for name, proj in projectors.items():
        pos, neg = evaluate(project, proj, tracks, manual)
        report(name, pos, neg)
    print("\nBigger separation (and lower neg_leak) = better spatial gate.\n")


if __name__ == "__main__":
    main()
