"""
Evaluate the team-provided per-camera homographies as the spatial gate.

Projects raw foot points through camera_<n>_H.npy (image px -> shared ground
frame), attaches them as world trajectories, then runs the real auto_match
pipeline at a sweep of world-distance thresholds and scores P/R/F1 against the
manual matches.

Run from Code/:
    python scripts/eval_team_homography.py <segment_folder> [--hdir DIR]
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib

from multicam_reid.core.project import Project
from multicam_reid.core import tracks_io

am = importlib.import_module("multicam_reid.reid.auto_match")


def load_team_H(hdir: str) -> dict[int, np.ndarray]:
    H = {}
    for f in sorted(glob.glob(os.path.join(hdir, "*.npy"))):
        m = re.search(r"camera_(\d+)_H", os.path.basename(f))
        if m:
            H[int(m.group(1))] = np.load(f)
    return H


def cam_number(name: str) -> int | None:
    m = re.findall(r"[Cc]amera[-_]?(\d+)", name)
    return int(m[0]) if m else None


def foot(box):
    x1, y1, x2, y2 = box
    return [(x1 + x2) * 0.5, float(y2)]


def project_pts(pts, Hm):
    p = np.hstack([np.asarray(pts, float), np.ones((len(pts), 1))])
    w = (Hm @ p.T).T
    return w[:, :2] / w[:, 2:3]


def pairs_from_matches(matches):
    """Every within-cluster cross-camera track pair -> set of frozensets."""
    out = set()
    for m in matches:
        present = [(c, int(t)) for c, t in m["tracks"].items() if t is not None]
        for a, b in combinations(present, 2):
            out.add(frozenset([a, b]))
    return out


def score(pred, gt):
    inter = len(pred & gt)
    p = inter / len(pred) if pred else 0.0
    r = inter / len(gt) if gt else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--hdir", default=str(Path(__file__).resolve().parents[1] / "Fw_ Homography"))
    ap.add_argument("--homography-json", default=None,
                    help="Use per-camera homographies from this JSON (e.g. MapAnything) "
                         "instead of the team .npy files.")
    ap.add_argument("--thresholds", default="10,15,20,25,30,40")
    ap.add_argument("--app-threshold", type=float, default=0.45)
    args = ap.parse_args()

    project_ = Project(args.folder)
    project_.load()
    cam_H = {}
    if args.homography_json:
        import json
        data = json.load(open(args.homography_json))
        by_key = data["cameras"]
        for cam in project_.cameras:
            entry = by_key.get(cam.name) or by_key.get(str(cam_number(cam.name)))
            if entry is not None:
                cam_H[cam.name] = np.array(entry["H"], dtype=np.float64)
    else:
        H = load_team_H(args.hdir)
        for cam in project_.cameras:
            n = cam_number(cam.name)
            if n in H:
                cam_H[cam.name] = H[n]

    # Monkeypatch the world-attach step to use team homographies on raw feet.
    def attach_team(project, data, ground_frame, stride=2):
        for cam in project.cameras:
            if cam.name not in cam_H:
                continue
            Hm = cam_H[cam.name]
            tracks = tracks_io.load_tracks(project.tracks_path(cam))
            for (ck, tid), entry in data.items():
                if ck != cam.name or tid not in tracks:
                    continue
                tr = tracks[tid]
                fr, bx = tr["frames"], tr["boxes"]
                idx = range(0, len(fr), stride)
                feet = [foot(bx[i]) for i in idx]
                frames = [fr[i] for i in idx]
                w = project_pts(feet, Hm)
                entry["world"] = {frames[k]: (float(w[k, 0]), float(w[k, 1]))
                                  for k in range(len(frames))}

    am._attach_world_trajectories = attach_team

    # Cache appearance features once; patch extraction so the sweep is fast.
    cache = Path("/tmp/team_homog_feats.npz")
    if cache.exists():
        blob = np.load(cache, allow_pickle=True)
        ckeys = [tuple(k) for k in blob["keys"]]
        feats = blob["feats"]; los = blob["los"]; his = blob["his"]
        fsets = blob["fsets"]
        cached_data = {k: {"feature": feats[i], "lo": int(los[i]), "hi": int(his[i]),
                           "frame_set": set(int(x) for x in fsets[i])}
                       for i, k in enumerate(ckeys)}
    else:
        from multicam_reid.reid.extractor import ReIDExtractor
        extractor = ReIDExtractor()
        cached_data = am._collect_track_crops(project_, extractor, 6, 24, 5)
        ck = list(cached_data.keys())
        np.savez(cache,
                 keys=np.array([[k[0], k[1]] for k in ck], dtype=object),
                 feats=np.stack([cached_data[k]["feature"] for k in ck]),
                 los=np.array([cached_data[k]["lo"] for k in ck]),
                 his=np.array([cached_data[k]["hi"] for k in ck]),
                 fsets=np.array([sorted(cached_data[k]["frame_set"]) for k in ck], dtype=object))

    def collect_cached(project, extractor, *a, **k):
        # return a deep-ish copy so per-run "world" keys don't accumulate
        return {kk: {kkk: (set(vv) if kkk == "frame_set" else vv)
                     for kkk, vv in v.items()}
                for kk, v in cached_data.items()}
    am._collect_track_crops = collect_cached

    class DummyExtractor:
        def __init__(self, *a, **k):
            pass
    am.ReIDExtractor = DummyExtractor

    class DummyFrame:
        calib = cam_H  # truthy, keys used only by the (now replaced) attach

    gt = pairs_from_matches(
        tracks_io.load_matches(project_.matches_path.parent / "matches.manual.backup.json")
    )

    # Patch load_ground_frame in ground_align (imported lazily inside auto_match).
    import multicam_reid.core.ground_align as ga
    ga.load_ground_frame = lambda path: DummyFrame()

    for thr in [float(x) for x in args.thresholds.split(",")]:
        am.auto_match(
            project_,
            dist_threshold=args.app_threshold,
            min_covis=3,
            mutual_only=True,
            ground_frame_path="__team__",
            max_world_dist=thr,
            require_world=True,
            world_min_shared=2,
        )
        pred = pairs_from_matches(tracks_io.load_matches(project_.matches_path))
        p, r, f1 = score(pred, gt)
        print(f"world_thr={thr:5.1f}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}  (pred pairs={len(pred)})")

    # restore manual matches
    backup = project_.matches_path.parent / "matches.manual.backup.json"
    import shutil
    shutil.copy(backup, project_.matches_path)
    print("restored manual matches.json")


if __name__ == "__main__":
    main()
