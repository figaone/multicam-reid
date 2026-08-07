"""
Sweep the spatial-gated matcher. Caches per-track appearance features AND
shared-frame world trajectories once, then scores several
(appearance_threshold, max_world_dist, require_world) settings against the
manual ground-truth pairs.

Run:
    python scripts/sweep_spatial_gate.py <segment>
"""

from __future__ import annotations

import importlib
import pickle
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multicam_reid.core.project import Project
from multicam_reid.core import tracks_io
from multicam_reid.core.ground_align import load_ground_frame
from multicam_reid.core.ground_bootstrap import project_track_feet

am = importlib.import_module("multicam_reid.reid.auto_match")
CACHE = Path("/tmp/spatial_sweep.pkl")


def pairs_from_matches(matches):
    out = set()
    for m in matches:
        present = sorted((cam, int(t)) for cam, t in m["tracks"].items() if t is not None)
        for a in range(len(present)):
            for b in range(a + 1, len(present)):
                out.add((present[a], present[b]))
    return out


def build(keys, cams, data, dist, thr, min_covis, max_world, require_world, cam_names):
    n = len(keys)
    best = [dict() for _ in range(n)]
    for i in range(n):
        di = data[keys[i]]
        for j in range(n):
            if cams[j] == cams[i] or dist[i, j] > thr:
                continue
            if am._covisible_frames(di, data[keys[j]]) < min_covis:
                continue
            wd = am._world_distance(di, data[keys[j]], 2)
            if wd is None:
                if require_world:
                    continue
            elif wd > max_world:
                continue
            cur = best[i].get(cams[j])
            if cur is None or dist[i, j] < cur[0]:
                best[i][cams[j]] = (float(dist[i, j]), j)
    edges = []
    for i in range(n):
        for cam_b, (d_ij, j) in best[i].items():
            back = best[j].get(cams[i])
            if back is not None and back[1] == i:
                edges.append((d_ij, min(i, j), max(i, j)))
    edges = sorted(set(edges), key=lambda e: e[0])
    uf = am._UnionFind(n)
    for idx in range(n):
        uf.cameras[idx].add(cams[idx])
    for _d, i, j in edges:
        uf.union(i, j)
    clusters = {}
    for idx in range(n):
        clusters.setdefault(uf.find(idx), []).append(idx)
    matches = []
    for members in clusters.values():
        if len({cams[m] for m in members}) < 2:
            continue
        tm = {name: None for name in cam_names}
        for m in members:
            c, tid = keys[m]
            tm[c] = int(tid)
        matches.append({"frame": 0, "tracks": tm})
    return matches


def main(folder):
    project = Project(folder)
    project.load()
    cam_names = project.camera_names
    reid = project.matches_path.parent
    gf = load_ground_frame(reid / "ground_frame.json")

    if CACHE.exists():
        with open(CACHE, "rb") as f:
            data = pickle.load(f)
        print(f"loaded cached features+world ({len(data)} tracks)")
    else:
        from multicam_reid.reid import ReIDExtractor
        extractor = ReIDExtractor()
        data = am._collect_track_crops(project, extractor, 6, 24, 5)
        # attach world trajectories
        for cam in project.cameras:
            if cam.name not in gf.calib:
                continue
            calib = gf.calib[cam.name]; S = gf.sim[cam.name]
            tracks = tracks_io.load_tracks(project.tracks_path(cam))
            for (ck, tid), entry in data.items():
                if ck != cam.name or tid not in tracks:
                    continue
                ground = project_track_feet(tracks[tid], calib, 2)
                world = {}
                for fr, g in ground.items():
                    h = np.array([g[0], g[1], 1.0]); w = S @ h
                    world[fr] = (w[0] / w[2], w[1] / w[2])
                entry["world"] = world
        with open(CACHE, "wb") as f:
            pickle.dump(data, f)
        print(f"extracted + cached ({len(data)} tracks)")

    keys = list(data.keys())
    cams = [k[0] for k in keys]
    feats = np.stack([data[k]["feature"] for k in keys])
    dist = np.clip(1.0 - feats @ feats.T, 0.0, None)

    gt = pairs_from_matches(tracks_io.load_matches(reid / "matches.manual.backup.json"))
    print(f"ground-truth pairs: {len(gt)}\n")
    print(f"{'appThr':>6} {'covis':>5} {'wDist':>6} {'reqW':>5} | "
          f"{'pred':>5} {'TP':>4} {'FP':>4} {'prec':>6} {'rec':>6} {'F1':>6}")
    print("-" * 68)
    for require_world in (True, False):
        for max_world in (0.5, 0.75, 1.0):
            for thr in (0.35, 0.45, 0.55):
                matches = build(keys, cams, data, dist, thr, 3, max_world, require_world, cam_names)
                pred = pairs_from_matches(matches)
                tp = len(gt & pred); fp = len(pred - gt)
                prec = tp / (tp + fp) if tp + fp else 0
                rec = tp / len(gt) if gt else 0
                f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
                print(f"{thr:>6.2f} {3:>5} {max_world:>6.2f} {str(require_world):>5} | "
                      f"{len(pred):>5} {tp:>4} {fp:>4} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f}")
        print()


if __name__ == "__main__":
    main(sys.argv[1])
