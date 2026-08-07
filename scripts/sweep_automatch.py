"""
Parameter sweep for the automatic matcher.

Extracts per-track features ONCE (cached to /tmp/automatch_sweep.pkl), then
tries several (threshold, min_covis, mutual) combinations and scores each
against the manual ground-truth pairs. Helps pick a high-precision operating
point for pre-populating the annotation tool.

Run:
    python scripts/sweep_automatch.py <segment_folder>
"""

from __future__ import annotations

import importlib
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multicam_reid.core.project import Project
from multicam_reid.core import tracks_io

am = importlib.import_module("multicam_reid.reid.auto_match")

CACHE = Path("/tmp/automatch_sweep.pkl")


def pairs_from_matches(matches: list[dict]) -> set[tuple]:
    out: set[tuple] = set()
    for m in matches:
        present = sorted(
            (cam, int(tid)) for cam, tid in m["tracks"].items() if tid is not None
        )
        for a in range(len(present)):
            for b in range(a + 1, len(present)):
                out.add((present[a], present[b]))
    return out


def build_matches(keys, cams, data, dist, threshold, min_covis, mutual, cam_names):
    n = len(keys)
    best_partner = [dict() for _ in range(n)]
    for i in range(n):
        di = data[keys[i]]
        for j in range(n):
            if cams[j] == cams[i] or dist[i, j] > threshold:
                continue
            if am._covisible_frames(di, data[keys[j]]) < min_covis:
                continue
            cur = best_partner[i].get(cams[j])
            if cur is None or dist[i, j] < cur[0]:
                best_partner[i][cams[j]] = (float(dist[i, j]), j)

    edges = []
    for i in range(n):
        for cam_b, (d_ij, j) in best_partner[i].items():
            back = best_partner[j].get(cams[i])
            if not mutual or (back is not None and back[1] == i):
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
        track_map = {name: None for name in cam_names}
        for m in members:
            cam_name, tid = keys[m]
            track_map[cam_name] = int(tid)
        matches.append({"frame": 0, "tracks": track_map})
    return matches


def main(folder: str):
    project = Project(folder)
    project.load()
    cam_names = project.camera_names

    if CACHE.exists():
        with open(CACHE, "rb") as f:
            data = pickle.load(f)
        print(f"loaded cached features ({len(data)} tracks)")
    else:
        from multicam_reid.reid import ReIDExtractor
        extractor = ReIDExtractor()
        data = am._collect_track_crops(project, extractor, 6, 24, 5)
        with open(CACHE, "wb") as f:
            pickle.dump(data, f)
        print(f"extracted + cached features ({len(data)} tracks)")

    keys = list(data.keys())
    cams = [k[0] for k in keys]
    features = np.stack([data[k]["feature"] for k in keys], axis=0)
    dist = np.clip(1.0 - features @ features.T, 0.0, None)

    gt = pairs_from_matches(
        tracks_io.load_matches(project.matches_path.parent / "matches.manual.backup.json")
    )
    print(f"ground-truth pairs: {len(gt)}\n")
    print(f"{'thr':>5} {'covis':>5} {'mut':>4} | {'pred':>5} {'TP':>4} {'FP':>4} "
          f"{'prec':>6} {'rec':>6} {'F1':>6}")
    print("-" * 60)

    for mutual in (True, False):
        for min_covis in (3, 5, 8):
            for threshold in (0.20, 0.25, 0.30, 0.35):
                matches = build_matches(keys, cams, data, dist, threshold,
                                        min_covis, mutual, cam_names)
                pred = pairs_from_matches(matches)
                tp = len(gt & pred)
                fp = len(pred - gt)
                prec = tp / (tp + fp) if (tp + fp) else 0.0
                rec = tp / len(gt) if gt else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
                print(f"{threshold:>5.2f} {min_covis:>5} {str(mutual):>4} | "
                      f"{len(pred):>5} {tp:>4} {fp:>4} "
                      f"{prec:>6.3f} {rec:>6.3f} {f1:>6.3f}")
        print()


if __name__ == "__main__":
    main(sys.argv[1])
