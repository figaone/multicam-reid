"""
Calibrate the spatial gate: measure shared-frame foot-point distances for
KNOWN-POSITIVE cross-camera pairs (manual matches) vs co-visible NEGATIVE pairs.
Tells us where to set ``max_world_dist``.

Run:
    python scripts/calibrate_world_gate.py <segment>
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multicam_reid.core.project import Project
from multicam_reid.core import tracks_io
from multicam_reid.core.ground_align import load_ground_frame
from multicam_reid.core.ground_bootstrap import project_track_feet

DEFAULT_STRIDE = 2


def track_world(track, calib, S, stride=DEFAULT_STRIDE):
    ground = project_track_feet(track, calib, stride)
    out = {}
    for f, g in ground.items():
        h = np.array([g[0], g[1], 1.0])
        w = S @ h
        out[f] = (w[0] / w[2], w[1] / w[2])
    return out


def med_dist(wa, wb, min_shared=2):
    shared = set(wa) & set(wb)
    if len(shared) < min_shared:
        return None
    ds = [((wa[f][0] - wb[f][0]) ** 2 + (wa[f][1] - wb[f][1]) ** 2) ** 0.5 for f in shared]
    ds.sort()
    return ds[len(ds) // 2]


def main(folder):
    project = Project(folder)
    project.load()
    reid = project.matches_path.parent
    gf = load_ground_frame(reid / "ground_frame.json")

    tracks = {c.name: tracks_io.load_tracks(project.tracks_path(c)) for c in project.cameras}
    world_cache = {}

    def get_world(cam, tid):
        key = (cam, tid)
        if key not in world_cache:
            if cam not in gf.calib or tid not in tracks[cam]:
                world_cache[key] = {}
            else:
                world_cache[key] = track_world(tracks[cam][tid], gf.calib[cam], gf.sim[cam])
        return world_cache[key]

    # positives from manual matches
    manual = tracks_io.load_matches(reid / "matches.manual.backup.json")
    pos = []
    pos_pairs = set()
    for m in manual:
        present = [(cam, int(tid)) for cam, tid in m["tracks"].items() if tid is not None]
        for (ca, ta), (cb, tb) in combinations(present, 2):
            pos_pairs.add(((ca, ta), (cb, tb)))
            d = med_dist(get_world(ca, ta), get_world(cb, tb))
            if d is not None:
                pos.append(d)

    # negatives: random co-visible cross-camera pairs NOT in positives
    rng = np.random.default_rng(0)
    all_keys = [(c.name, tid) for c in project.cameras for tid in tracks[c.name]]
    neg = []
    tries = 0
    while len(neg) < 4000 and tries < 200000:
        tries += 1
        a = all_keys[rng.integers(len(all_keys))]
        b = all_keys[rng.integers(len(all_keys))]
        if a[0] == b[0]:
            continue
        if (a, b) in pos_pairs or (b, a) in pos_pairs:
            continue
        d = med_dist(get_world(*a), get_world(*b))
        if d is not None:
            neg.append(d)

    pos = np.array(pos); neg = np.array(neg)
    print(f"  positive pairs with world dist: {len(pos)}")
    print(f"  negative pairs with world dist: {len(neg)}\n")
    for name, arr in [("POSITIVE (same vehicle)", pos), ("NEGATIVE (diff vehicle)", neg)]:
        if len(arr):
            qs = np.percentile(arr, [50, 75, 90, 95, 99])
            print(f"  {name:26s} median={qs[0]:.4f}  p75={qs[1]:.4f}  "
                  f"p90={qs[2]:.4f}  p95={qs[3]:.4f}  p99={qs[4]:.4f}")

    print("\n  gate recall (pos kept) / negatives kept, by threshold:")
    for thr in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]:
        pk = (pos <= thr).mean() if len(pos) else 0
        nk = (neg <= thr).mean() if len(neg) else 0
        print(f"    thr={thr:5.2f}  pos_kept={pk:5.2f}  neg_kept={nk:5.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
