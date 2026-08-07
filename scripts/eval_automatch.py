"""
Diagnostic: how well do the ReID embeddings separate manually-matched
cross-camera track pairs from random cross-camera pairs?

Run:
    python scripts/eval_automatch.py <segment_folder>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multicam_reid.core.project import Project
from multicam_reid.core import tracks_io
from multicam_reid.reid.extractor import ReIDExtractor

import importlib
am = importlib.import_module("multicam_reid.reid.auto_match")


def main(folder: str):
    project = Project(folder)
    project.load()

    cache = Path("/tmp/automatch_feats.npz")
    if cache.exists():
        blob = np.load(cache, allow_pickle=True)
        keys = [tuple(k) for k in blob["keys"]]
        feats = blob["feats"]
        los = blob["los"]; his = blob["his"]
        data = {k: {"feature": feats[i], "lo": int(los[i]), "hi": int(his[i])}
                for i, k in enumerate(keys)}
    else:
        extractor = ReIDExtractor()
        data = am._collect_track_crops(project, extractor, samples_per_track=6,
                                       min_box_size=24, min_track_len=5)
        keys = list(data.keys())
        feats = np.stack([data[k]["feature"] for k in keys], axis=0)
        np.savez(cache,
                 keys=np.array([[k[0], k[1]] for k in keys], dtype=object),
                 feats=feats,
                 los=np.array([data[k]["lo"] for k in keys]),
                 his=np.array([data[k]["hi"] for k in keys]))

    index = {k: i for i, k in enumerate(keys)}
    cos = feats @ feats.T
    cos_dist = np.clip(1.0 - cos, 0.0, None)
    rr = am.re_ranking(feats)

    # Load manual matches -> set of positive cross-camera (key,key) pairs.
    manual = tracks_io.load_matches(
        project.matches_path.parent / "matches.manual.backup.json"
    )
    pos_pairs = []
    pos_set = set()
    for m in manual:
        present = [(cam, tid) for cam, tid in m["tracks"].items() if tid is not None]
        for a in range(len(present)):
            for b in range(a + 1, len(present)):
                ka = (present[a][0], int(present[a][1]))
                kb = (present[b][0], int(present[b][1]))
                if ka in index and kb in index:
                    pos_pairs.append((index[ka], index[kb]))
                    pos_set.add((index[ka], index[kb]))
                    pos_set.add((index[kb], index[ka]))

    if not pos_pairs:
        print("No manual positive pairs found among embedded tracks.")
        return

    pos_cos = np.array([cos_dist[i, j] for i, j in pos_pairs])
    pos_rr = np.array([rr[i, j] for i, j in pos_pairs])

    # Random cross-camera negatives.
    rng = np.random.default_rng(0)
    cams = np.array([k[0] for k in keys])
    neg_cos, neg_rr = [], []
    tries = 0
    while len(neg_cos) < 3000 and tries < 60000:
        i, j = rng.integers(0, len(keys), 2)
        tries += 1
        if cams[i] == cams[j]:
            continue
        neg_cos.append(cos_dist[i, j])
        neg_rr.append(rr[i, j])
    neg_cos = np.array(neg_cos)
    neg_rr = np.array(neg_rr)

    def summ(name, pos, neg):
        print(f"\n[{name}]")
        print(f"  positive  n={len(pos):4d}  mean={pos.mean():.3f}  "
              f"median={np.median(pos):.3f}  p90={np.percentile(pos,90):.3f}")
        print(f"  negative  n={len(neg):4d}  mean={neg.mean():.3f}  "
              f"median={np.median(neg):.3f}  p10={np.percentile(neg,10):.3f}")
        for t in (0.2, 0.3, 0.4, 0.45, 0.5, 0.6):
            tp = float((pos <= t).mean())
            fp = float((neg <= t).mean())
            print(f"  thr={t:.2f}  recall(pos<=t)={tp:5.2f}  fpr(neg<=t)={fp:5.2f}")

    summ("cosine distance", pos_cos, neg_cos)
    summ("k-reciprocal re-ranked", pos_rr, neg_rr)
    print(f"\nEmbedded tracks: {len(keys)}  |  manual positive pairs matched: {len(pos_pairs)}")

    # ---- rank-based analysis (is the true match the nearest neighbor?) ----
    cams = np.array([k[0] for k in keys])
    los = np.array([data[k]["lo"] for k in keys])
    his = np.array([data[k]["hi"] for k in keys])

    def temporally_ok(i, j, gap):
        lo = max(los[i], los[j]); hi = min(his[i], his[j])
        if hi >= lo:
            return True
        return (lo - hi) <= gap

    for name, dmat in (("cosine", cos_dist), ("re-ranked", rr)):
        r1 = r5 = mutual = total = 0
        for i, j in pos_pairs:
            # rank of j among cross-camera candidates queried from i
            cam_j = cams[j]
            cand = np.where(cams == cam_j)[0]
            order = cand[np.argsort(dmat[i, cand])]
            rank_ij = int(np.where(order == j)[0][0])
            # reverse
            cam_i = cams[i]
            cand2 = np.where(cams == cam_i)[0]
            order2 = cand2[np.argsort(dmat[j, cand2])]
            rank_ji = int(np.where(order2 == i)[0][0])
            total += 1
            if rank_ij == 0:
                r1 += 1
            if rank_ij < 5:
                r5 += 1
            if rank_ij == 0 and rank_ji == 0:
                mutual += 1
        print(f"\n[rank analysis (ALL cross-cam candidates): {name}]  (n={total})")
        print(f"  rank-1: {r1/total:.2f}   rank-5: {r5/total:.2f}   mutual-NN: {mutual/total:.2f}")

    # ---- realistic: rank within TEMPORALLY-GATED candidates ----
    for gap in (10, 20, 40):
        for name, dmat in (("cosine", cos_dist),):
            r1 = r5 = mutual = total = 0
            cand_sizes = []
            for i, j in pos_pairs:
                cam_j = cams[j]
                cand = np.array([c for c in np.where(cams == cam_j)[0]
                                 if temporally_ok(i, c, gap)])
                if j not in cand:
                    cand = np.append(cand, j)
                cand_sizes.append(len(cand))
                order = cand[np.argsort(dmat[i, cand])]
                rank_ij = int(np.where(order == j)[0][0])
                # reverse gate
                cam_i = cams[i]
                cand2 = np.array([c for c in np.where(cams == cam_i)[0]
                                  if temporally_ok(j, c, gap)])
                if i not in cand2:
                    cand2 = np.append(cand2, i)
                order2 = cand2[np.argsort(dmat[j, cand2])]
                rank_ji = int(np.where(order2 == i)[0][0])
                total += 1
                if rank_ij == 0:
                    r1 += 1
                if rank_ij < 5:
                    r5 += 1
                if rank_ij == 0 and rank_ji == 0:
                    mutual += 1
            print(f"\n[TEMPORAL-GATE gap={gap}: {name}]  (n={total}, "
                  f"avg candidates={np.mean(cand_sizes):.1f})")
            print(f"  rank-1: {r1/total:.2f}   rank-5: {r5/total:.2f}   mutual-NN: {mutual/total:.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
