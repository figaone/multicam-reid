"""
Score the automatic matcher against the manual ground-truth matches.

Compares cross-camera track *pairs*: for each global ID, every pair of
(camera, track) members is a positive. We report pairwise precision / recall /
F1 of the automatic matches vs the manual backup.

Run:
    python scripts/score_automatch.py <segment_folder>
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multicam_reid.core.project import Project
from multicam_reid.core import tracks_io


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


def main(folder: str):
    project = Project(folder)
    project.load()

    manual_path = project.matches_path.parent / "matches.manual.backup.json"
    gt = pairs_from_matches(tracks_io.load_matches(manual_path))
    pred = pairs_from_matches(tracks_io.load_matches(project.matches_path))

    if not gt:
        print("No manual ground-truth pairs found.")
        return

    tp = len(gt & pred)
    fp = len(pred - gt)
    fn = len(gt - pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"  ground-truth pairs : {len(gt)}")
    print(f"  predicted pairs    : {len(pred)}")
    print(f"  true positives     : {tp}")
    print(f"  false positives    : {fp}")
    print(f"  false negatives    : {fn}")
    print(f"  precision          : {precision:.3f}")
    print(f"  recall             : {recall:.3f}")
    print(f"  F1                 : {f1:.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
