"""
Build the shared ground frame ONCE from the reliable manual matches and persist
it as the intersection's calibration (``.reid/ground_frame.json``).

This is a one-time step: once saved, the auto-matcher loads it and uses
real-world distance to gate candidate matches.

Run:
    python scripts/build_shared_frame.py <segment> [--matches manual|<path>] [--intrinsics <json>]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multicam_reid.core.project import Project
from multicam_reid.core import tracks_io
from multicam_reid.core.ground_bootstrap import (
    build_calib_by_cam,
    collect_correspondences,
    solve_shared_frame,
)
from multicam_reid.core.ground_align import save_ground_frame

DEFAULT_INTRINSICS = "/home/kojo/Downloads/intersection_cameras_intrinsics.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("segment")
    ap.add_argument("--matches", default="manual",
                    help="'manual' (matches.manual.backup.json) or a path")
    ap.add_argument("--intrinsics", default=DEFAULT_INTRINSICS)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--ransac-thr", type=float, default=0.05)
    args = ap.parse_args()

    project = Project(args.segment)
    project.load()

    calib_by_cam = build_calib_by_cam(project, args.intrinsics)
    if len(calib_by_cam) < 2:
        print("  Need calibration for at least 2 cameras.")
        return

    tracks = {c.name: tracks_io.load_tracks(project.tracks_path(c)) for c in project.cameras}

    reid_dir = project.matches_path.parent
    if args.matches == "manual":
        mpath = reid_dir / "matches.manual.backup.json"
    else:
        mpath = Path(args.matches)
    matches = tracks_io.load_matches(mpath)
    print(f"  matches source: {mpath.name}  ({len(matches)} clusters)")

    corr = collect_correspondences(tracks, calib_by_cam, matches, args.stride)
    frame, stats = solve_shared_frame(calib_by_cam, corr, args.ransac_thr)

    print(f"  reference camera: {frame.reference}")
    for cam, st in stats.items():
        if cam == frame.reference:
            print(f"    {cam:45s} REFERENCE")
        else:
            print(f"    {cam:45s} pts={st['pts']:4d}  inliers={st['inliers']:4d}  RMS={st['rms']:.4f}")

    out = reid_dir / "ground_frame.json"
    save_ground_frame(out, frame)
    print(f"\n  shared ground frame saved -> {out}")


if __name__ == "__main__":
    main()
