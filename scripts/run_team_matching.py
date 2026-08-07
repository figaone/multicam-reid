"""
Run cross-camera vehicle ID matching using the TEAM-SUPPLIED homographies as the
spatial gate, then render an inspection video:

    +-----------------------------------------------------------+
    |  Camera 1        |  Camera 2        |  Camera 3           |   <- colored ID boxes
    +-----------------------------------------------------------+
    |            BIRD'S-EYE SHARED MAP (team homography)         |   <- one dot per car,
    |     same global ID = same colour across all 3 cameras     |      coloured by ID
    +-----------------------------------------------------------+

A car matched across cameras shows up as 2-3 dots of the SAME colour sitting on
top of each other on the map -> visual proof the cross-camera ID is spatially
consistent.

Run from Code/:
    python scripts/run_team_matching.py <segment_folder> [--world-thr 40] [--frames 300]
"""

from __future__ import annotations

import argparse
import glob
import importlib
import os
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multicam_reid.core.project import Project
from multicam_reid.core import tracks_io

am = importlib.import_module("multicam_reid.reid.auto_match")


# --------------------------------------------------------------------------- #
# Team homography helpers
# --------------------------------------------------------------------------- #
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


def color_for_id(gid: int):
    np.random.seed(int(gid) % 256)
    c = np.random.randint(60, 256, 3).tolist()
    return (int(c[0]), int(c[1]), int(c[2]))  # BGR


# --------------------------------------------------------------------------- #
# Step 1: matching with the team-H spatial gate
# --------------------------------------------------------------------------- #
def run_matching(project_, cam_H, world_thr, app_thr):
    """Runs auto_match with team homographies as the world gate. Writes matches.json."""

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

    # cached appearance features (built by eval_team_homography.py)
    cache = Path("/tmp/team_homog_feats.npz")
    if cache.exists():
        blob = np.load(cache, allow_pickle=True)
        ckeys = [tuple(k) for k in blob["keys"]]
        feats = blob["feats"]; los = blob["los"]; his = blob["his"]; fsets = blob["fsets"]
        cached = {k: {"feature": feats[i], "lo": int(los[i]), "hi": int(his[i]),
                      "frame_set": set(int(x) for x in fsets[i])}
                  for i, k in enumerate(ckeys)}

        def collect_cached(project, extractor, *a, **k):
            return {kk: {kkk: (set(vv) if kkk == "frame_set" else vv)
                         for kkk, vv in v.items()}
                    for kk, v in cached.items()}
        am._collect_track_crops = collect_cached

        class DummyExtractor:
            def __init__(self, *a, **k):
                pass
        am.ReIDExtractor = DummyExtractor

    class DummyFrame:
        calib = cam_H
    import multicam_reid.core.ground_align as ga
    ga.load_ground_frame = lambda path: DummyFrame()

    matches = am.auto_match(
        project_,
        dist_threshold=app_thr,
        min_covis=3,
        mutual_only=True,
        ground_frame_path="__team__",
        max_world_dist=world_thr,
        require_world=True,
        world_min_shared=2,
    )
    return matches


# --------------------------------------------------------------------------- #
# Step 2: render the camera grid + bird's-eye map
# --------------------------------------------------------------------------- #
def build_world_transform(world_by_key, size, margin=40):
    """Fit a world->canvas transform from robust bounds of all world points."""
    allpts = np.array([p for wk in world_by_key.values() for p in wk.values()])
    lo = np.percentile(allpts, 2, axis=0)
    hi = np.percentile(allpts, 98, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    W, H = size
    sx = (W - 2 * margin) / span[0]
    sy = (H - 2 * margin) / span[1]
    s = min(sx, sy)

    def to_canvas(x, y):
        cx = margin + (x - lo[0]) * s
        cy = H - margin - (y - lo[1]) * s   # flip Y so "up" is up
        cx = min(max(cx, margin), W - margin)   # clamp so far-field stays visible
        cy = min(max(cy, margin), H - margin)
        return int(round(cx)), int(round(cy))

    return to_canvas


def render(project_, cam_H, matches, out_path, max_frames, panel_h=480):
    cams = [c for c in project_.cameras if c.name in cam_H]
    tracks = {c.name: tracks_io.load_tracks(project_.tracks_path(c)) for c in cams}

    # (cam, tid) -> global id
    key_to_gid = {}
    for gi, m in enumerate(matches, start=1):
        for cam_name, tid in m["tracks"].items():
            if tid is not None:
                key_to_gid[(cam_name, int(tid))] = gi

    # world trajectories per (cam, tid) for the map
    world_by_key = {}
    for c in cams:
        Hm = cam_H[c.name]
        for tid, tr in tracks[c.name].items():
            feet = [foot(b) for b in tr["boxes"]]
            w = project_pts(feet, Hm)
            world_by_key[(c.name, int(tid))] = {
                int(f): (float(w[i, 0]), float(w[i, 1]))
                for i, f in enumerate(tr["frames"])
            }

    # panel geometry
    panel_w, scale, x_off = {}, {}, {}
    x = 0
    for c in cams:
        in_w, in_h = int(c.width), int(c.height)
        ow = int(round(in_w * (panel_h / in_h)))
        panel_w[c.name] = ow
        scale[c.name] = (ow / in_w, panel_h / in_h)
        x_off[c.name] = x
        x += ow
    grid_w = x
    map_h = 560
    total_h = panel_h + map_h

    to_canvas = build_world_transform(world_by_key, (grid_w, map_h))

    caps = {c.name: cv2.VideoCapture(str(project_.folder / c.video)) for c in cams}
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, 10.0, (grid_w, total_h))

    # short recent trails on the map
    trail = {}  # key -> list of (cx, cy)

    fi = 0
    while fi < max_frames:
        frames = {}
        ok = True
        for c in cams:
            ret, fr = caps[c.name].read()
            if not ret:
                ok = False
                break
            frames[c.name] = cv2.resize(fr, (panel_w[c.name], panel_h))
        if not ok:
            break

        canvas = np.zeros((total_h, grid_w, 3), dtype=np.uint8)
        canvas[panel_h:, :] = (28, 28, 28)  # map background

        # faint map grid
        for gx in range(0, grid_w, 80):
            cv2.line(canvas, (gx, panel_h), (gx, total_h), (45, 45, 45), 1)
        for gy in range(panel_h, total_h, 80):
            cv2.line(canvas, (0, gy), (grid_w, gy), (45, 45, 45), 1)

        # camera panels + boxes
        for c in cams:
            canvas[:panel_h, x_off[c.name]:x_off[c.name] + panel_w[c.name]] = frames[c.name]
            sx, sy = scale[c.name]
            xo = x_off[c.name]
            for tid, tr in tracks[c.name].items():
                fr_list = tr["frames"]
                if fi not in fr_list:
                    continue
                b = tr["boxes"][fr_list.index(fi)]
                x1, y1 = int(b[0] * sx) + xo, int(b[1] * sy)
                x2, y2 = int(b[2] * sx) + xo, int(b[3] * sy)
                gid = key_to_gid.get((c.name, int(tid)))
                if gid is not None:
                    col = color_for_id(gid)
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 3)
                    cv2.putText(canvas, f"ID {gid}", (x1, max(y1 - 6, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                else:
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (110, 110, 110), 1)

        # camera labels
        for c in cams:
            lbl = c.name.split("_synced")[0]
            cv2.putText(canvas, lbl, (x_off[c.name] + 6, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if x_off[c.name] > 0:
                cv2.line(canvas, (x_off[c.name], 0), (x_off[c.name], panel_h), (0, 0, 0), 2)
        cv2.putText(canvas, "SHARED GROUND MAP (team homography) - same colour = same car across cameras",
                    (10, panel_h + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(canvas, f"Frame {fi}", (grid_w - 140, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # map dots — draw unmatched first (behind), matched on top
        map_offset = panel_h
        for (cam_name, tid), wk in world_by_key.items():
            if fi not in wk:
                continue
            x_, y_ = wk[fi]
            cx, cy = to_canvas(x_, y_)
            cy = cy + map_offset
            if key_to_gid.get((cam_name, int(tid))) is None:
                cv2.circle(canvas, (cx, cy), 4, (150, 150, 150), -1)
        for (cam_name, tid), wk in world_by_key.items():
            if fi not in wk:
                continue
            gid = key_to_gid.get((cam_name, int(tid)))
            if gid is None:
                continue
            x_, y_ = wk[fi]
            cx, cy = to_canvas(x_, y_)
            cy = cy + map_offset
            col = color_for_id(gid)
            tk = (cam_name, tid)
            trail.setdefault(tk, []).append((cx, cy))
            pts = trail[tk][-25:]
            for k in range(1, len(pts)):
                cv2.line(canvas, pts[k - 1], pts[k], col, 1)
            cv2.circle(canvas, (cx, cy), 8, col, -1)
            cv2.circle(canvas, (cx, cy), 8, (255, 255, 255), 1)
            cv2.putText(canvas, str(gid), (cx + 9, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        writer.write(canvas)
        fi += 1
        if fi % 50 == 0:
            print(f"  rendered {fi} frames ...")

    writer.release()
    for cap in caps.values():
        cap.release()
    return fi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--hdir", default=str(Path(__file__).resolve().parents[1] / "Fw_ Homography"))
    ap.add_argument("--world-thr", type=float, default=40.0)
    ap.add_argument("--app-thr", type=float, default=0.45)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    project_ = Project(args.folder)
    project_.load()
    H = load_team_H(args.hdir)
    cam_H = {c.name: H[cam_number(c.name)] for c in project_.cameras
             if cam_number(c.name) in H}
    print(f"Loaded team homographies for {len(cam_H)} cameras.")

    backup = project_.matches_path.parent / "matches.manual.backup.json"
    matches = run_matching(project_, cam_H, args.world_thr, args.app_thr)

    # keep a copy of the team result, then restore manual matches.json
    team_path = project_.matches_path.parent / "matches.team.json"
    shutil.copy(project_.matches_path, team_path)

    n_multi = sum(1 for m in matches
                  if sum(1 for t in m["tracks"].values() if t is not None) >= 2)
    n_tri = sum(1 for m in matches
                if sum(1 for t in m["tracks"].values() if t is not None) >= 3)
    print(f"Global IDs: {len(matches)} total, {n_multi} span >=2 cameras, {n_tri} span all 3.")

    out = args.out or str(project_.folder / "team_matched_tracks.mp4")
    n = render(project_, cam_H, matches, out, args.frames)
    print(f"Wrote {n} frames -> {out}")

    if backup.exists():
        shutil.copy(backup, project_.matches_path)
        print("Restored manual matches.json (team result saved as matches.team.json).")


if __name__ == "__main__":
    main()
