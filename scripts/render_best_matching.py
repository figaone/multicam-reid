"""
Best-quality cross-camera vehicle IDing + inspection video.

Uses the calibrated SHARED GROUND FRAME (.reid/ground_frame.json) for both:
  * matching  -> auto_match's spatial gate (appearance + real-world distance),
                 the high-precision configuration (~0.76 precision on set_12),
  * the map   -> every track's foot point projected into the shared frame
                 (undistort -> per-camera ground homography -> shared similarity),
                 which yields clean, coincident dots for the same car.

Layout:
    +-----------------------------------------------------------+
    |  Camera 1        |  Camera 2        |  Camera 3           |   colored ID boxes
    +-----------------------------------------------------------+
    |              SHARED GROUND MAP (calibrated)               |   one dot per car,
    |     same global ID = same colour across all 3 cameras     |   colour = global ID
    +-----------------------------------------------------------+

Run from Code/:
    python scripts/render_best_matching.py <segment_folder> [--frames 400]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multicam_reid.core.project import Project
from multicam_reid.core import tracks_io
from multicam_reid.core.ground_align import load_ground_frame
from multicam_reid.reid.auto_match import auto_match


def color_for_id(gid: int):
    """Distinct, bright BGR colour per global ID (golden-angle hue spread)."""
    h = (int(gid) * 47) % 180
    hsv = np.uint8([[[h, 210, 245]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
    return (int(b), int(g), int(r))


def _foot(box):
    x1, y1, x2, y2 = box
    return [(x1 + x2) * 0.5, float(y2)]


def track_world(track, ground_frame, cam_name, stride=1):
    """{frame -> shared-frame (x, y)} for a track via the frame's own projector."""
    frames, boxes = track["frames"], track["boxes"]
    idx = range(0, len(frames), stride)
    feet = [_foot(boxes[i]) for i in idx]
    fr = [frames[i] for i in idx]
    if not feet:
        return {}
    w = ground_frame.project_feet(cam_name, feet)
    return {int(fr[k]): (float(w[k, 0]), float(w[k, 1]))
            for k in range(len(fr)) if np.isfinite(w[k]).all()}


def build_world_transform(world_by_key, size, margin=50):
    allpts = np.array([p for wk in world_by_key.values() for p in wk.values()])
    lo = np.percentile(allpts, 4, axis=0)
    hi = np.percentile(allpts, 96, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    W, H = size
    s = min((W - 2 * margin) / span[0], (H - 2 * margin) / span[1])
    cx0 = (W - span[0] * s) / 2.0
    cy0 = (H - span[1] * s) / 2.0

    def to_canvas(x, y):
        cx = cx0 + (x - lo[0]) * s
        cy = H - cy0 - (y - lo[1]) * s
        cx = min(max(cx, 4), W - 4)
        cy = min(max(cy, 4), H - 4)
        return int(round(cx)), int(round(cy))

    return to_canvas


def render(project_, ground_frame, matches, out_path, max_frames, panel_h=520):
    cams = [c for c in project_.cameras if ground_frame.has(c.name)]
    tracks = {c.name: tracks_io.load_tracks(project_.tracks_path(c)) for c in cams}

    key_to_gid = {}
    for gi, m in enumerate(matches, start=1):
        for cam_name, tid in m["tracks"].items():
            if tid is not None:
                key_to_gid[(cam_name, int(tid))] = gi

    # world trajectories for the map
    world_by_key = {}
    for c in cams:
        for tid, tr in tracks[c.name].items():
            world_by_key[(c.name, int(tid))] = track_world(tr, ground_frame, c.name, stride=1)

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
    map_h = 600
    total_h = panel_h + map_h
    to_canvas = build_world_transform(world_by_key, (grid_w, map_h))

    caps = {c.name: cv2.VideoCapture(str(project_.folder / c.video)) for c in cams}
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, 10.0, (grid_w, total_h))
    trail = {}

    fi = 0
    while fi < max_frames:
        frames, ok = {}, True
        for c in cams:
            ret, fr = caps[c.name].read()
            if not ret:
                ok = False
                break
            frames[c.name] = cv2.resize(fr, (panel_w[c.name], panel_h))
        if not ok:
            break

        canvas = np.zeros((total_h, grid_w, 3), dtype=np.uint8)
        canvas[panel_h:, :] = (32, 32, 34)
        for gx in range(0, grid_w, 90):
            cv2.line(canvas, (gx, panel_h), (gx, total_h), (50, 50, 52), 1)
        for gy in range(panel_h, total_h, 90):
            cv2.line(canvas, (0, gy), (grid_w, gy), (50, 50, 52), 1)

        # camera panels
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
                    label = f"ID {gid}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(canvas, (x1, max(y1 - th - 8, 0)), (x1 + tw + 6, max(y1, th + 8)), col, -1)
                    cv2.putText(canvas, label, (x1 + 3, max(y1 - 5, th + 2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                else:
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (120, 120, 120), 1)

        for c in cams:
            lbl = c.name.split("_synced")[0]
            cv2.putText(canvas, lbl, (x_off[c.name] + 8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(canvas, lbl, (x_off[c.name] + 8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2)
            if x_off[c.name] > 0:
                cv2.line(canvas, (x_off[c.name], 0), (x_off[c.name], panel_h), (0, 0, 0), 2)

        cv2.putText(canvas, "SHARED GROUND MAP (calibrated) - same colour = same car across cameras",
                    (12, panel_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (210, 210, 210), 1)
        cv2.putText(canvas, f"Frame {fi}", (grid_w - 150, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # map: unmatched behind, matched on top
        for (cam_name, tid), wk in world_by_key.items():
            if fi in wk and key_to_gid.get((cam_name, int(tid))) is None:
                cx, cy = to_canvas(*wk[fi]); cy += panel_h
                cv2.circle(canvas, (cx, cy), 4, (150, 150, 150), -1)
        for (cam_name, tid), wk in world_by_key.items():
            gid = key_to_gid.get((cam_name, int(tid)))
            if gid is None or fi not in wk:
                continue
            cx, cy = to_canvas(*wk[fi]); cy += panel_h
            col = color_for_id(gid)
            tk = (cam_name, tid)
            trail.setdefault(tk, []).append((cx, cy))
            pts = trail[tk][-30:]
            for k in range(1, len(pts)):
                cv2.line(canvas, pts[k - 1], pts[k], col, 2)
            cv2.circle(canvas, (cx, cy), 9, col, -1)
            cv2.circle(canvas, (cx, cy), 9, (255, 255, 255), 1)
            cv2.putText(canvas, str(gid), (cx + 10, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        writer.write(canvas)
        fi += 1
        if fi % 50 == 0:
            print(f"  rendered {fi} frames ...")

    writer.release()
    for cap in caps.values():
        cap.release()
    return fi, grid_w, total_h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--max-world-dist", type=float, default=None,
                    help="World-distance gate. Default: 0.75 for the calibrated bootstrap "
                         "frame, 6.0 for a homography frame (MapAnything).")
    ap.add_argument("--ground-frame", default=None,
                    help="Path to the shared frame. Default: <.reid>/ground_frame.json. "
                         "Point at a homography-type JSON (e.g. MapAnything) to run anchor-free.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--reuse-matches", action="store_true",
                    help="Render from an existing matches.best.json instead of re-matching.")
    args = ap.parse_args()

    project_ = Project(args.folder)
    project_.load()

    gf_path = Path(args.ground_frame) if args.ground_frame else \
        project_.matches_path.parent / "ground_frame.json"
    if not gf_path.exists():
        raise SystemExit(f"No shared frame at {gf_path}. Build it first "
                         f"(scripts/build_shared_frame.py or mapanything_to_ground_frame.py).")
    ground_frame = load_ground_frame(gf_path)

    from multicam_reid.core.ground_align import HomographyGroundFrame
    is_homography = isinstance(ground_frame, HomographyGroundFrame)
    max_world = args.max_world_dist if args.max_world_dist is not None else \
        (6.0 if is_homography else 0.75)
    print(f"Shared frame: {gf_path.name}  ({'homography/auto' if is_homography else 'calibrated bootstrap'}), "
          f"max_world_dist={max_world}")

    backup = project_.matches_path.parent / "matches.manual.backup.json"
    best_path = project_.matches_path.parent / "matches.best.json"

    if args.reuse_matches and best_path.exists():
        print(f"Reusing existing matches from {best_path.name} ...")
        matches = tracks_io.load_matches(best_path)
    else:
        print("Running high-precision matching (appearance + calibrated spatial gate) ...")
        matches = auto_match(
            project_,
            dist_threshold=args.threshold,
            min_covis=3,
            mutual_only=True,
            ground_frame_path=str(gf_path),
            max_world_dist=max_world,
            require_world=True,
            world_min_shared=2,
        )
        shutil.copy(project_.matches_path, best_path)
        if backup.exists():
            shutil.copy(backup, project_.matches_path)
            print("Restored manual matches.json (best result saved as matches.best.json).")

    n_multi = sum(1 for m in matches if sum(1 for t in m["tracks"].values() if t is not None) >= 2)
    n_tri = sum(1 for m in matches if sum(1 for t in m["tracks"].values() if t is not None) >= 3)
    print(f"Global IDs: {len(matches)} total, {n_multi} span >=2 cameras, {n_tri} span all 3.")

    out = args.out or str(project_.folder / "best_matched_tracks.mp4")
    n, w, h = render(project_, ground_frame, matches, out, args.frames)
    print(f"Wrote {n} frames ({w}x{h}) -> {out}")


if __name__ == "__main__":
    main()
