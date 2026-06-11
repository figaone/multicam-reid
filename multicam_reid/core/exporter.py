"""
ReID dataset exporter.

Turns confirmed cross-camera matches into a clean, shareable image dataset:
each match becomes a "global ID", and for every camera in that match we crop
the object across several sampled frames.

Output layout:
    <project>/.reid/export/
    |- id_0001/
    |  |- <cam_north>_f00123.jpg
    |  |- <cam_east>_f00130.jpg
    |  `- ...
    |- id_0002/
    |  `- ...
    `- export_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
from loguru import logger

from .project import Project
from . import tracks_io


def _sample_frames(frames: list[int], n: int) -> list[int]:
    """Pick up to n frames spread evenly across a track's frame list."""
    if not frames:
        return []
    if len(frames) <= n:
        return list(frames)
    step = len(frames) / n
    return [frames[int(i * step)] for i in range(n)]


def export_reid_dataset(
    project: Project,
    samples_per_camera: int = 5,
    min_box_size: int = 16,
    padding: float = 0.0,
) -> dict:
    """
    Export cropped images for every match into a per-global-ID folder.

    Args:
        project: An initialized Project (with matches and tracks).
        samples_per_camera: Number of frames to crop per camera per match.
        min_box_size: Skip crops smaller than this (in pixels, either side).
        padding: Fractional padding added around each box (0.1 = +10%).

    Returns:
        A summary dict (also written to export_summary.json).
    """
    project.load()
    matches = tracks_io.load_matches(project.matches_path)
    if not matches:
        raise ValueError("No matches to export. Create matches with the matcher first.")

    export_dir = project.workspace / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Cache loaded tracks and video captures per camera.
    tracks_by_cam = {
        cam.name: tracks_io.load_tracks(project.tracks_path(cam))
        for cam in project.cameras
    }
    caps = {cam.name: cv2.VideoCapture(str(project.video_path(cam))) for cam in project.cameras}

    summary = {"num_ids": 0, "num_crops": 0, "ids": []}
    try:
        for i, match in enumerate(matches):
            gid = f"id_{i + 1:04d}"
            id_dir = export_dir / gid
            id_dir.mkdir(parents=True, exist_ok=True)
            id_crops = 0
            id_entry = {"global_id": gid, "cameras": {}}

            for cam_name, tid in match.get("tracks", {}).items():
                if tid is None:
                    continue
                track = tracks_by_cam.get(cam_name, {}).get(int(tid))
                if not track:
                    continue
                cap = caps[cam_name]
                sampled = _sample_frames(track["frames"], samples_per_camera)
                cam_count = 0
                for frame_idx in sampled:
                    box = tracks_io.get_box_for_track(tracks_by_cam[cam_name], int(tid), frame_idx)
                    if box is None:
                        continue
                    crop = _crop(cap, frame_idx, box, padding, min_box_size)
                    if crop is None:
                        continue
                    out = id_dir / f"{cam_name}_f{frame_idx:06d}.jpg"
                    cv2.imwrite(str(out), crop)
                    cam_count += 1
                    id_crops += 1
                if cam_count:
                    id_entry["cameras"][cam_name] = {"track_id": int(tid), "crops": cam_count}

            if id_crops:
                summary["num_ids"] += 1
                summary["num_crops"] += id_crops
                summary["ids"].append(id_entry)
                logger.info(f"  {gid}: {id_crops} crops across {len(id_entry['cameras'])} cameras")
            else:
                id_dir.rmdir()  # remove empty id folder
    finally:
        for cap in caps.values():
            cap.release()

    with open(export_dir / "export_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(
        f"  Exported {summary['num_crops']} crops for {summary['num_ids']} IDs to {export_dir}"
    )
    return summary


def _crop(cap, frame_idx: int, box, padding: float, min_box_size: int):
    """Read a frame and return the (optionally padded) crop, or None."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    if padding > 0:
        bw, bh = (x2 - x1), (y2 - y1)
        x1 -= bw * padding
        x2 += bw * padding
        y1 -= bh * padding
        y2 += bh * padding
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if (x2 - x1) < min_box_size or (y2 - y1) < min_box_size:
        return None
    return frame[y1:y2, x1:x2]
