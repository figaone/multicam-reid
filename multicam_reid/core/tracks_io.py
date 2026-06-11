"""
Track and match I/O.

Track JSON schema (per camera):
    {
        "<track_id>": {
            "frames":  [int, ...],
            "boxes":   [[x1, y1, x2, y2], ...],   # pixel coords in original video
            "classes": [int, ...],                # COCO class ids
            "confs":   [float, ...],
            "class_name": str
        },
        ...
    }

Matches JSON schema:
    {
        "version": 1,
        "matches": [
            {
                "frame": int,                 # frame the match was made on (reference)
                "tracks": {                   # camera_name -> track_id (or null)
                    "cam_north": 12,
                    "cam_east": 7,
                    "cam_west": null
                }
            },
            ...
        ]
    }
"""

from __future__ import annotations

import json
from pathlib import Path

MATCHES_VERSION = 1


def load_tracks(path: Path) -> dict:
    """Load a per-camera tracks file. Track ids are kept as int keys."""
    with open(path) as f:
        raw = json.load(f)
    return {int(tid): track for tid, track in raw.items()}


def save_tracks(path: Path, tracks: dict) -> None:
    """Save a per-camera tracks file (keys serialized as strings)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {str(tid): track for tid, track in tracks.items()}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def load_matches(path: Path) -> list[dict]:
    """
    Load the matches file. Returns a list of match dicts of the form
    {"frame": int, "tracks": {cam_name: track_id_or_None}}.

    Accepts both the new schema (with "version"/"matches") and a bare list
    for forward/backward tolerance.
    """
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("matches", [])
    if isinstance(data, list):
        return data
    return []


def save_matches(path: Path, matches: list[dict]) -> None:
    """Save the matches file with version metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": MATCHES_VERSION, "matches": matches}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def get_boxes_at_frame(tracks: dict, frame_idx: int) -> list[dict]:
    """Return all visible track detections at a specific frame."""
    visible = []
    for tid, track in tracks.items():
        frames = track["frames"]
        # frames are stored in ascending order; a direct membership test is
        # simplest and correct for our data sizes.
        if frame_idx in frames:
            idx = frames.index(frame_idx)
            visible.append(
                {
                    "track_id": int(tid),
                    "box": track["boxes"][idx],
                    "class_name": track.get("class_name", "object"),
                    "conf": track["confs"][idx] if "confs" in track else 1.0,
                }
            )
    return visible


def get_box_for_track(tracks: dict, track_id: int, frame_idx: int):
    """Return the box for a specific track at a frame, or None if absent."""
    track = tracks.get(int(track_id))
    if not track:
        return None
    frames = track["frames"]
    if frame_idx in frames:
        return track["boxes"][frames.index(frame_idx)]
    return None
