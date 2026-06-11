"""
Manual synchronization data.

The manual sync tool lets a user scrub each camera independently to a common
visual instant, then records that as an "anchor". From the anchor we derive a
constant per-camera frame offset relative to a reference camera:

    offset[cam] = anchor_frame[cam] - anchor_frame[reference]

To find camera `cam`'s frame for a given reference-timeline frame `r`:

    frame[cam] = r + offset[cam]

The user can then mark one or more segments (in/out points on the reference
timeline) and export each as its own set of aligned clips.

sync.json schema (stored in <project>/.reid/sync.json):
    {
        "version": 1,
        "reference": "cam_north",
        "anchor": {"cam_north": 420, "cam_east": 408, "cam_west": 425},
        "offsets": {"cam_north": 0, "cam_east": -12, "cam_west": 5},
        "segments": [
            {"name": "segment_01", "ref_in": 100, "ref_out": 700,
             "output_fps": 10.0, "exported": true}
        ]
    }
"""

from __future__ import annotations

import json
from pathlib import Path

SYNC_VERSION = 1
SYNC_NAME = "sync.json"


def sync_path(project) -> Path:
    return project.workspace / SYNC_NAME


def default_sync(project) -> dict:
    """A fresh sync record assuming videos are not yet aligned (offsets 0)."""
    return {
        "version": SYNC_VERSION,
        "reference": project.camera_names[0] if project.camera_names else None,
        "anchor": {},
        "offsets": {name: 0 for name in project.camera_names},
        "segments": [],
    }


def load_sync(project) -> dict:
    """Load sync data for a project, or return a default record."""
    path = sync_path(project)
    if not path.exists():
        return default_sync(project)
    with open(path) as f:
        data = json.load(f)
    # Fill in any cameras missing from offsets (e.g. added later).
    data.setdefault("offsets", {})
    for name in project.camera_names:
        data["offsets"].setdefault(name, 0)
    data.setdefault("anchor", {})
    data.setdefault("segments", [])
    data.setdefault("reference", project.camera_names[0] if project.camera_names else None)
    return data


def save_sync(project, data: dict) -> None:
    """Persist sync data to <project>/.reid/sync.json."""
    project.ensure_workspace()
    data["version"] = SYNC_VERSION
    with open(sync_path(project), "w") as f:
        json.dump(data, f, indent=2)


def compute_offsets(anchor: dict[str, int], reference: str) -> dict[str, int]:
    """Derive per-camera offsets from an anchor and a reference camera."""
    if reference not in anchor:
        return {cam: 0 for cam in anchor}
    ref_frame = anchor[reference]
    return {cam: frame - ref_frame for cam, frame in anchor.items()}


def next_segment_name(segments: list[dict]) -> str:
    """Generate the next sequential segment name (segment_01, segment_02, ...)."""
    n = len(segments) + 1
    existing = {s.get("name") for s in segments}
    while f"segment_{n:02d}" in existing:
        n += 1
    return f"segment_{n:02d}"
