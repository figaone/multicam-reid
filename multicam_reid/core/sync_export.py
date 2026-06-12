"""
Export manually-aligned segments as synchronized video clips.

Given a reference in/out window and per-camera offsets, writes one clip per
camera where output frame k corresponds to the same instant across cameras.
Each segment lands in its own folder so it can be opened directly as a new
project (for tracking + matching):

    <project>/.reid/synced/<segment_name>/<camera>_synced.mp4
"""

from __future__ import annotations

from pathlib import Path

import cv2
from loguru import logger


def export_segment(
    project,
    reference: str,
    offsets: dict[str, int],
    ref_in: int,
    ref_out: int,
    name: str,
    output_fps: float | None = None,
) -> Path:
    """
    Write aligned clips for one segment.

    Args:
        project: An initialized Project.
        reference: Reference camera name (defines the timeline).
        offsets: cam_name -> frame offset relative to the reference.
        ref_in, ref_out: Segment bounds on the reference timeline (frames).
        name: Segment folder name (e.g. "segment_01").
        output_fps: Output frame rate. Defaults to the reference camera's fps.

    Returns:
        The output folder path containing the per-camera clips.
    """
    if ref_out <= ref_in:
        raise ValueError(f"Segment OUT ({ref_out}) must be greater than IN ({ref_in}).")

    ref_cam = next((c for c in project.cameras if c.name == reference), None)
    if ref_cam is None:
        raise ValueError(f"Reference camera '{reference}' not found in project.")

    fps = output_fps or ref_cam.fps or 10.0

    # Trim the requested window to a range where EVERY camera has a real source
    # frame, so all output clips are exactly the same length with no duplicate /
    # frozen padding frames at the edges. For camera c at offset o, the source
    # frame for reference frame r is (r + o); it is valid only while
    # 0 <= r + o <= total_c - 1, i.e. -o <= r <= total_c - 1 - o.
    totals = {}
    for cam in project.cameras:
        cap = cv2.VideoCapture(str(project.video_path(cam)))
        if not cap.isOpened():
            cap.release()
            raise IOError(f"Could not open video for camera '{cam.name}'")
        totals[cam.name] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    valid_lo = max([ref_in] + [-offsets.get(c.name, 0) for c in project.cameras])
    valid_hi = min(
        [ref_out]
        + [totals[c.name] - 1 - offsets.get(c.name, 0) for c in project.cameras]
    )
    if valid_hi - valid_lo < 1:
        raise ValueError(
            "No overlapping frames across all cameras for this segment "
            "(check the offsets / segment bounds)."
        )
    if valid_lo != ref_in or valid_hi != ref_out:
        logger.info(
            f"  Trimmed segment to common range {valid_lo}..{valid_hi} "
            f"(was {ref_in}..{ref_out}) so all cameras have equal length."
        )
    ref_in, ref_out = valid_lo, valid_hi
    n_frames = ref_out - ref_in

    out_dir = project.workspace / "synced" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"  Exporting '{name}': {n_frames} frames @ {fps:.2f}fps -> {out_dir}")

    for cam in project.cameras:
        offset = offsets.get(cam.name, 0)
        cap = cv2.VideoCapture(str(project.video_path(cam)))
        if not cap.isOpened():
            cap.release()
            raise IOError(f"Could not open video for camera '{cam.name}'")
        total = totals[cam.name]

        out_path = out_dir / f"{cam.name}_synced.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (cam.width, cam.height))

        prev_src = -1
        cached = None
        written = 0
        for k in range(n_frames):
            src = ref_in + k + offset
            src = max(0, min(src, total - 1))  # clamp (should not trigger after trim)
            if src != prev_src:
                cap.set(cv2.CAP_PROP_POS_FRAMES, src)
                ret, frame = cap.read()
                if ret:
                    cached = frame
                prev_src = src
            if cached is not None:
                writer.write(cached)
                written += 1

        cap.release()
        writer.release()
        logger.info(f"    {cam.name}: offset {offset:+d} -> {out_path.name} "
                    f"({written} frames)")

    return out_dir
