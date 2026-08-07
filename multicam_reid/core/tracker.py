"""
Generic detection + tracking using YOLO + ByteTrack (via ultralytics).

Decoupled from any project layout: give it a video path and a model, get
back a tracks dict. The Project class decides where results are cached.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from loguru import logger

# Default COCO classes of interest (vehicles + people).
DEFAULT_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

VEHICLE_CLASS_NAMES = {"car", "motorcycle", "bus", "truck", "bicycle"}


def _box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return float(inter / (area_a + area_b - inter + 1e-6))


def _stabilize_track_boxes(
    boxes: list[list[float]],
    frames: list[int],
    max_ratio_per_frame: float = 1.65,
    min_iou_for_jump: float = 0.04,
) -> tuple[list[list[float]], int]:
    """Smooth jitter while rejecting implausible one-step teleport updates."""
    if len(boxes) <= 1:
        return boxes, 0

    stabilized: list[list[float]] = []
    prev_center: np.ndarray | None = None
    prev_size: np.ndarray | None = None
    prev_velocity = np.zeros(2, dtype=float)
    corrected = 0

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [float(v) for v in box]
        center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)
        size = np.array([max(1.0, x2 - x1), max(1.0, y2 - y1)], dtype=float)

        if prev_center is None or prev_size is None:
            blended_center = center
            blended_size = size
            frame_gap = 1
        else:
            frame_gap = max(1, int(frames[i] - frames[i - 1]))
            disp = center - prev_center
            dist = float(np.linalg.norm(disp))
            scale = max(prev_size[0], prev_size[1], size[0], size[1], 1.0)
            motion_ratio = dist / scale
            prev_box = stabilized[-1]
            iou = _box_iou(prev_box, [x1, y1, x2, y2])

            is_implausible_jump = (
                frame_gap <= 2
                and motion_ratio > (max_ratio_per_frame * frame_gap)
                and iou < min_iou_for_jump
            )

            if is_implausible_jump:
                corrected += 1
                # Keep trajectory continuity when tracker snaps to another object.
                blended_center = prev_center + prev_velocity * frame_gap
                blended_size = 0.75 * prev_size + 0.25 * size
            else:
                predicted = prev_center + prev_velocity * frame_gap
                blended_center = 0.68 * center + 0.32 * predicted
                blended_size = 0.7 * size + 0.3 * prev_size

        half_w = max(2.0, blended_size[0] / 2.0)
        half_h = max(2.0, blended_size[1] / 2.0)
        out_box = [
            float(blended_center[0] - half_w),
            float(blended_center[1] - half_h),
            float(blended_center[0] + half_w),
            float(blended_center[1] + half_h),
        ]
        stabilized.append(out_box)

        if prev_center is not None:
            prev_velocity = 0.5 * prev_velocity + 0.5 * ((blended_center - prev_center) / frame_gap)
        prev_center = blended_center
        prev_size = blended_size

    return stabilized, corrected


def run_tracker(
    video_path: Path,
    model,
    conf: float = 0.3,
    tracker: str = "bytetrack.yaml",
    classes: list[int] | None = None,
    class_names: dict[int, str] | None = None,
) -> dict:
    """
    Run detection + tracking on a single video.

    Args:
        video_path: Path to the video file.
        model: A loaded ultralytics YOLO model.
        conf: Detection confidence threshold.
        tracker: Tracker config name (e.g. "bytetrack.yaml", "botsort.yaml").
        classes: COCO class ids to keep. Defaults to DEFAULT_CLASSES keys.
        class_names: Mapping of class id -> human name for labelling.

    Returns:
        tracks dict: {track_id: {frames, boxes, classes, confs, class_name}}
    """
    if classes is None:
        classes = list(DEFAULT_CLASSES.keys())
    if class_names is None:
        class_names = DEFAULT_CLASSES

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()

    logger.info(f"  Tracking {video_path.name} ({total_frames} frames, {fps:.1f}fps)")

    results = model.track(
        source=str(video_path),
        conf=conf,
        classes=classes,
        tracker=tracker,
        stream=True,
        verbose=False,
        persist=True,
    )

    tracks: dict[int, dict] = {}
    frame_idx = 0
    for result in results:
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()

            per_track_state: dict[int, tuple[list[float], int, float]] = {}
            for box, tid, cls_id, conf_val in zip(boxes, track_ids, cls_ids, confs):
                tid = int(tid)
                per_track_state[tid] = ([float(x) for x in box], int(cls_id), float(conf_val))

            for tid, (box_vals, class_id, conf_val) in per_track_state.items():
                if tid not in tracks:
                    tracks[tid] = {
                        "frames": [],
                        "boxes": [],
                        "classes": [],
                        "confs": [],
                        "class_name": class_names.get(class_id, f"class_{class_id}"),
                    }

                tracks[tid]["frames"].append(frame_idx)
                tracks[tid]["boxes"].append(box_vals)
                tracks[tid]["classes"].append(class_id)
                tracks[tid]["confs"].append(conf_val)

        frame_idx += 1
        if total_frames and frame_idx % 1000 == 0:
            logger.info(f"    frame {frame_idx}/{total_frames}")

    logger.info(f"    done: {len(tracks)} tracks over {frame_idx} frames")

    corrected_total = 0
    for track in tracks.values():
        if len(track["boxes"]) >= 2:
            track["boxes"], corrected = _stabilize_track_boxes(track["boxes"], track["frames"])
            corrected_total += corrected

    if corrected_total:
        logger.info(f"    stabilized {corrected_total} implausible box jumps")

    return tracks


def load_model(model_path: str = "yolov8x.pt"):
    """
    Load a YOLO model. Imported lazily so the matcher and exporter work
    even in environments where ultralytics/torch are not installed.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "ultralytics is required for tracking. Install it with:\n"
            "    pip install ultralytics"
        ) from exc

    candidate = Path(model_path)
    if not candidate.exists():
        # Ultralytics will auto-download known weight names (e.g. yolov8x.pt).
        logger.info(f"  Model '{model_path}' not found locally; ultralytics will fetch it")
    logger.info(f"  Loading model: {model_path}")
    return YOLO(str(model_path))
