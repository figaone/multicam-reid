"""
Generic detection + tracking using YOLO + ByteTrack (via ultralytics).

Decoupled from any project layout: give it a video path and a model, get
back a tracks dict. The Project class decides where results are cached.
"""

from __future__ import annotations

from pathlib import Path

import cv2
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

            for box, tid, cls_id, conf_val in zip(boxes, track_ids, cls_ids, confs):
                tid = int(tid)
                if tid not in tracks:
                    tracks[tid] = {
                        "frames": [],
                        "boxes": [],
                        "classes": [],
                        "confs": [],
                        "class_name": class_names.get(int(cls_id), f"class_{int(cls_id)}"),
                    }
                tracks[tid]["frames"].append(frame_idx)
                tracks[tid]["boxes"].append([float(x) for x in box])
                tracks[tid]["classes"].append(int(cls_id))
                tracks[tid]["confs"].append(float(conf_val))

        frame_idx += 1
        if total_frames and frame_idx % 1000 == 0:
            logger.info(f"    frame {frame_idx}/{total_frames}")

    logger.info(f"    done: {len(tracks)} tracks over {frame_idx} frames")
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
