"""
multicam_reid — Cross-Camera Vehicle Re-Identification Annotation Toolkit

A self-contained tool for building cross-camera ground-truth matches:
  1. Point it at a folder of videos (one per camera).
  2. It runs YOLO + ByteTrack detection/tracking (cached automatically).
  3. Use the interactive matcher to link the same object across cameras.
  4. Optionally export a clean ReID dataset of cropped vehicle images.

Works with any number of cameras (2, 3, 4+) and any video filenames.
"""

__version__ = "1.0.0"
