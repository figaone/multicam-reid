"""
High-quality anti-aliased text rendering for the OpenCV windows.

OpenCV's built-in Hershey font looks jagged and "blocky". This module renders
text with a real TrueType font (DejaVu Sans) via Pillow, with a dark outline so
labels stay readable on any background — giving the tools a polished, native
look.

`draw_text` keeps the same call signature the tools already use
(`draw_text(img, text, org, scale=..., color=..., weight=...)`), so it is a
drop-in replacement for the old cv2-based helper. `org` is the bottom-left of
the text, matching cv2.putText semantics. Colors are BGR (cv2 convention).

For speed, only the small image region under each label is converted to/from
Pillow, so this stays fast even with many labels per frame.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Candidate font files, in preference order.
_REGULAR_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/home/kojo/.config/Ultralytics/Arial.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]
_BOLD_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


@lru_cache(maxsize=64)
def _load_font(size: int, bold: bool) -> ImageFont.FreeTypeFont:
    candidates = _BOLD_CANDIDATES + _REGULAR_CANDIDATES if bold else _REGULAR_CANDIDATES
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, ValueError):
            continue
    return ImageFont.load_default()


def _scale_to_px(scale: float) -> int:
    """Map a legacy cv2 font scale to a TrueType pixel size."""
    return max(11, int(round(scale * 38)))


def measure_text(text: str, scale: float = 0.55, weight: int = 1) -> tuple[int, int]:
    """Return the (width, height) in pixels of text at the given scale."""
    font = _load_font(_scale_to_px(scale), bold=weight >= 2)
    left, top, right, bottom = font.getbbox(text)
    return right - left, bottom - top


def draw_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.55,
    color: tuple[int, int, int] = (255, 255, 255),
    weight: int = 1,
) -> np.ndarray:
    """
    Draw crisp anti-aliased text with a dark outline onto a BGR image in place.

    Args:
        img: BGR image (modified in place).
        text: The string to draw.
        org: Bottom-left corner of the text (cv2.putText convention).
        scale: Legacy cv2 font scale; mapped to a TrueType size.
        color: BGR color.
        weight: >=2 uses a bold face.
    """
    if not text:
        return img

    size = _scale_to_px(scale)
    bold = weight >= 2
    font = _load_font(size, bold)
    stroke = max(1, size // 11)

    x, y = int(org[0]), int(org[1])
    left, top, right, bottom = font.getbbox(text, stroke_width=stroke)
    text_w = right - left
    text_h = bottom - top
    top_y = y - text_h  # convert baseline-ish bottom-left to top-left

    pad = stroke + 2
    h, w = img.shape[:2]
    x0 = max(0, x - pad)
    y0 = max(0, top_y - pad)
    x1 = min(w, x + text_w + pad)
    y1 = min(h, y + pad)
    if x1 <= x0 or y1 <= y0:
        return img

    region = img[y0:y1, x0:x1]
    pil = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    rgb = (color[2], color[1], color[0])
    draw.text(
        (x - x0 - left, top_y - y0 - top),
        text,
        font=font,
        fill=rgb,
        stroke_width=stroke,
        stroke_fill=(0, 0, 0),
    )
    region[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img
