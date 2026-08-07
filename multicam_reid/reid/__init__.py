"""Appearance-based ReID subpackage (AI City Challenge style)."""

from .extractor import ReIDExtractor
from .auto_match import auto_match

__all__ = ["ReIDExtractor", "auto_match"]
