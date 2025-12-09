"""Skull stripping (brain extraction) implementations.

This module provides skull stripping operations using HD-BET and SynthStrip
algorithms from brainles_preprocessing package.
"""

from mengrowth.preprocessing.src.skull_stripping.base import BaseSkullStripper
from mengrowth.preprocessing.src.skull_stripping.hdbet import HDBetSkullStripper
from mengrowth.preprocessing.src.skull_stripping.synthstrip import SynthStripSkullStripper

__all__ = [
    "BaseSkullStripper",
    "HDBetSkullStripper",
    "SynthStripSkullStripper",
]
