"""Pipelines for higher level sports video processing tasks."""

from .keyframes import (  # noqa: F401
    FOOTBALL,
    TENNIS,
    Keyframe,
    KeyframeGenerator,
    SportConfig,
)

__all__ = ["FOOTBALL", "TENNIS", "Keyframe", "KeyframeGenerator", "SportConfig"]
