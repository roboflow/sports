"""Shared OpenCV text helpers used by annotations and player_motion."""

from __future__ import annotations

import cv2
import numpy as np


def cv2_safe_text(text: str) -> str:
    """OpenCV Hershey fonts only render ASCII; map common Unicode punctuation."""
    for src, dst in (
        ("\u00b7", " "),  # middle dot
        ("\u2192", "->"),  # right arrow
        ("\u2014", "-"),  # em dash
        ("\u2013", "-"),  # en dash
        ("\u2026", "..."),  # ellipsis
        ("\u00d7", "x"),  # multiplication sign
        ("\u2264", "<="),  # less-than or equal
        ("\u2265", ">="),  # greater-than or equal
        ("\u00b0", " deg"),  # degree sign
    ):
        text = text.replace(src, dst)
    return text.encode("ascii", "replace").decode("ascii")


def draw_text_shadow(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    font_scale: float = 0.7,
    color_bgr: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    shadow_offset: tuple[int, int] = (2, 2),
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    ascii_safe: bool = True,
) -> None:
    if ascii_safe:
        text = cv2_safe_text(text)
    x, y = org
    sx, sy = shadow_offset
    shadow_thickness = thickness + 1 if font == cv2.FONT_HERSHEY_SIMPLEX else thickness
    shadow_color = (12, 12, 12) if font == cv2.FONT_HERSHEY_SIMPLEX else (0, 0, 0)
    cv2.putText(
        frame,
        text,
        (x + sx, y + sy),
        font,
        font_scale,
        shadow_color,
        shadow_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        color_bgr,
        thickness,
        cv2.LINE_AA,
    )
