import cv2
import numpy as np


def draw_text_shadow(
    frame: np.ndarray,
    text: str,
    org: tuple,
    font_scale: float = 0.7,
    color_bgr: tuple = (255, 255, 255),
    thickness: int = 2,
    shadow_offset: tuple = (2, 2),
    font=cv2.FONT_HERSHEY_SIMPLEX,
) -> None:
    """
    Draw text with a dark shadow for readability on video frames.

    Args:
        frame (np.ndarray): BGR image to draw on (modified in place).
        text (str): Label text (ASCII).
        org (tuple): Bottom-left origin (x, y) for the foreground text.
        font_scale (float): OpenCV font scale.
        color_bgr (tuple): Foreground text color in BGR.
        thickness (int): Foreground stroke thickness.
        shadow_offset (tuple): Shadow displacement (dx, dy) from the origin.
        font: OpenCV font identifier.
    """
    x, y = org
    sx, sy = shadow_offset
    shadow_thickness = thickness + 1 if font == cv2.FONT_HERSHEY_SIMPLEX else thickness
    shadow_color = (12, 12, 12) if font == cv2.FONT_HERSHEY_SIMPLEX else (0, 0, 0)
    cv2.putText(
        frame, text, (x + sx, y + sy), font, font_scale,
        shadow_color, shadow_thickness, cv2.LINE_AA,
    )
    cv2.putText(
        frame, text, (x, y), font, font_scale,
        color_bgr, thickness, cv2.LINE_AA,
    )
