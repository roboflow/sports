import cv2
import numpy as np

ROBOFLOW_PURPLE_BGR = (249, 21, 131)  # #8315F9 in BGR


def ease_out_cubic(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return 1.0 - (1.0 - t) ** 3


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


def draw_hud_bar(
    frame: np.ndarray,
    title: str,
    *,
    height: int = 44,
    color_bgr: tuple[int, int, int] = ROBOFLOW_PURPLE_BGR,
) -> np.ndarray:
    """Semi-transparent top title bar."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, height), (18, 18, 22), -1)
    frame[:] = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)
    draw_text_shadow(
        frame, title, (14, 30), font_scale=0.75, color_bgr=color_bgr, thickness=2,
    )
    return frame


def draw_score_chip(
    frame: np.ndarray,
    text: str,
    center: tuple[int, int],
    *,
    bg_bgr: tuple[int, int, int],
) -> None:
    """Centered translucent label chip (e.g. turnover callouts)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.5, 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    cx, cy = center
    x0, y0 = cx - tw // 2 - 8, cy - th // 2 - 6
    x1, y1 = cx + tw // 2 + 8, cy + th // 2 + baseline + 6
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), bg_bgr, -1)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 255, 255), 1)
    frame[:] = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
    draw_text_shadow(
        frame, text, (x0 + 8, y0 + th + 4),
        font_scale=scale, color_bgr=(255, 255, 255), thickness=thick,
    )


def make_end_card(
    width: int,
    height: int,
    *,
    bg_bgr: tuple[int, int, int] = (18, 18, 18),
) -> np.ndarray:
    """Solid background canvas for post-clip summary cards."""
    return np.full((height, width, 3), bg_bgr, dtype=np.uint8)
