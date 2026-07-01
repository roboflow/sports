import cv2
import numpy as np
import supervision as sv

from sports.common.kinematics import DEFAULT_MIN_SPEED_PX
from sports.annotators.soccer import player_ellipse_annotator
from sports.configs.soccer import (
    BALL_CLASS_ID,
    REFEREE_CLASS_ID,
    TEAM_NONE,
)

TEAM_COLORS = [sv.Color.from_hex("#FF1493"), sv.Color.from_hex("#00BFFF")]
NEUTRAL_COLOR = sv.Color.from_hex("#CCCCCC")

JOYSTICK_MIN_SPEED_PX = 0.5
JOYSTICK_MAX_SPEED_PX = 4.0
JOYSTICK_ELLIPSE_THICKNESS = 2.0


def _team_color(team: int) -> sv.Color:
    if team in (0, 1):
        return TEAM_COLORS[team]
    return NEUTRAL_COLOR


def team_ellipse_color_lookup(detections: sv.Detections) -> np.ndarray:
    """Map each detection row to a PLAYER_VIS_COLORS palette index."""
    n = len(detections)
    lookup = np.full(n, 2, dtype=int)
    if n == 0 or detections.data is None:
        return lookup
    teams = detections.data.get("team", np.full(n, TEAM_NONE))
    for i in range(n):
        if int(detections.class_id[i]) == REFEREE_CLASS_ID:
            lookup[i] = REFEREE_CLASS_ID
        elif int(teams[i]) in (0, 1):
            lookup[i] = int(teams[i])
    return lookup


def annotate_team_ellipses(
    frame: np.ndarray,
    detections: sv.Detections,
) -> np.ndarray:
    """Draw team ellipses via the shared player ellipse annotator."""
    if len(detections) == 0:
        return frame
    return player_ellipse_annotator.annotate(
        scene=frame,
        detections=detections,
        custom_color_lookup=team_ellipse_color_lookup(detections),
    )


def _dot_radius_for_ellipse(semi_axis_a: float) -> int:
    return int(np.clip(round(semi_axis_a * 0.13), 3, 8))


def _ellipse_extent_in_direction(a: float, b: float, ux: float, uy: float) -> float:
    denom = (b * ux) ** 2 + (a * uy) ** 2
    if denom < 1e-12:
        return float(min(a, b))
    return float((a * b) / np.sqrt(denom))


def _joystick_dot_reach(
    stick: float,
    a: float,
    b: float,
    ux: float,
    uy: float,
    dot_radius: float,
    ellipse_thickness: float = JOYSTICK_ELLIPSE_THICKNESS,
) -> float:
    edge = _ellipse_extent_in_direction(a, b, ux, uy)
    outer = edge + 0.5 * ellipse_thickness + dot_radius
    return float(stick) * outer


def kalman_speed_stick(
    speed_px: float,
    min_speed_px: float = JOYSTICK_MIN_SPEED_PX,
    max_speed_px: float = JOYSTICK_MAX_SPEED_PX,
):
    """Map Kalman speed (px/frame) to a joystick deflection in [0, 1]."""
    if not np.isfinite(speed_px) or speed_px < min_speed_px:
        return None
    if max_speed_px <= min_speed_px:
        return 1.0
    linear = float(np.clip((speed_px - min_speed_px) / (max_speed_px - min_speed_px), 0.0, 1.0))
    return float(np.sqrt(linear))


def draw_joystick_dots(
    frame: np.ndarray,
    detections: sv.Detections,
    joystick_smoother=None,
) -> None:
    """Draw a team-colored directional velocity dot on each player."""
    if len(detections) == 0 or detections.data is None:
        return
    kf_vx = detections.data.get("kf_vx")
    kf_vy = detections.data.get("kf_vy")
    if kf_vx is None or kf_vy is None:
        return
    teams = detections.data.get("team", np.full(len(detections), TEAM_NONE))
    tids = (
        detections.tracker_id
        if detections.tracker_id is not None
        else np.full(len(detections), -1)
    )
    for i, xyxy in enumerate(detections.xyxy):
        cls = int(detections.class_id[i])
        if cls in (BALL_CLASS_ID, REFEREE_CLASS_ID):
            continue
        team = int(teams[i])
        if team not in (0, 1):
            continue
        color = _team_color(team).as_bgr()
        x1, y1, x2, y2 = xyxy
        cx = (float(x1) + float(x2)) / 2.0
        cy = float(y2)
        a = float(x2 - x1)
        b = 0.35 * a
        radius = _dot_radius_for_ellipse(a)
        px, py = cx, cy
        vx, vy = float(kf_vx[i]), float(kf_vy[i])
        if np.isfinite(vx) and np.isfinite(vy):
            speed = float(np.hypot(vx, vy))
            if speed >= DEFAULT_MIN_SPEED_PX:
                stick = kalman_speed_stick(speed)
                if stick is not None:
                    ux, uy = vx / speed, vy / speed
                    reach = _joystick_dot_reach(
                        stick, a, b, ux, uy, dot_radius=float(radius),
                    )
                    px, py = cx + ux * reach, cy + uy * reach
        tid = int(tids[i])
        if joystick_smoother is not None:
            px, py = joystick_smoother.smooth(tid, cx, cy, px, py)
        ipx, ipy = int(round(px)), int(round(py))
        cv2.circle(frame, (ipx, ipy), radius, color, -1, cv2.LINE_AA)
