import colorsys

import cv2
import numpy as np
import supervision as sv

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch, draw_goals_on_pitch, player_ellipse_annotator
from sports.common.draw import draw_text_shadow, make_end_card
from sports.common.homography import valid_pitch_cm
from sports.common.kinematics import (
    DEFAULT_MIN_SPEED_PX,
    HOMOGRAPHY_PITCH_SMOOTH,
    PlayerTrack,
    _smooth_trajectory,
    feet_xy,
    player_mask,
)
from sports.common.view import ViewTransformer
from sports.configs.soccer import (
    BALL_CLASS_ID,
    REFEREE_CLASS_ID,
    TEAM_NONE,
    SoccerPitchConfiguration,
)

TEAM_COLORS = [sv.Color.from_hex("#FF1493"), sv.Color.from_hex("#00BFFF")]
NEUTRAL_COLOR = sv.Color.from_hex("#CCCCCC")

JOYSTICK_MAX_SPEED_PX = 4.0
JOYSTICK_ELLIPSE_THICKNESS = 2.0

RADAR_MINIMAP_SCALE = 0.065
RADAR_MINIMAP_PAD = 30
RADAR_MINIMAP_ALPHA = 0.6
SPOTLIGHT_RADIUS = 210
SPOTLIGHT_STRENGTH = 0.88
# ~18 km/h: highlight sprint-speed badges with a stronger chip border.
SPEED_SPRINT_MS = 5.0
_SPEED_BADGE_BG_BGR = (16, 18, 24)
_CHIP_FONT = cv2.FONT_HERSHEY_DUPLEX
_CHIP_VALUE_SCALE = 0.48
_CHIP_VALUE_THICK = 1
_CHIP_PAD_X = 3
_CHIP_PAD_Y = 2
_CHIP_RAIL_W = 2
_CHIP_TEXT_BGR = (240, 242, 248)


def team_color(team: int) -> sv.Color:
    """Return the palette color for team 0/1, else neutral."""
    if team in (0, 1):
        return TEAM_COLORS[team]
    return NEUTRAL_COLOR


def team_color_bgr(team: int) -> tuple[int, int, int]:
    """BGR tuple for OpenCV drawing from :func:`team_color`."""
    return team_color(team).as_bgr()


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


def draw_joystick_dots(
    frame: np.ndarray,
    detections: sv.Detections,
    joystick_smoother=None,
    *,
    speed_by_tid: dict | None = None,
    show_speed: bool = False,
    min_speed_ms: float = 0.0,
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
        color = team_color_bgr(team)
        x1, y1, x2, y2 = xyxy
        cx = (float(x1) + float(x2)) / 2.0
        cy = float(y2)
        a = float(x2 - x1)
        b = 0.35 * a
        radius = int(np.clip(round(a * 0.13), 3, 8))
        px, py = cx, cy
        vx, vy = float(kf_vx[i]), float(kf_vy[i])
        if np.isfinite(vx) and np.isfinite(vy):
            speed = float(np.hypot(vx, vy))
            if speed >= DEFAULT_MIN_SPEED_PX:
                if JOYSTICK_MAX_SPEED_PX <= DEFAULT_MIN_SPEED_PX:
                    stick = 1.0
                else:
                    linear = float(np.clip(
                        (speed - DEFAULT_MIN_SPEED_PX)
                        / (JOYSTICK_MAX_SPEED_PX - DEFAULT_MIN_SPEED_PX),
                        0.0,
                        1.0,
                    ))
                    stick = float(np.sqrt(linear))
                ux, uy = vx / speed, vy / speed
                denom = (b * ux) ** 2 + (a * uy) ** 2
                if denom < 1e-12:
                    edge = float(min(a, b))
                else:
                    edge = float((a * b) / np.sqrt(denom))
                reach = stick * (edge + 0.5 * JOYSTICK_ELLIPSE_THICKNESS + radius)
                px, py = cx + ux * reach, cy + uy * reach
        else:
            vx, vy = 0.0, 0.0
        tid = int(tids[i])
        if joystick_smoother is not None:
            px, py = joystick_smoother.smooth(tid, cx, cy, px, py)
        ipx, ipy = int(round(px)), int(round(py))
        cv2.circle(frame, (ipx, ipy), radius, color, -1, cv2.LINE_AA)
        if show_speed and speed_by_tid is not None and tid >= 0:
            spd = speed_by_tid.get(tid)
            if spd is not None and spd >= min_speed_ms:
                draw_speed_badge(
                    frame, float(spd), cx, cy, ipx, ipy, vx, vy,
                    team_bgr=color, dot_radius=radius,
                )


def _format_speed_value(speed_m_s: float) -> str:
    return f"{round(max(0.0, float(speed_m_s)), 1):.1f}"


def _chip_box_size(text: str) -> tuple[int, int, int, int]:
    (vw, vh), baseline = cv2.getTextSize(
        text, _CHIP_FONT, _CHIP_VALUE_SCALE, _CHIP_VALUE_THICK
    )
    box_w = vw + _CHIP_PAD_X * 2 + _CHIP_RAIL_W
    box_h = vh + baseline + _CHIP_PAD_Y * 2
    return box_w, box_h, vh, baseline


def _draw_chip(
    frame: np.ndarray,
    text: str,
    center: tuple[float, float],
    *,
    team_bgr: tuple[int, int, int],
    border_bgr: tuple[int, int, int] | None = None,
) -> None:
    box_w, box_h, vh, _baseline = _chip_box_size(text)
    bcx, bcy = center
    x0 = int(round(bcx - box_w * 0.5))
    y0 = int(round(bcy - box_h * 0.5))
    fh, fw = frame.shape[:2]
    x0 = int(np.clip(x0, 2, max(2, fw - box_w - 2)))
    y0 = int(np.clip(y0, 2, max(2, fh - box_h - 2)))
    x1, y1 = x0 + box_w, y0 + box_h
    if border_bgr is None:
        border_bgr = tuple(int(c * 0.7) for c in team_bgr)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), _SPEED_BADGE_BG_BGR, -1)
    cv2.rectangle(overlay, (x0, y0), (x0 + _CHIP_RAIL_W, y1), team_bgr, -1)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), border_bgr, 1, cv2.LINE_AA)
    frame[:] = cv2.addWeighted(overlay, 0.62, frame, 0.38, 0)
    draw_text_shadow(
        frame,
        text,
        (x0 + _CHIP_RAIL_W + _CHIP_PAD_X, y0 + _CHIP_PAD_Y + vh),
        font_scale=_CHIP_VALUE_SCALE,
        color_bgr=_CHIP_TEXT_BGR,
        thickness=_CHIP_VALUE_THICK,
        font=_CHIP_FONT,
    )


def _speed_badge_radial(
    cx: float, cy: float, px: float, py: float, vx: float, vy: float,
    *, min_speed_px: float = DEFAULT_MIN_SPEED_PX,
) -> tuple[float, float]:
    dx, dy = float(px - cx), float(py - cy)
    dist = float(np.hypot(dx, dy))
    if dist >= 1.0:
        return dx / dist, dy / dist
    speed = float(np.hypot(vx, vy))
    if np.isfinite(vx) and np.isfinite(vy) and speed >= min_speed_px:
        return vx / speed, vy / speed
    return 0.0, -1.0


def draw_speed_badge(
    frame: np.ndarray,
    speed_m_s: float,
    cx: float, cy: float, px: int, py: int, vx: float, vy: float,
    *,
    team_bgr: tuple[int, int, int],
    dot_radius: int,
    min_speed_px: float = DEFAULT_MIN_SPEED_PX,
) -> None:
    value = _format_speed_value(speed_m_s)
    _box_w, box_h, _vh, _baseline = _chip_box_size(value)
    ux, uy = _speed_badge_radial(
        cx, cy, float(px), float(py), vx, vy, min_speed_px=min_speed_px
    )
    outward = float(dot_radius) + 5.0 + box_h * 0.5
    bcx, bcy = float(px) + ux * outward, float(py) + uy * outward
    border = team_bgr if speed_m_s >= SPEED_SPRINT_MS else tuple(int(c * 0.7) for c in team_bgr)
    _draw_chip(frame, value, (bcx, bcy), team_bgr=team_bgr, border_bgr=border)


def draw_speed_legend(frame: np.ndarray) -> None:
    text = "speed  m/s"
    scale, thick = 0.42, 1
    (tw, th), baseline = cv2.getTextSize(text, _CHIP_FONT, scale, thick)
    pad_x, pad_y = 8, 5
    x0, y1 = 12, frame.shape[0] - 12
    y0 = y1 - th - baseline - pad_y * 2
    x1 = x0 + tw + pad_x * 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (16, 18, 24), -1)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (70, 72, 82), 1, cv2.LINE_AA)
    frame[:] = cv2.addWeighted(overlay, 0.62, frame, 0.38, 0)
    draw_text_shadow(
        frame, text, (x0 + pad_x, y1 - pad_y - baseline),
        font_scale=scale, color_bgr=(220, 222, 230), thickness=thick,
        font=_CHIP_FONT,
    )


def dim_frame(frame: np.ndarray, level: float = 0.22) -> np.ndarray:
    return np.clip(frame.astype(np.float32) * level, 0, 255).astype(np.uint8)


def spotlight(
    frame: np.ndarray,
    cx: int,
    cy: int,
    radius: int = SPOTLIGHT_RADIUS,
    strength: float = SPOTLIGHT_STRENGTH,
) -> np.ndarray:
    """Dim the frame, then restore the spotlighted player within a soft circle."""
    dimmed = dim_frame(frame)
    mask = np.zeros(frame.shape[:2], dtype=np.float32)
    cv2.circle(mask, (cx, cy), radius, 1.0, -1, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=radius * 0.38)
    mask = (mask[..., np.newaxis] * strength).astype(np.float32)
    out = dimmed.astype(np.float32) * (1.0 - mask) + frame.astype(np.float32) * mask
    return np.clip(out, 0, 255).astype(np.uint8)


def overlay_minimap(
    frame: np.ndarray,
    radar: np.ndarray,
    *,
    margin_x: int = 12,
    margin_y: int = 12,
    alpha: float = RADAR_MINIMAP_ALPHA,
) -> None:
    rh, rw = radar.shape[:2]
    fh, fw = frame.shape[:2]
    x0 = fw - rw - margin_x
    y0 = fh - rh - margin_y
    if x0 < 0 or y0 < 0:
        return
    roi = frame[y0:y0 + rh, x0:x0 + rw]
    cv2.addWeighted(radar, alpha, roi, 1.0 - alpha, 0, roi)


def draw_radar_minimap(
    frame: np.ndarray,
    detections: sv.Detections,
    transformer: ViewTransformer | None,
    *,
    minimap_scale: float = RADAR_MINIMAP_SCALE,
    padding: int = RADAR_MINIMAP_PAD,
    margin_x: int = 12,
    margin_y: int = 12,
    alpha: float = RADAR_MINIMAP_ALPHA,
    locked_goal_defenders: tuple[int, int] | None = None,
) -> np.ndarray:
    """Overlay a plain translucent radar minimap in the bottom-right corner."""
    if transformer is None:
        return frame
    config = SoccerPitchConfiguration()
    radar = draw_pitch(config=config, padding=padding, scale=minimap_scale)
    if locked_goal_defenders is not None:
        left_def, right_def = locked_goal_defenders
        if left_def in (0, 1) and right_def in (0, 1):
            radar = draw_goals_on_pitch(
                config,
                left_defender_team=left_def,
                right_defender_team=right_def,
                padding=padding,
                scale=minimap_scale,
                pitch=radar,
            )
    pmask = player_mask(detections)
    if pmask.any():
        pdet = detections[pmask]
        xy = feet_xy(pdet).astype(np.float32)
        xy_cm = transformer.transform_points(xy)
        teams = (
            detections.data.get("team")
            if detections.data is not None
            else np.full(len(detections), TEAM_NONE)
        )
        p_teams = teams[pmask] if teams is not None else np.full(pmask.sum(), TEAM_NONE)
        for tid_val in (0, 1):
            mask = p_teams == tid_val
            if mask.any():
                radar = draw_points_on_pitch(
                    config=config,
                    xy=xy_cm[mask],
                    face_color=TEAM_COLORS[tid_val],
                    edge_color=sv.Color.BLACK,
                    radius=int(18 * minimap_scale / 0.1),
                    padding=padding,
                    scale=minimap_scale,
                    pitch=radar,
                )
    overlay_minimap(frame, radar, margin_x=margin_x, margin_y=margin_y, alpha=alpha)
    return frame


def _format_distance_value(distance_m: float) -> str:
    return f"{max(0, int(round(float(distance_m))))} m"


def draw_distance_labels(
    frame: np.ndarray,
    detections: sv.Detections,
    distance_by_tid: dict[int, float],
) -> None:
    """Draw cumulative distance above each tracked player as a styled chip."""
    if len(detections) == 0 or detections.tracker_id is None:
        return
    teams = (
        detections.data.get("team", np.full(len(detections), TEAM_NONE))
        if detections.data else np.full(len(detections), TEAM_NONE)
    )
    for i, tid in enumerate(detections.tracker_id):
        tid = int(tid)
        if tid < 0:
            continue
        dist_m = distance_by_tid.get(tid)
        if dist_m is None:
            continue
        xyxy = detections.xyxy[i]
        x1, y1, x2 = xyxy[0], xyxy[1], xyxy[2]
        cx = (float(x1) + float(x2)) / 2.0
        label = _format_distance_value(dist_m)
        _, box_h, _vh, _baseline = _chip_box_size(label)
        cy = float(y1) - 8 - box_h * 0.5
        team = int(teams[i])
        team_bgr = team_color_bgr(team)
        _draw_chip(frame, label, (cx, cy), team_bgr=team_bgr)


def track_id_color(tid: int) -> tuple[int, int, int]:
    """Deterministic BGR color from tracker id."""
    hue = (tid * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_trace_on_minimap(
    radar: np.ndarray,
    trace_cm: np.ndarray,
    color_bgr: tuple[int, int, int],
    *,
    padding: int = RADAR_MINIMAP_PAD,
    scale: float = RADAR_MINIMAP_SCALE,
    thickness: int = 2,
    smooth_window: int | None = None,
    margin_cm: float = 80.0,
) -> np.ndarray:
    """Draw a pitch-cm polyline trace on a minimap image."""
    window = HOMOGRAPHY_PITCH_SMOOTH if smooth_window is None else smooth_window
    trace_cm = np.asarray(trace_cm, dtype=np.float64)
    if trace_cm.ndim == 2 and len(trace_cm):
        on_pitch = valid_pitch_cm(trace_cm, margin_cm=margin_cm)
        trace_cm = trace_cm[on_pitch]
        trace_cm = _smooth_trajectory(trace_cm, window)
    pts: list[tuple[int, int]] = []
    for pt in trace_cm:
        if np.any(np.isnan(pt)):
            continue
        px = int(pt[0] * scale) + padding
        py = int(pt[1] * scale) + padding
        pts.append((px, py))
    if len(pts) >= 2:
        cv2.polylines(radar, [np.array(pts, dtype=np.int32)], False, color_bgr, thickness, cv2.LINE_AA)
    if pts:
        cv2.circle(radar, pts[-1], 7, color_bgr, -1, cv2.LINE_AA)
        cv2.circle(radar, pts[-1], 7, (255, 255, 255), 1, cv2.LINE_AA)
    return radar


def build_trace_minimap(
    detections: sv.Detections,
    transformer: ViewTransformer | None,
    trace_by_tid: dict[int, list[np.ndarray]],
    focus_tid: int | None = None,
    *,
    config=None,
    scale: float = RADAR_MINIMAP_SCALE,
    padding: int = RADAR_MINIMAP_PAD,
    locked_goal_defenders: tuple[int, int] | None = None,
) -> np.ndarray:
    """Build a radar minimap with per-track colored traces and current player dots."""
    if focus_tid is not None:
        focus_tid = int(focus_tid)

    if config is None:
        config = SoccerPitchConfiguration()
    radar = draw_pitch(config=config, padding=padding, scale=scale)
    if locked_goal_defenders is not None:
        left_def, right_def = locked_goal_defenders
        if left_def in (0, 1) and right_def in (0, 1):
            radar = draw_goals_on_pitch(
                config,
                left_defender_team=left_def,
                right_defender_team=right_def,
                padding=padding,
                scale=scale,
                pitch=radar,
            )

    if focus_tid is None:
        trace_items = trace_by_tid.items()
    else:
        pts = trace_by_tid.get(focus_tid)
        trace_items = ((focus_tid, pts),) if pts is not None else ()
    for tid, pts in trace_items:
        if len(pts) < 2:
            continue
        trace = np.stack(pts, axis=0)
        color = track_id_color(int(tid))
        radar = draw_trace_on_minimap(radar, trace, color, padding=padding, scale=scale)

    if transformer is not None and len(detections):
        pmask = player_mask(detections)
        if pmask.any():
            pdet = detections[pmask]
            if focus_tid is not None and pdet.tracker_id is not None:
                focus_mask = pdet.tracker_id == focus_tid
                pdet = pdet[focus_mask] if focus_mask.any() else sv.Detections.empty()
            if len(pdet) == 0:
                return radar
            xy = feet_xy(pdet).astype(np.float32)
            xy_cm = transformer.transform_points(xy)
            on_pitch = valid_pitch_cm(xy_cm, config, margin_cm=80.0)
            tids_p = pdet.tracker_id if pdet.tracker_id is not None else np.full(len(pdet), -1)
            teams = (
                pdet.data.get("team", np.full(len(pdet), TEAM_NONE))
                if pdet.data else np.full(len(pdet), TEAM_NONE)
            )
            for i in range(len(pdet)):
                if not on_pitch[i]:
                    continue
                t_id = int(tids_p[i])
                team = int(teams[i])
                if team in (0, 1):
                    color = TEAM_COLORS[team].as_bgr()
                else:
                    color = track_id_color(t_id) if t_id >= 0 else (150, 150, 150)
                pt_cm = xy_cm[i]
                px = int(pt_cm[0] * scale) + padding
                py = int(pt_cm[1] * scale) + padding
                cv2.circle(radar, (px, py), 8, color, -1, cv2.LINE_AA)
                cv2.circle(radar, (px, py), 8, (255, 255, 255), 1, cv2.LINE_AA)
    return radar


def draw_distance_end_card(
    width: int,
    height: int,
    tracks: dict[int, PlayerTrack],
    *,
    n_top: int = 10,
) -> np.ndarray:
    """Black end-card with distance leaderboard."""
    card = make_end_card(width, height, bg_bgr=(0, 0, 0))
    title = "DISTANCE LEADERBOARD"
    cv2.putText(
        card, title, (40, 60), cv2.FONT_HERSHEY_DUPLEX, 1.2,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.line(card, (40, 80), (width - 40, 80), (120, 120, 120), 1)

    ranked = sorted(
        ((tid, t.distance_m) for tid, t in tracks.items()),
        key=lambda x: x[1],
        reverse=True,
    )[:n_top]

    y = 130
    for rank, (tid, dist_m) in enumerate(ranked, 1):
        label = f"#{rank:2d}   Track {tid:4d}   {dist_m:7.1f} m"
        color = (255, 215, 0) if rank == 1 else (200, 200, 200)
        cv2.putText(
            card, label, (60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 1, cv2.LINE_AA,
        )
        y += 46

    cv2.putText(
        card,
        "Distance measured via gated pitch homography",
        (40, height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (90, 90, 90),
        1,
        cv2.LINE_AA,
    )
    return card
