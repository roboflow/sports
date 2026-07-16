"""OpenCV overlays for PASS_NETWORK: carrier, pass arrows, collaboration, end-card."""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

from sports.annotators.motion import (
    annotate_team_ellipses,
    draw_radar_minimap,
    team_color_bgr,
    team_ellipse_color_lookup,
)
from sports.common.draw import (
    ROBOFLOW_PURPLE_BGR,
    draw_hud_bar,
    draw_score_chip,
    draw_text_shadow,
    ease_out_cubic,
    make_end_card,
)
from sports.common.kinematics import feet_xy
from sports.common.pass_network import (
    PassNetwork,
    strongest_collaboration_pair,
)
from sports.common.pass_pitch import image_to_pitch_m, pitch_circle_to_image
from sports.common.passes import InferredPass, InferredTurnover, passes_for_overlay
from sports.common.possession import ball_xy
from sports.common.view import ViewTransformer
from sports.configs.soccer import BALL_CLASS_ID

TURNOVER_ARROW_BGR = (48, 168, 255)
CARRIER_SHADOW_BGR = (228, 228, 228)

_BALL_TRI = sv.TriangleAnnotator(
    color=sv.Color.from_hex("#FFD700"),
    base=18,
    height=14,
)
_TRACK_LABEL = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(["#FF1493", "#00BFFF", "#FF6347", "#FFD700"]),
    text_color=sv.Color.from_hex("#FFFFFF"),
    text_padding=4,
    text_thickness=1,
    text_scale=0.4,
    text_position=sv.Position.BOTTOM_CENTER,
)



def annotate_ball(frame: np.ndarray, dets: sv.Detections) -> np.ndarray:
    ball = ball_xy(dets)
    if ball is None:
        return frame
    x, y = float(ball[0]), float(ball[1])
    ball_dets = sv.Detections(
        xyxy=np.array([[x - 6, y - 6, x + 6, y + 6]], dtype=np.float32),
        class_id=np.array([BALL_CLASS_ID]),
    )
    return _BALL_TRI.annotate(frame, ball_dets)


def annotate_pass_players(
    frame: np.ndarray,
    dets: sv.Detections,
    *,
    show_tracker_ids: bool = True,
) -> np.ndarray:
    """Team ellipses plus optional tracker-id labels (skips the ball row)."""
    if len(dets) == 0:
        return frame
    players = dets
    if dets.class_id is not None:
        mask = dets.class_id != BALL_CLASS_ID
        players = dets[mask] if mask.any() else sv.Detections.empty()
    frame = annotate_team_ellipses(frame, players)
    if show_tracker_ids and players.tracker_id is not None and len(players):
        labels = [str(int(tid)) for tid in players.tracker_id]
        frame = _TRACK_LABEL.annotate(
            frame,
            players,
            labels,
            custom_color_lookup=team_ellipse_color_lookup(players),
        )
    return frame


def _draw_projected_ground_zone(
    frame: np.ndarray,
    poly: np.ndarray,
    color_bgr: tuple[int, int, int],
    *,
    alpha: float,
    thickness: int,
    filled: bool,
) -> None:
    if poly is None or len(poly) < 3:
        return
    if filled and alpha > 0.01:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [poly], color_bgr, lineType=cv2.LINE_AA)
        frame[:] = cv2.addWeighted(overlay, alpha * 0.45, frame, 1.0 - alpha * 0.45, 0)
    if thickness > 0 and alpha > 0.01:
        outline = frame.copy()
        cv2.polylines(
            outline,
            [poly],
            isClosed=True,
            color=color_bgr,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
        frame[:] = cv2.addWeighted(outline, alpha, frame, 1.0 - alpha, 0)


def _image_space_ground_ellipse(
    frame: np.ndarray,
    center: tuple[int, int],
    *,
    radius_px: int,
    color_bgr: tuple[int, int, int],
    alpha: float,
    thickness: int,
    filled: bool,
) -> None:
    cx, cy = int(center[0]), int(center[1])
    axes = (max(radius_px, 8), max(radius_px // 3, 4))
    if filled and alpha > 0.01:
        overlay = frame.copy()
        cv2.ellipse(overlay, (cx, cy), axes, 0, 0, 360, color_bgr, -1, cv2.LINE_AA)
        frame[:] = cv2.addWeighted(overlay, alpha * 0.45, frame, 1.0 - alpha * 0.45, 0)
    if thickness > 0 and alpha > 0.01:
        outline = frame.copy()
        cv2.ellipse(outline, (cx, cy), axes, 0, 0, 360, color_bgr, thickness, cv2.LINE_AA)
        frame[:] = cv2.addWeighted(outline, alpha, frame, 1.0 - alpha, 0)


def draw_carrier_ground_ellipse(
    frame: np.ndarray,
    center: tuple[float, float] | np.ndarray,
    *,
    transformer: ViewTransformer | None = None,
    color_bgr: tuple[int, int, int] = ROBOFLOW_PURPLE_BGR,
    radius_m: float = 0.55,
    radius_px: int = 28,
    alpha: float = 0.7,
    thickness: int = 2,
    filled: bool = True,
    pulse_t: float | None = None,
) -> bool:
    """Highlight the carrier zone as a pitch-space circle (ellipse on the turf)."""
    feet = np.asarray(center, dtype=np.float64).reshape(1, 2)
    center_px = (int(round(feet[0, 0])), int(round(feet[0, 1])))

    if pulse_t is not None:
        for i, base_r in enumerate((0.40, 0.56, 0.72)):
            phase = (pulse_t + i * 0.22) % 1.0
            wave = 0.5 + 0.5 * np.sin(phase * 2.0 * np.pi)
            ring_r = base_r * radius_m * (0.92 + 0.12 * wave)
            ring_alpha = 0.18 + 0.16 * wave
            if transformer is not None:
                center_m = image_to_pitch_m(feet, transformer)
                if center_m is not None:
                    poly = pitch_circle_to_image(center_m[0], ring_r, transformer)
                    _draw_projected_ground_zone(
                        frame, poly, color_bgr,
                        alpha=ring_alpha, thickness=2, filled=False,
                    )
                    continue
            ring_px = int(radius_px * (0.75 + 0.35 * (i + 1) / 3.0) * (0.92 + 0.12 * wave))
            _image_space_ground_ellipse(
                frame, center_px, radius_px=ring_px, color_bgr=color_bgr,
                alpha=ring_alpha, thickness=2, filled=False,
            )

    if transformer is not None:
        center_m = image_to_pitch_m(feet, transformer)
        if center_m is not None:
            poly = pitch_circle_to_image(center_m[0], radius_m, transformer)
            _draw_projected_ground_zone(
                frame, poly, color_bgr,
                alpha=alpha, thickness=thickness, filled=filled,
            )
            return True

    _image_space_ground_ellipse(
        frame, center_px, radius_px=radius_px, color_bgr=color_bgr,
        alpha=alpha, thickness=thickness, filled=filled,
    )
    return False


def draw_glow_arrow(
    frame: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color_bgr: tuple[int, int, int],
    *,
    thickness: int = 4,
    alpha: float = 1.0,
) -> None:
    if alpha <= 0.01:
        return
    layer = frame.copy()
    slim = thickness <= 3
    shadow = thickness + (2 if slim else 3)
    tip_len = 0.035 if slim else 0.05
    cv2.arrowedLine(layer, start, end, (20, 20, 20), shadow, cv2.LINE_AA, tipLength=tip_len)
    cv2.arrowedLine(layer, start, end, color_bgr, thickness, cv2.LINE_AA, tipLength=tip_len)
    if alpha >= 0.99:
        frame[:] = layer
    else:
        frame[:] = cv2.addWeighted(layer, alpha, frame, 1.0 - alpha, 0)



def _get_player_box(dets: sv.Detections, tid: int) -> np.ndarray | None:
    if dets.tracker_id is None:
        return None
    idx = np.flatnonzero(dets.tracker_id == tid)
    if len(idx) == 0:
        return None
    return dets.xyxy[idx[0]]


def _get_player_feet(dets: sv.Detections, tid: int) -> np.ndarray | None:
    if dets.tracker_id is None:
        return None
    idx = np.flatnonzero(dets.tracker_id == tid)
    if len(idx) == 0:
        return None
    return feet_xy(dets)[idx[0]]


def _feet_point(dets: sv.Detections, tid: int) -> tuple[int, int] | None:
    feet = _get_player_feet(dets, tid)
    if feet is None:
        return None
    return int(round(float(feet[0]))), int(round(float(feet[1])))


def _draw_ground_highlight(
    image: np.ndarray,
    dets: sv.Detections,
    tid: int,
    color_bgr: tuple[int, int, int],
    *,
    alpha: float = 1.0,
    scale: float = 1.0,
    transformer=None,
) -> None:
    feet = _get_player_feet(dets, tid)
    box = _get_player_box(dets, tid)
    if feet is None or box is None:
        return
    x0, _y0, x1, _y1 = box
    radius_m = 0.45 * scale
    radius_px = max(int((x1 - x0) * 0.35 * scale), 10)
    draw_carrier_ground_ellipse(
        image, feet, transformer=transformer, color_bgr=color_bgr,
        radius_m=radius_m, radius_px=radius_px, alpha=alpha, thickness=2, filled=True,
    )


def _draw_pass_highlights(
    image: np.ndarray,
    dets: sv.Detections,
    frame_idx: int,
    passes: tuple[InferredPass, ...],
    frame_rate: float,
    *,
    draw_player_halos: bool = True,
    transformer=None,
) -> None:
    passes = passes_for_overlay(passes)
    in_flight = [
        p for p in passes
        if p.frame_idx <= frame_idx <= p.frame_idx + p.gap_frames
    ]
    if len(in_flight) > 1:
        in_flight.sort(key=lambda p: p.frame_idx)
        in_flight = in_flight[:1]
    highlight_passes = tuple(in_flight) if in_flight else passes

    for p in highlight_passes:
        receive_idx = p.frame_idx + p.gap_frames
        twinkle_duration = int(frame_rate * 0.5)
        end_idx = receive_idx + twinkle_duration
        if not (p.frame_idx <= frame_idx <= end_idx):
            continue

        p_feet = _feet_point(dets, p.passer_tid)
        r_feet = _feet_point(dets, p.receiver_tid)
        color = team_color_bgr(p.team)

        if frame_idx <= receive_idx:
            t = (frame_idx - p.frame_idx) / p.gap_frames if p.gap_frames > 0 else 1.0
            pulse_alpha = 0.5 + 0.3 * np.sin(t * np.pi * 4)
            if draw_player_halos:
                _draw_ground_highlight(
                    image, dets, p.passer_tid, color,
                    alpha=pulse_alpha, transformer=transformer,
                )
                _draw_ground_highlight(
                    image, dets, p.receiver_tid, color,
                    alpha=pulse_alpha, transformer=transformer,
                )
            if p_feet is not None and r_feet is not None:
                ball_pos = ball_xy(dets)
                expected_x = p_feet[0] + (r_feet[0] - p_feet[0]) * t
                expected_y = p_feet[1] + (r_feet[1] - p_feet[1]) * t
                if ball_pos is not None:
                    dist_expected_to_recv = np.hypot(r_feet[0] - expected_x, r_feet[1] - expected_y)
                    dist_ball_to_recv = np.hypot(r_feet[0] - ball_pos[0], r_feet[1] - ball_pos[1])
                    if dist_ball_to_recv <= dist_expected_to_recv + 20:
                        tip = (int(ball_pos[0]), int(ball_pos[1]))
                    else:
                        tip = (int(expected_x), int(expected_y))
                else:
                    tip = (int(expected_x), int(expected_y))
                origin = (
                    int(p_feet[0] * (1 - t) + (tip[0] - (r_feet[0] - p_feet[0]) * t) * t),
                    int(p_feet[1] * (1 - t) + (tip[1] - (r_feet[1] - p_feet[1]) * t) * t),
                )
                if np.hypot(tip[0] - origin[0], tip[1] - origin[1]) > 5:
                    draw_glow_arrow(image, origin, tip, color, alpha=0.35)
        elif draw_player_halos:
            prog = (frame_idx - receive_idx) / twinkle_duration
            twinkle_val = np.sin(prog * 2 * 2 * np.pi)
            twinkle_alpha = max(0.0, twinkle_val)
            _draw_ground_highlight(
                image, dets, p.receiver_tid, color,
                alpha=twinkle_alpha * 0.9,
                scale=1.0 + 0.4 * twinkle_alpha,
                transformer=transformer,
            )


def _draw_turnover_notice(
    image: np.ndarray,
    dets: sv.Detections,
    turnover: InferredTurnover,
    frame_idx: int,
    frame_rate: float,
    *,
    transformer=None,
) -> None:
    hold_frames = max(10, int(round(1.2 * frame_rate)))
    start = turnover.interception_frame
    if frame_idx < start or frame_idx > start + hold_frames:
        return
    elapsed = frame_idx - start
    fade = 1.0 - ease_out_cubic(min(1.0, elapsed / max(hold_frames, 1)))
    h, w = image.shape[:2]
    pulse = 0.5 + 0.3 * np.sin(elapsed / max(frame_rate / 5, 1) * np.pi)
    _draw_ground_highlight(
        image, dets, turnover.passer_tid, team_color_bgr(turnover.passer_team),
        alpha=0.42 * fade, transformer=transformer,
    )
    _draw_ground_highlight(
        image, dets, turnover.interceptor_tid, team_color_bgr(turnover.interceptor_team),
        alpha=pulse * fade, scale=1.08, transformer=transformer,
    )
    feet = _get_player_feet(dets, turnover.interceptor_tid)
    if feet is not None:
        draw_carrier_ground_ellipse(
            image, feet, transformer=transformer, color_bgr=TURNOVER_ARROW_BGR,
            radius_m=0.52, alpha=pulse * fade * 0.85,
            pulse_t=min(1.0, elapsed / max(frame_rate * 0.2, 1)),
        )
    p_feet = _feet_point(dets, turnover.passer_tid)
    r_feet = _feet_point(dets, turnover.interceptor_tid)
    if p_feet is not None and r_feet is not None:
        draw_glow_arrow(image, p_feet, r_feet, TURNOVER_ARROW_BGR, thickness=3, alpha=0.4 * fade)
    draw_score_chip(
        image,
        f"TURNOVER  #{turnover.passer_tid} → #{turnover.interceptor_tid}",
        (w // 2, h - 36),
        bg_bgr=(18, 18, 22),
    )


def draw_collaboration_web(
    image: np.ndarray,
    dets: sv.Detections,
    frame_idx: int,
    passes: tuple[InferredPass, ...],
) -> None:
    """Draw completed-pass links between players still on screen."""
    connections: dict[tuple[int, int], dict] = {}
    for p in passes:
        receive_idx = p.frame_idx + p.gap_frames
        if receive_idx <= frame_idx:
            pair = tuple(sorted([p.passer_tid, p.receiver_tid]))
            if pair not in connections:
                connections[pair] = {"count": 0, "team": p.team}
            connections[pair]["count"] += 1
    if not connections:
        return
    overlay = image.copy()
    max_count = max(c["count"] for c in connections.values())
    for (t1, t2), data in connections.items():
        p1 = _feet_point(dets, t1)
        p2 = _feet_point(dets, t2)
        if p1 is None or p2 is None:
            continue
        intensity = min(data["count"] / max(max_count, 1), 1.0)
        alpha = 0.15 + (intensity * 0.45)
        thickness = 1 + int(intensity * 3)
        temp = image.copy()
        cv2.line(temp, p1, p2, team_color_bgr(data["team"]), thickness, cv2.LINE_AA)
        cv2.addWeighted(temp, alpha, overlay, 1.0 - alpha, 0, overlay)
    image[:] = overlay


def draw_pass_network_frame_overlays(
    image: np.ndarray,
    dets: sv.Detections,
    frame_idx: int,
    passes: tuple[InferredPass, ...],
    turnovers: tuple[InferredTurnover, ...],
    frame_rate: float,
    *,
    transformer=None,
) -> None:
    """In-flight pass highlights and turnover callouts for one frame."""
    _draw_pass_highlights(
        image, dets, frame_idx, passes, frame_rate, transformer=transformer,
    )
    for turnover in turnovers:
        _draw_turnover_notice(
            image, dets, turnover, frame_idx, frame_rate, transformer=transformer,
        )


def _player_teams(network: PassNetwork) -> dict[int, int]:
    teams: dict[int, int] = {}
    for player in network.players:
        teams[player.tracker_id] = player.team
    for link in network.links:
        teams.setdefault(link.passer_tid, link.team)
        teams.setdefault(link.receiver_tid, link.team)
    return teams


def _collaboration_graph_panel(
    card: np.ndarray,
    network: PassNetwork,
    *,
    origin: tuple[int, int],
    size: tuple[int, int],
) -> None:
    ox, oy = origin
    gw, gh = size
    if gw < 80 or gh < 80:
        return
    overlay = card.copy()
    cv2.rectangle(overlay, (ox, oy), (ox + gw, oy + gh), (28, 28, 34), -1)
    cv2.rectangle(overlay, (ox, oy), (ox + gw, oy + gh), (55, 55, 65), 1)
    card[:] = cv2.addWeighted(overlay, 0.92, card, 0.08, 0)
    draw_text_shadow(
        card, "COLLABORATION GRAPH", (ox + 14, oy + 28),
        font_scale=0.55, color_bgr=(200, 200, 210), thickness=1,
    )
    if not network.links:
        draw_text_shadow(
            card, "no inferred links", (ox + 14, oy + gh // 2),
            font_scale=0.5, color_bgr=(140, 140, 150), thickness=1,
        )
        return
    teams = _player_teams(network)
    nodes = sorted(
        {link.passer_tid for link in network.links}
        | {link.receiver_tid for link in network.links}
    )
    cx, cy = ox + gw // 2, oy + gh // 2 + 12
    radius = min(gw, gh) * 0.34
    positions: dict[int, tuple[int, int]] = {}
    for i, tid in enumerate(nodes):
        angle = 2.0 * np.pi * i / max(len(nodes), 1) - np.pi / 2
        positions[tid] = (
            int(cx + radius * np.cos(angle)),
            int(cy + radius * np.sin(angle)),
        )
    max_count = max(link.count for link in network.links)
    for link in network.links:
        p0 = positions.get(link.passer_tid)
        p1 = positions.get(link.receiver_tid)
        if p0 is None or p1 is None:
            continue
        intensity = min(link.count / max(max_count, 1), 1.0)
        thickness = 1 + int(intensity * 4)
        cv2.arrowedLine(
            card, p0, p1, team_color_bgr(link.team), thickness, cv2.LINE_AA, tipLength=0.08,
        )
    for tid, (x, y) in positions.items():
        color = team_color_bgr(teams.get(tid, -1))
        cv2.circle(card, (x, y), 16, color, -1, cv2.LINE_AA)
        cv2.circle(card, (x, y), 16, (255, 255, 255), 1, cv2.LINE_AA)
        draw_text_shadow(
            card, str(tid), (x - 10, y + 5),
            font_scale=0.4, color_bgr=(255, 255, 255), thickness=1,
        )


def draw_pass_network_end_card(
    size: tuple[int, int],
    network: PassNetwork,
    *,
    top_n: int = 5,
) -> np.ndarray:
    """Short summary card: top collaborators + small graph."""
    w, h = size
    card = make_end_card(w, h, bg_bgr=(18, 18, 18))
    draw_text_shadow(
        card, "PASS NETWORK", (40, 70),
        font_scale=1.4, color_bgr=ROBOFLOW_PURPLE_BGR, thickness=2,
    )
    mode = "metric lanes" if network.metric else "image-space lanes"
    draw_text_shadow(
        card,
        f"{network.n_passes} passes  |  {network.n_turnovers} turnovers  |  {mode}",
        (42, 108),
        font_scale=0.55, color_bgr=(170, 170, 170), thickness=1,
    )
    y_links = 160
    draw_text_shadow(
        card, "TOP COLLABORATORS", (40, y_links),
        font_scale=0.72, color_bgr=(120, 230, 120), thickness=2,
    )
    for rank, link in enumerate(network.links[:top_n], 1):
        line = (
            f"{rank}.  #{link.passer_tid} -> #{link.receiver_tid}"
            f"   {link.count} passes"
        )
        draw_text_shadow(
            card, line, (44, y_links + rank * 38),
            font_scale=0.62, color_bgr=team_color_bgr(link.team), thickness=2,
        )
    graph_x = int(w * 0.52)
    _collaboration_graph_panel(
        card, network, origin=(graph_x, 140), size=(w - graph_x - 40, h - 200),
    )
    strongest = strongest_collaboration_pair(network.links)
    if strongest is not None:
        tid_a, tid_b, _team, count = strongest
        draw_text_shadow(
            card,
            f"STRONGEST LINK:  #{tid_a} <-> #{tid_b}  ({count} passes)",
            (44, h - 60),
            font_scale=0.7, color_bgr=(120, 230, 120), thickness=2,
        )
    return card


RANK_COLORS_BGR = (
    (80, 180, 255),   # gold-ish BGR
    (200, 200, 200),  # silver
    (180, 130, 70),   # bronze
)
RANK_LABELS = ("1st", "2nd", "3rd")


def draw_pass_alternatives_overlay(
    frame: np.ndarray,
    dets: sv.Detections,
    event,
    *,
    revealed_options: int | None = None,
    reveal_progress: float = 1.0,
    transformer=None,
    locked_goal_defenders: tuple[int, int] | None = None,
    metric: bool = True,
) -> np.ndarray:
    """Dimmed freeze frame with ranked pass-lane arrows (reuses shared draw helpers)."""
    dim = (frame.astype(np.float32) * 0.32).astype(np.uint8)
    options = list(event.options)
    visible = options
    if revealed_options is not None:
        visible = options[: max(0, revealed_options)]

    dim = annotate_pass_players(dim, dets, show_tracker_ids=True)
    dim = annotate_ball(dim, dets)

    feet = feet_xy(dets)
    carrier_xy = feet[event.carrier.index]
    cx, cy = int(carrier_xy[0]), int(carrier_xy[1])
    draw_carrier_ground_ellipse(
        dim,
        carrier_xy,
        transformer=transformer,
        color_bgr=CARRIER_SHADOW_BGR,
        radius_m=0.55,
        alpha=0.55,
        filled=True,
        thickness=2,
    )

    if revealed_options == 0:
        draw_score_chip(dim, "ON BALL", (cx, cy - 42), bg_bgr=ROBOFLOW_PURPLE_BGR)
        return draw_hud_bar(dim, "PASS ALTERNATIVES")

    progress = float(np.clip(reveal_progress, 0.0, 1.0))
    for rank, option in enumerate(visible):
        color = RANK_COLORS_BGR[min(rank, len(RANK_COLORS_BGR) - 1)]
        recv_xy = feet[option.receiver_index]
        rx, ry = int(recv_xy[0]), int(recv_xy[1])
        is_new = rank == len(visible) - 1
        alpha = ease_out_cubic(progress) if is_new else 1.0
        draw_glow_arrow(dim, (cx, cy), (rx, ry), color, thickness=5, alpha=alpha)
        if alpha < 0.85:
            continue
        midx, midy = (cx + rx) // 2, (cy + ry) // 2
        label = RANK_LABELS[min(rank, len(RANK_LABELS) - 1)]
        chip = f"{label}  {option.score:.2f}"
        if metric:
            chip += f"  {option.length:.1f} m"
        draw_score_chip(dim, chip, (midx, midy), bg_bgr=color)

    dim = draw_radar_minimap(
        dim,
        dets,
        transformer,
        locked_goal_defenders=locked_goal_defenders,
    )
    return draw_hud_bar(dim, "PASS ALTERNATIVES  -  top open lanes")


# Re-export for runners that already use motion.draw_radar_minimap
__all__ = [
    "annotate_ball",
    "annotate_pass_players",
    "draw_carrier_ground_ellipse",
    "draw_collaboration_web",
    "draw_glow_arrow",
    "draw_hud_bar",
    "draw_pass_alternatives_overlay",
    "draw_pass_network_end_card",
    "draw_pass_network_frame_overlays",
    "draw_radar_minimap",
    "CARRIER_SHADOW_BGR",
]
