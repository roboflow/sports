"""Ball-carrier detection, possession config thresholds, and touch validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import supervision as sv

from sports.configs.soccer import (
    BALL_CLASS_ID as ROLE_BALL,
    GOALKEEPER_CLASS_ID as ROLE_GOALKEEPER,
    PLAYER_CLASS_ID as ROLE_PLAYER,
)
from sports.common.geometry import unit
from sports.common.kinematics import feet_xy, player_mask

from sports.common.pass_pitch import image_to_pitch_m

# Tight dribble at the feet — used for pass passer logic and lane-scoring freezes.
CONTROL_MAX_DISTANCE_PX = 55.0
CONTROL_MAX_DISTANCE_M = 0.8

# Looser first-touch gate — pass detection only (control is tried first).
RECEPTION_MAX_DISTANCE_PX = 100.0
RECEPTION_MAX_DISTANCE_M = 1.5

# Vertical offset in image space above/below feet that vetoes ground possession.
AERIAL_DY_THRESHOLD_PX = 20.0

# Aliases for possession modules that use CARRIER_* threshold names.
CARRIER_MAX_DISTANCE_PX = CONTROL_MAX_DISTANCE_PX
CARRIER_MAX_DISTANCE_M = CONTROL_MAX_DISTANCE_M


def bbox_center_xy(detections: sv.Detections) -> np.ndarray:
    """BBox center — better for lane blocking than feet when players lean across a pass."""
    boxes = detections.xyxy
    return np.stack(
        [(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2],
        axis=1,
    )


def ball_xy(detections: sv.Detections) -> np.ndarray | None:
    """Return the ball's ground position, or ``None`` if no ball this frame."""
    mask = detections.class_id == ROLE_BALL
    if not mask.any():
        return None
    boxes = detections.xyxy[mask]
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = boxes[:, 3]
    return np.stack([cx, cy], axis=1)[0]


@dataclass(frozen=True)
class Carrier:
    index: int          # row index into the detections
    team: int
    distance: float     # pixels (v1) or meters (v2) from ball to carrier feet
    ball: np.ndarray    # (2,) ball ground position


def find_ball_carrier(
    detections: sv.Detections,
    *,
    max_distance_px: float = CARRIER_MAX_DISTANCE_PX,
    transformer=None,
    max_distance_m: float = CARRIER_MAX_DISTANCE_M,
    require_both_spaces: bool = False,
) -> Carrier | None:
    """Nearest player to the ball, if within range (pixels or pitch meters).

    When ``transformer`` is set, we check both metric and pixel distance. By default
    a player is valid if they are within the limit in **either** space (robust to brief
    homography glitches). With ``require_both_spaces=True``, both must pass — used for
    pass-alternative freeze picking so warped metric alone cannot extend possession.
    """
    ball = ball_xy(detections)
    if ball is None:
        return None
    pmask = player_mask(detections)
    if not pmask.any():
        return None

    feet_img = feet_xy(detections)[pmask]
    global_indices = np.flatnonzero(pmask)
    roles = detections.class_id[pmask]

    # Always calculate pixel distance as a robust fallback
    # To prevent aerial balls from triggering false possession, we heavily penalize 
    # balls that are significantly above OR below the feet in the 2D image.
    dx = feet_img[:, 0] - ball[0]
    dy = feet_img[:, 1] - ball[1]
    
    # If the ball is significantly displaced on the Y-axis (abs(dy) > 20px), 
    # it is likely an aerial pass (either behind or in front of the player).
    # We use this as a strict veto against the 3D metric projection.
    is_aerial = np.abs(dy) > 20
    
    # We stretch the effective Y distance to prevent pixel-fallback fly-bys.
    dy_penalty = np.where(np.abs(dy) > 10, dy * 2.5, dy)
    
    dist_px = np.hypot(dx, dy_penalty)
    limit_px = np.full(len(dist_px), max_distance_px, dtype=np.float32)
    limit_px[roles == ROLE_GOALKEEPER] = max_distance_px * 2.5

    pixel_valid = dist_px <= limit_px
    valid_mask = pixel_valid
    dist_to_use = dist_px

    if transformer is not None:
        feet_m = image_to_pitch_m(feet_img, transformer)
        ball_m = image_to_pitch_m(np.array([ball], dtype=np.float32), transformer)
        if feet_m is not None and ball_m is not None:
            dist_m = np.linalg.norm(feet_m - ball_m[0], axis=1)
            limit_m = np.full(len(dist_m), max_distance_m, dtype=np.float32)
            limit_m[roles == ROLE_GOALKEEPER] = max_distance_m * 3.5

            # Veto metric when the ball is clearly aerial in 2D (flat-ground assumption).
            metric_valid = (dist_m <= limit_m) & ~is_aerial
            if require_both_spaces:
                valid_mask = pixel_valid & metric_valid
            else:
                valid_mask = pixel_valid | metric_valid
            dist_to_use = dist_m

    if not valid_mask.any():
        return None

    # Find the closest player among the valid candidates
    # We use dist_to_use (metric if available, else px) to pick the 'closest'
    # We set invalid players to infinity so they aren't chosen
    dist_to_use[~valid_mask] = float('inf')
    local = int(np.argmin(dist_to_use))
    
    global_idx = int(global_indices[local])
    team = int(detections.data["team"][global_idx])
    return Carrier(global_idx, team, float(dist_to_use[local]), ball)


def carrier_from_tracker_id(
    detections: sv.Detections,
    tracker_id: int,
) -> Carrier | None:
    """Build a :class:`Carrier` for a known player row (e.g. inferred passer at release).

    Does not require the ball to be at their feet — used for pass-alternative freezes on
    the release frame, when control-range carrier lookup would already have failed.
    """
    if tracker_id < 0 or detections.tracker_id is None:
        return None
    pmask = player_mask(detections)
    rows = np.flatnonzero(pmask & (detections.tracker_id == tracker_id))
    if len(rows) == 0:
        return None
    idx = int(rows[0])
    ball = ball_xy(detections)
    feet = feet_xy(detections)[idx]
    if ball is None:
        ball = feet
    dist = float(np.linalg.norm(feet - ball))
    team = int(detections.data["team"][idx])
    return Carrier(idx, team, dist, np.asarray(ball, dtype=np.float64))


def find_control_carrier(
    detections: sv.Detections,
    *,
    transformer=None,
    max_distance_px: float = CONTROL_MAX_DISTANCE_PX,
    max_distance_m: float = CONTROL_MAX_DISTANCE_M,
    require_both_spaces: bool = False,
) -> Carrier | None:
    """Nearest player within tight dribble range (pass passer / lane-scoring gate)."""
    return find_ball_carrier(
        detections,
        max_distance_px=max_distance_px,
        transformer=transformer,
        max_distance_m=max_distance_m,
        require_both_spaces=require_both_spaces,
    )


def find_reception_carrier(
    detections: sv.Detections,
    *,
    transformer=None,
    max_distance_px: float = RECEPTION_MAX_DISTANCE_PX,
    max_distance_m: float = RECEPTION_MAX_DISTANCE_M,
    require_both_spaces: bool = False,
) -> Carrier | None:
    """Nearest player within looser first-touch range (pass detection only)."""
    return find_ball_carrier(
        detections,
        max_distance_px=max_distance_px,
        transformer=transformer,
        max_distance_m=max_distance_m,
        require_both_spaces=require_both_spaces,
    )


def find_active_carrier(
    detections: sv.Detections,
    *,
    transformer=None,
    control_max_distance_px: float = CONTROL_MAX_DISTANCE_PX,
    control_max_distance_m: float = CONTROL_MAX_DISTANCE_M,
    reception_max_distance_px: float = RECEPTION_MAX_DISTANCE_PX,
    reception_max_distance_m: float = RECEPTION_MAX_DISTANCE_M,
    require_both_spaces: bool = False,
) -> tuple[Carrier | None, str | None]:
    """Control carrier if any, else reception; matches pass-detection possession logic."""
    control = find_control_carrier(
        detections,
        transformer=transformer,
        max_distance_px=control_max_distance_px,
        max_distance_m=control_max_distance_m,
        require_both_spaces=require_both_spaces,
    )
    if control is not None:
        return control, "control"
    reception = find_reception_carrier(
        detections,
        transformer=transformer,
        max_distance_px=reception_max_distance_px,
        max_distance_m=reception_max_distance_m,
        require_both_spaces=require_both_spaces,
    )
    if reception is not None:
        return reception, "reception"
    return None, None


@dataclass(frozen=True)
class TouchValidationConfig:
    aerial_dy_threshold_px: float = AERIAL_DY_THRESHOLD_PX
    # Transit fly-by: fast ball through the control radius without settling at feet.
    transit_min_speed_m_s: float = 9.0
    transit_min_feet_px: float = 30.0
    transit_min_speed_px_per_frame: float = 10.0
    max_plausible_transit_speed_m_s: float = 35.0
    ball_speed_lookback_frames: int = 10
    ball_speed_min_lookback_frames: int = 3
    # Long in-flight path from a known release point: slow average inbound speed
    # means the ball is dropping through a zone, not possession at the feet.
    transit_min_release_travel_px: float = 450.0
    # Align with gravity-flyby window so medium-length passes (≈15–50f) still
    # veto slow drop-throughs; 50f was too late for typical interceptions.
    transit_min_release_gap_frames: int = 15
    transit_release_flyby_max_speed_px_per_frame: float = 8.0
    # Ball path redirect at a touch (one-touch kick / intercept) vs straight fly-by.
    redirect_lookback_frames: int = 5
    redirect_lookahead_frames: int = 5
    redirect_min_angle_deg: float = 28.0
    redirect_min_speed_ratio: float = 1.35
    redirect_min_segment_px: float = 10.0
    # In-flight opponent touch during a known release: no path redirect ⇒ gravity fly-by.
    # Keep this small — the first ~0.5s of a pass is when skims past bystanders happen.
    gravity_flyby_min_release_gap_frames: int = 3


def nearest_player_tid(dets: sv.Detections, ball: np.ndarray) -> int | None:
    found = nearest_player_feet(dets, ball)
    return found[0] if found is not None else None


def nearest_player_feet(
    dets: sv.Detections, ball: np.ndarray,
) -> tuple[int, float] | None:
    """Return ``(tracker_id, feet distance px)`` for the closest outfield player."""
    pmask = player_mask(dets)
    if not pmask.any() or dets.tracker_id is None:
        return None
    feet = feet_xy(dets)[pmask]
    tids = dets.tracker_id[pmask]
    dist = np.hypot(feet[:, 0] - ball[0], feet[:, 1] - ball[1])
    local = int(np.argmin(dist))
    tid = int(tids[local])
    if tid < 0:
        return None
    return tid, float(dist[local])


def is_aerial_touch(
    dets: sv.Detections,
    carrier: Carrier,
    *,
    threshold_px: float,
) -> bool:
    ball = ball_xy(dets)
    if ball is None:
        return False
    feet = feet_xy(dets)[carrier.index]
    return abs(float(ball[1] - feet[1])) > threshold_px


def is_aerial_flyby_below_feet(
    dets: sv.Detections,
    carrier: Carrier,
    *,
    threshold_px: float,
) -> bool:
    """Ball visibly below the feet in image space (aerial fly-by, not chest reception)."""
    ball = ball_xy(dets)
    if ball is None:
        return False
    feet = feet_xy(dets)[carrier.index]
    return float(ball[1] - feet[1]) > threshold_px


def reception_aerial_veto_threshold(config: TouchValidationConfig) -> float:
    """Looser than control: chest receptions are above the feet (negative dy)."""
    return config.aerial_dy_threshold_px * 2.0


def ball_instant_speed_m_s(
    prev_ball: np.ndarray,
    ball: np.ndarray,
    *,
    fps: float,
    transformer,
    frame_gap: int = 1,
    prev_transformer=None,
) -> float | None:
    """Ball speed in m/s from two image positions (optionally separated by >1 frame)."""
    if fps <= 0 or transformer is None or frame_gap < 1:
        return None

    prev_t = prev_transformer if prev_transformer is not None else transformer
    pitch_prev = image_to_pitch_m(np.array([prev_ball], dtype=np.float32), prev_t)
    pitch_curr = image_to_pitch_m(np.array([ball], dtype=np.float32), transformer)
    if pitch_prev is not None and pitch_curr is not None:
        dist_m = float(np.linalg.norm(pitch_curr[0] - pitch_prev[0]))
    else:
        pts = np.stack([prev_ball, ball], axis=0).astype(np.float32)
        pitch = image_to_pitch_m(pts, transformer)
        if pitch is None:
            return None
        dist_m = float(np.linalg.norm(pitch[1] - pitch[0]))
    return dist_m * fps / frame_gap


def inbound_speed_px_per_frame(
    ref_ball: np.ndarray,
    ball: np.ndarray,
    *,
    frame_gap: int,
) -> tuple[float, float]:
    """Return ``(travel_px, speed_px_per_frame)`` for an inbound ball path."""
    gap = max(1, frame_gap)
    travel_px = float(np.linalg.norm(ball - ref_ball))
    return travel_px, travel_px / gap


def is_fast_inbound_transit(
    ref_ball: np.ndarray,
    ball: np.ndarray,
    *,
    frame_gap: int,
    config: TouchValidationConfig,
    fps: float = 25.0,
    transformer=None,
    prev_transformer=None,
) -> bool:
    """Primary speed gate: fast inbound ball transit, independent of aerial dy or feet px."""
    gap = max(1, frame_gap)
    travel_px, speed_px_f = inbound_speed_px_per_frame(ref_ball, ball, frame_gap=gap)
    if speed_px_f >= config.transit_min_speed_px_per_frame:
        return True
    speed_m_s = ball_instant_speed_m_s(
        ref_ball,
        ball,
        fps=fps,
        transformer=transformer,
        frame_gap=gap,
        prev_transformer=prev_transformer,
    )
    return (
        speed_m_s is not None
        and speed_m_s <= config.max_plausible_transit_speed_m_s
        and speed_m_s >= config.transit_min_speed_m_s
    )


def is_release_inbound_flyby(
    release_ball: np.ndarray,
    ball: np.ndarray,
    *,
    release_gap_frames: int,
    config: TouchValidationConfig,
) -> bool:
    """Ball traveled far from a pass release but arrived slowly — dropping through a zone."""
    if release_gap_frames < config.transit_min_release_gap_frames:
        return False
    travel_px, speed_px_f = inbound_speed_px_per_frame(
        release_ball, ball, frame_gap=release_gap_frames
    )
    return (
        travel_px >= config.transit_min_release_travel_px
        and speed_px_f < config.transit_release_flyby_max_speed_px_per_frame
    )


def is_transit_flyby_touch(
    dets: sv.Detections,
    carrier: Carrier,
    *,
    prev_ball: np.ndarray | None,
    config: TouchValidationConfig,
    fps: float = 25.0,
    transformer=None,
    frame_gap: int = 1,
    prev_transformer=None,
    speed_prev_ball: np.ndarray | None = None,
    speed_frame_gap: int | None = None,
    speed_prev_transformer=None,
    release_ball: np.ndarray | None = None,
    release_gap_frames: int | None = None,
) -> bool:
    """Fast ball through a player's zone without settling at feet (not real possession)."""
    ball = ball_xy(dets)
    if ball is None:
        return False

    ref_ball = speed_prev_ball if speed_prev_ball is not None else prev_ball
    if ref_ball is not None:
        gap = max(1, speed_frame_gap if speed_prev_ball is not None else frame_gap)
        ref_t = speed_prev_transformer if speed_prev_ball is not None else prev_transformer
        if is_fast_inbound_transit(
            ref_ball,
            ball,
            frame_gap=gap,
            config=config,
            fps=fps,
            transformer=transformer,
            prev_transformer=ref_t,
        ):
            return True

    if (
        release_ball is not None
        and release_gap_frames is not None
        and is_release_inbound_flyby(
            release_ball,
            ball,
            release_gap_frames=release_gap_frames,
            config=config,
        )
    ):
        return True

    if ref_ball is None:
        return False

    feet = feet_xy(dets)[carrier.index]
    feet_dist_px = float(np.hypot(ball[0] - feet[0], ball[1] - feet[1]))
    zone_px = feet_dist_px
    if transformer is None and carrier.distance <= CONTROL_MAX_DISTANCE_PX:
        zone_px = max(feet_dist_px, float(carrier.distance))
    if zone_px < config.transit_min_feet_px:
        return False

    gap = max(1, speed_frame_gap if speed_prev_ball is not None else frame_gap)
    _, speed_px_f = inbound_speed_px_per_frame(ref_ball, ball, frame_gap=gap)
    return speed_px_f >= config.transit_min_speed_px_per_frame


_MAX_BALL_PATH_JUMP_PX_PER_FRAME = 180.0


def _ball_path_samples(
    frames_by_idx: dict[int, sv.Detections],
    touch_frame: int,
    *,
    lookback: int,
    lookahead: int,
    max_jump_px_per_frame: float = _MAX_BALL_PATH_JUMP_PX_PER_FRAME,
) -> list[tuple[int, np.ndarray]]:
    """Ball positions near a touch, dropping single-frame detection teleports."""
    samples: list[tuple[int, np.ndarray]] = []
    prev_ball: np.ndarray | None = None
    prev_frame: int | None = None
    for frame_idx in range(touch_frame - lookback, touch_frame + lookahead + 1):
        dets = frames_by_idx.get(frame_idx)
        if dets is None:
            continue
        ball = ball_xy(dets)
        if ball is None:
            continue
        point = np.asarray(ball, dtype=np.float64)
        if prev_ball is not None and prev_frame is not None:
            gap = max(1, frame_idx - prev_frame)
            jump = float(np.hypot(point[0] - prev_ball[0], point[1] - prev_ball[1]))
            if jump > max_jump_px_per_frame * gap:
                continue
        samples.append((frame_idx, point))
        prev_ball = point
        prev_frame = frame_idx
    return samples


def _ball_touch_path_metrics(
    frames_by_idx: dict[int, sv.Detections],
    touch_frame: int,
    *,
    lookback: int = 5,
    lookahead: int = 5,
    min_segment_px: float = 10.0,
    filter_teleports: bool = False,
) -> tuple[float, float] | None:
    """Inbound angle (deg) and outbound/inbound speed ratio at ``touch_frame``."""

    if filter_teleports:
        samples = _ball_path_samples(
            frames_by_idx, touch_frame, lookback=lookback, lookahead=lookahead
        )
    else:
        samples: list[tuple[int, np.ndarray]] = []
        for frame_idx in range(touch_frame - lookback, touch_frame + lookahead + 1):
            dets = frames_by_idx.get(frame_idx)
            if dets is None:
                continue
            ball = ball_xy(dets)
            if ball is not None:
                samples.append((frame_idx, np.asarray(ball, dtype=np.float64)))
    if len(samples) < 4:
        return None

    before = [(f, p) for f, p in samples if f <= touch_frame]
    after = [(f, p) for f, p in samples if f >= touch_frame]
    if len(before) < 2 or len(after) < 2:
        return None

    pivot = before[-1][1]
    for frame_idx, point in samples:
        if frame_idx == touch_frame:
            pivot = point
            break

    f_in0, p_in0 = before[-2]
    v_in = pivot - (p_in0 if f_in0 < touch_frame else before[-1][1])
    in_len = float(np.linalg.norm(v_in))
    if in_len < min_segment_px:
        return None

    after_touch = [(f, p) for f, p in after if f >= touch_frame]
    if len(after_touch) < 2:
        return None
    _, p_out1 = after_touch[1]
    v_out = p_out1 - pivot
    out_len = float(np.linalg.norm(v_out))
    if out_len < min_segment_px:
        return None

    u_in, u_out = unit(v_in), unit(v_out)
    if u_in is None or u_out is None:
        return None
    cos_angle = float(np.clip(np.dot(u_in, u_out), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cos_angle)))
    speed_ratio = out_len / max(in_len, 1e-6)
    return angle_deg, speed_ratio


def _redirect_signature(
    angle_deg: float,
    speed_ratio: float,
    *,
    min_angle_deg: float,
    min_speed_ratio: float,
) -> bool:
    """True when path change looks like a kick, not a speed blip on a fly-by.

    Speed alone is not enough: a ball skimming a teammate mid-pass can briefly
    change measured speed without a real deflection. Require an angular turn.
    """
    return min_angle_deg <= angle_deg <= 135.0 or (
        speed_ratio >= min_speed_ratio and angle_deg >= min_angle_deg
    )


def ball_redirected_at_touch(
    frames_by_idx: dict[int, sv.Detections],
    touch_frame: int,
    *,
    lookback: int = 5,
    lookahead: int = 5,
    min_angle_deg: float = 28.0,
    min_speed_ratio: float = 1.35,
    min_segment_px: float = 10.0,
) -> bool:
    """True when inbound/outbound ball vectors diverge at ``touch_frame`` (kick, not fly-by)."""
    metrics = _ball_touch_path_metrics(
        frames_by_idx,
        touch_frame,
        lookback=lookback,
        lookahead=lookahead,
        min_segment_px=min_segment_px,
    )
    if metrics is None:
        return False
    angle_deg, speed_ratio = metrics
    return _redirect_signature(
        angle_deg,
        speed_ratio,
        min_angle_deg=min_angle_deg,
        min_speed_ratio=min_speed_ratio,
    )


def is_gravity_arc_flyby_at_touch(
    frames_by_idx: dict[int, sv.Detections],
    touch_frame: int,
    *,
    lookback: int = 5,
    lookahead: int = 5,
    max_angle_deg: float = 28.0,
    max_speed_ratio: float = 1.35,
    min_segment_px: float = 10.0,
) -> bool:
    """True when the ball continues on the same arc through ``touch_frame`` (gravity only)."""
    metrics = _ball_touch_path_metrics(
        frames_by_idx,
        touch_frame,
        lookback=lookback,
        lookahead=lookahead,
        min_segment_px=min_segment_px,
    )
    if metrics is None:
        return False
    angle_deg, speed_ratio = metrics
    return angle_deg < max_angle_deg and speed_ratio < max_speed_ratio


def redirect_overrides_transit_flyby(
    frames_by_idx: dict[int, sv.Detections],
    touch_frame: int,
    *,
    config: TouchValidationConfig,
    release_gap_frames: int | None = None,
) -> bool:
    """True when a touch redirected the ball path enough to count as possession."""
    metrics = _ball_touch_path_metrics(
        frames_by_idx,
        touch_frame,
        lookback=config.redirect_lookback_frames,
        lookahead=config.redirect_lookahead_frames,
        min_segment_px=config.redirect_min_segment_px,
        filter_teleports=(
            release_gap_frames is not None
            and release_gap_frames >= config.gravity_flyby_min_release_gap_frames
        ),
    )
    if metrics is None:
        return False
    angle_deg, speed_ratio = metrics
    min_angle = config.redirect_min_angle_deg
    min_ratio = config.redirect_min_speed_ratio
    if (
        release_gap_frames is not None
        and release_gap_frames >= config.gravity_flyby_min_release_gap_frames
        and speed_ratio > 4.0
    ):
        # Teleport / sparse-detection speed spike during a long flight.
        return False
    return _redirect_signature(
        angle_deg,
        speed_ratio,
        min_angle_deg=min_angle,
        min_speed_ratio=min_ratio,
    )


def is_valid_possession_touch(
    dets: sv.Detections,
    carrier: Carrier,
    *,
    touch_kind: str,
    config: TouchValidationConfig,
    prev_ball: np.ndarray | None = None,
    fps: float = 25.0,
    transformer=None,
    frame_gap: int = 1,
    prev_transformer=None,
    speed_prev_ball: np.ndarray | None = None,
    speed_frame_gap: int | None = None,
    speed_prev_transformer=None,
    release_ball: np.ndarray | None = None,
    release_gap_frames: int | None = None,
    frames_by_idx: dict[int, sv.Detections] | None = None,
    frame_idx: int | None = None,
) -> bool:
    """Reject fly-bys and nearest-player mismatches.

    Speed-based transit fly-by is evaluated first and is independent of aerial
    dy / chest-height checks. Aerial vetoes only apply to clearly off-ground
    contacts in image space.
    """
    transit_kwargs = dict(
        prev_ball=prev_ball,
        config=config,
        fps=fps,
        transformer=transformer,
        frame_gap=frame_gap,
        prev_transformer=prev_transformer,
        speed_prev_ball=speed_prev_ball,
        speed_frame_gap=speed_frame_gap,
        speed_prev_transformer=speed_prev_transformer,
        release_ball=release_ball,
        release_gap_frames=release_gap_frames,
    )
    ball = carrier.ball
    release_inbound_flyby = (
        release_ball is not None
        and release_gap_frames is not None
        and is_release_inbound_flyby(
            release_ball,
            ball,
            release_gap_frames=release_gap_frames,
            config=config,
        )
    )
    if is_transit_flyby_touch(dets, carrier, **transit_kwargs):
        if release_inbound_flyby:
            return False
        if (
            frames_by_idx is not None
            and frame_idx is not None
            and redirect_overrides_transit_flyby(
                frames_by_idx,
                frame_idx,
                config=config,
                release_gap_frames=release_gap_frames,
            )
        ):
            if (
                release_gap_frames is not None
                and release_gap_frames >= config.gravity_flyby_min_release_gap_frames
                and is_aerial_touch(
                    dets, carrier, threshold_px=config.aerial_dy_threshold_px
                )
            ):
                return False
        else:
            return False
    if (
        release_ball is not None
        and release_gap_frames is not None
        and release_gap_frames >= config.gravity_flyby_min_release_gap_frames
        and frames_by_idx is not None
        and frame_idx is not None
        and is_gravity_arc_flyby_at_touch(
            frames_by_idx,
            frame_idx,
            lookback=config.redirect_lookback_frames,
            lookahead=config.redirect_lookahead_frames,
            max_angle_deg=config.redirect_min_angle_deg,
            max_speed_ratio=config.redirect_min_speed_ratio,
            min_segment_px=config.redirect_min_segment_px,
        )
    ):
        return False
    # Known pass still in flight, no redirect, ball still moving: skim/fly-by,
    # not settled control (catches brief proximity pauses that break gravity-arc).
    if (
        release_ball is not None
        and release_gap_frames is not None
        and release_gap_frames >= config.gravity_flyby_min_release_gap_frames
        and frames_by_idx is not None
        and frame_idx is not None
        and not ball_redirected_at_touch(
            frames_by_idx,
            frame_idx,
            lookback=config.redirect_lookback_frames,
            lookahead=config.redirect_lookahead_frames,
            min_angle_deg=config.redirect_min_angle_deg,
            min_speed_ratio=config.redirect_min_speed_ratio,
            min_segment_px=config.redirect_min_segment_px,
        )
    ):
        prev = frames_by_idx.get(frame_idx - 1)
        prev_ball = ball_xy(prev) if prev is not None else None
        if prev_ball is not None:
            step_px = float(
                np.hypot(ball[0] - prev_ball[0], ball[1] - prev_ball[1])
            )
            # Real traps are near-stationary in image space; skims keep moving.
            if step_px >= 2.5:
                return False
    if touch_kind == "control" and is_aerial_touch(
        dets, carrier, threshold_px=config.aerial_dy_threshold_px
    ):
        return False
    if touch_kind == "reception" and is_aerial_flyby_below_feet(
        dets,
        carrier,
        threshold_px=reception_aerial_veto_threshold(config),
    ):
        return False
    ball = carrier.ball
    tid = int(dets.tracker_id[carrier.index]) if dets.tracker_id is not None else -1
    nearest_tid = nearest_player_tid(dets, ball)
    return nearest_tid is not None and nearest_tid == tid


def ball_departed_for_one_touch(
    touch_ball: np.ndarray,
    in_flight_ball: np.ndarray | None,
    *,
    touch_frame: int,
    in_flight_frame: int,
    depart_min_px: float,
    frame_balls: list[tuple[int, np.ndarray]] | None = None,
) -> bool:
    """True when the ball left the one-touch passer zone (not a stationary fly-by).

    When the ball is missing on every in-flight frame we allow the release anchor.
    ``frame_balls`` lists ``(frame_idx, ball_xy)`` samples after the touch so slow
    roll-outs can satisfy the departure threshold a few frames later.
    """
    max_travel = 0.0
    samples = list(frame_balls or [])
    if in_flight_ball is not None:
        samples.append((in_flight_frame, in_flight_ball))
    if not samples:
        return True
    for sample_frame, ball in samples:
        travel_px, _ = inbound_speed_px_per_frame(
            touch_ball,
            ball,
            frame_gap=max(1, sample_frame - touch_frame),
        )
        max_travel = max(max_travel, travel_px)
    return max_travel >= depart_min_px
