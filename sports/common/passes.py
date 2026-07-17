"""Infer pass events from per-frame ball-carrier handoffs.

Simple rule set (per team, frame by frame)
------------------------------------------
1. **Valid touch** — ball is nearest this player's feet; reject aerial *control*
   and reception fly-bys below the feet (chest-height receptions are OK).
2. **Passer** — player who *released* the ball:
   - outfield: ``min_control_frames`` consecutive control frames, OR
   - goalkeeper: one control/reception frame, OR
   - any role: last valid touch within ``pre_flight_release_window`` when the
     ball goes in-flight (covers punts and one-touch releases), OR
   - one-touch: ``min_one_touch_reception_frames`` valid reception frames at
     the feet, then the ball departs within ``one_touch_release_window_frames``
     (give-and-go without settling into control).
3. **Receiver** — teammate who gets the ball after a gap:
   - ``min_arrival_frames`` consecutive valid touches (filters deflections).
   - if gap passer→receiver < ``adjacent_pass_max_gap_frames``, receiver must
     also show brief control (filters fly-by false receptions on quick plays).
   - longer gaps still need ``min_arrival_control_frames`` with the ball at
     the feet (filters early credit while the ball is still travelling in).
4. **Emit pass** — same team, frame gap in range, min ball travel, no opponent
   *touch* (control or reception) between passer and receiver, dedupe nearby
   duplicates. Post-filter drops overlapping cross-team passes (keep the
   earlier release) so only one pass is active at a time in the overlay.
5. **Emit turnover** — team A releases the ball; an opponent touches it before
   any teammate of A arrives (snapshot release on first opponent touch). When
   that opponent later completes a pass, attribute the turnover to their
   first touch frame. Teammate arrivals are ignored if an opponent controlled
   the ball in between.

Distance gates: tight *control* at the feet; looser *reception* for first
contact / one-touch. Anchors survive in-flight stretches (ball visible but no
player in range) and brief ball-detection dropouts (``missing_ball_tolerance``).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import supervision as sv

from sports.common.pass_pitch import image_to_pitch_m
from sports.common.possession import (
    AERIAL_DY_THRESHOLD_PX,
    CONTROL_MAX_DISTANCE_M,
    CONTROL_MAX_DISTANCE_PX,
    RECEPTION_MAX_DISTANCE_M,
    RECEPTION_MAX_DISTANCE_PX,
    Carrier,
    TouchValidationConfig,
    ball_departed_for_one_touch,
    ball_redirected_at_touch,
    ball_xy,
    bbox_center_xy,
    find_active_carrier,
    find_control_carrier,
    find_reception_carrier,
    is_aerial_flyby_below_feet,
    is_aerial_touch,
    is_release_inbound_flyby,
    is_valid_possession_touch,
    nearest_player_tid,
    reception_aerial_veto_threshold,
    redirect_overrides_transit_flyby,
)
from sports.configs.soccer import GOALKEEPER_CLASS_ID as ROLE_GOALKEEPER

DetectionIterator = Iterator[tuple[int, sv.Detections]]


@dataclass(frozen=True)
class PassDetectionConfig:
    """Heuristic gates for carrier-to-carrier pass inference.

    Defaults assume ~25 fps. Frame counts can be scaled via
    :meth:`for_frame_rate` when working at a different rate.
    """

    min_carrier_gap_frames: int = 1
    max_pass_gap_frames: int = 110  # ~4.4 s at 25 fps; long aerials + ball dropouts
    min_ball_travel_m: float = 1.0
    min_ball_travel_px: float = 25.0
    control_max_distance_m: float = CONTROL_MAX_DISTANCE_M
    control_max_distance_px: float = CONTROL_MAX_DISTANCE_PX
    reception_max_distance_m: float = RECEPTION_MAX_DISTANCE_M
    reception_max_distance_px: float = RECEPTION_MAX_DISTANCE_PX
    missing_ball_tolerance: int = 10  # ~0.4 s at 25 fps; bridge intermittent ball loss
    dedupe_window_frames: int = 12
    min_arrival_frames: int = 3
    min_reception_arrival_frames: int = 2
    min_arrival_control_frames: int = 1
    min_control_frames: int = 2
    min_turnover_passer_control_frames: int = 2
    min_gk_control_frames: int = 1
    pre_flight_release_window: int = 10  # ~0.4 s before ball leaves range
    min_one_touch_reception_frames: int = 2
    one_touch_release_window_frames: int = 12
    one_touch_depart_min_px: float = 35.0
    adjacent_pass_max_gap_frames: int = 15  # quick plays need control at receiver
    # Gravity-arc fly-by veto this many frames after a known release (keep short —
    # bystander skims happen early; do not reuse adjacent_pass_max_gap_frames).
    gravity_flyby_min_release_gap_frames: int = 3
    aerial_dy_threshold_px: float = AERIAL_DY_THRESHOLD_PX
    max_plausible_travel_m: float = 40.0
    min_long_gap_opponent_control_streak: int = 2
    transit_min_speed_m_s: float = 9.0
    transit_min_feet_px: float = 30.0
    transit_min_speed_px_per_frame: float = 10.0
    max_plausible_transit_speed_m_s: float = 35.0
    ball_speed_lookback_frames: int = 40
    ball_speed_min_lookback_frames: int = 3
    # Long-gap opponent control when speed homography is missing (pixel fallback only).
    long_gap_opponent_control_max_px: float = 32.0
    turnover_recovery_window_frames: int = 15
    # Single-frame ball detection outliers (teleports): max plausible image-space
    # travel per frame before we distrust the ball position for possession.
    max_ball_teleport_px_per_frame: float = 180.0
    # A real turnover follows a release that travels; a 1-3 frame possession flip is
    # a tackle/contested ball, not a turnover.
    min_turnover_gap_frames: int = 8
    redirect_instant_pass_max_gap_frames: int = 42

    def for_frame_rate(self, fps: float, *, base_fps: float = 25.0) -> PassDetectionConfig:
        """Scale time-based frame counts for a different video frame rate."""
        if fps <= 0:
            return self
        scale = fps / base_fps
        return PassDetectionConfig(
            min_carrier_gap_frames=self.min_carrier_gap_frames,
            max_pass_gap_frames=max(1, round(self.max_pass_gap_frames * scale)),
            min_ball_travel_m=self.min_ball_travel_m,
            min_ball_travel_px=self.min_ball_travel_px,
            control_max_distance_m=self.control_max_distance_m,
            control_max_distance_px=self.control_max_distance_px,
            reception_max_distance_m=self.reception_max_distance_m,
            reception_max_distance_px=self.reception_max_distance_px,
            missing_ball_tolerance=max(1, round(self.missing_ball_tolerance * scale)),
            dedupe_window_frames=max(1, round(self.dedupe_window_frames * scale)),
            min_arrival_frames=max(1, round(self.min_arrival_frames * scale)),
            min_reception_arrival_frames=max(
                1, round(self.min_reception_arrival_frames * scale)
            ),
            min_arrival_control_frames=max(
                1, round(self.min_arrival_control_frames * scale)
            ),
            min_control_frames=max(1, round(self.min_control_frames * scale)),
            min_turnover_passer_control_frames=max(
                1, round(self.min_turnover_passer_control_frames * scale)
            ),
            min_gk_control_frames=max(1, round(self.min_gk_control_frames * scale)),
            pre_flight_release_window=max(
                1, round(self.pre_flight_release_window * scale)
            ),
            min_one_touch_reception_frames=max(
                1, round(self.min_one_touch_reception_frames * scale)
            ),
            one_touch_release_window_frames=max(
                1, round(self.one_touch_release_window_frames * scale)
            ),
            one_touch_depart_min_px=self.one_touch_depart_min_px,
            adjacent_pass_max_gap_frames=max(
                1, round(self.adjacent_pass_max_gap_frames * scale)
            ),
            gravity_flyby_min_release_gap_frames=max(
                1, round(self.gravity_flyby_min_release_gap_frames * scale)
            ),
            aerial_dy_threshold_px=self.aerial_dy_threshold_px,
            max_plausible_travel_m=self.max_plausible_travel_m,
            min_long_gap_opponent_control_streak=self.min_long_gap_opponent_control_streak,
            transit_min_speed_m_s=self.transit_min_speed_m_s,
            transit_min_feet_px=self.transit_min_feet_px,
            transit_min_speed_px_per_frame=self.transit_min_speed_px_per_frame,
            max_plausible_transit_speed_m_s=self.max_plausible_transit_speed_m_s,
            ball_speed_lookback_frames=max(
                1, round(self.ball_speed_lookback_frames * scale)
            ),
            ball_speed_min_lookback_frames=max(
                1, round(self.ball_speed_min_lookback_frames * scale)
            ),
            long_gap_opponent_control_max_px=self.long_gap_opponent_control_max_px,
            turnover_recovery_window_frames=max(
                1, round(self.turnover_recovery_window_frames * scale)
            ),
            max_ball_teleport_px_per_frame=self.max_ball_teleport_px_per_frame,
            min_turnover_gap_frames=max(
                1, round(self.min_turnover_gap_frames * scale)
            ),
            redirect_instant_pass_max_gap_frames=max(
                1, round(self.redirect_instant_pass_max_gap_frames * scale)
            ),
        )

    def touch_validation_config(self) -> TouchValidationConfig:
        return TouchValidationConfig(
            aerial_dy_threshold_px=self.aerial_dy_threshold_px,
            transit_min_speed_m_s=self.transit_min_speed_m_s,
            transit_min_feet_px=self.transit_min_feet_px,
            transit_min_speed_px_per_frame=self.transit_min_speed_px_per_frame,
            max_plausible_transit_speed_m_s=self.max_plausible_transit_speed_m_s,
            ball_speed_lookback_frames=self.ball_speed_lookback_frames,
            ball_speed_min_lookback_frames=self.ball_speed_min_lookback_frames,
            gravity_flyby_min_release_gap_frames=self.gravity_flyby_min_release_gap_frames,
        )

@dataclass(frozen=True)
class InferredPass:
    """One directed pass A -> B at the release frame."""

    frame_idx: int
    passer_tid: int
    receiver_tid: int
    team: int
    gap_frames: int
    pass_length_m: float | None
    touch_kind: str = "control"


@dataclass(frozen=True)
class InferredTurnover:
    """Possession lost: passer released the ball, opponent received it in-flight."""

    release_frame: int
    interception_frame: int
    passer_tid: int
    passer_team: int
    interceptor_tid: int
    interceptor_team: int
    gap_frames: int


@dataclass(frozen=True)
class PossessionScanResult:
    """Completed passes and interceptions from one scan of the clip."""

    passes: tuple[InferredPass, ...]
    turnovers: tuple[InferredTurnover, ...]


def _ball_travel(
    ball_from: np.ndarray,
    ball_to: np.ndarray,
    *,
    metric: bool,
    transformer_from,
    transformer_to,
) -> float | None:
    """Distance the ball moved between two carrier frames (m or px)."""
    travel_px = float(np.linalg.norm(ball_to - ball_from))
    if metric and transformer_from is not None and transformer_to is not None:
        p0 = image_to_pitch_m(np.array([ball_from], dtype=np.float32), transformer_from)
        p1 = image_to_pitch_m(np.array([ball_to], dtype=np.float32), transformer_to)
        if p0 is not None and p1 is not None:
            travel_m = float(np.linalg.norm(p1[0] - p0[0]))
            return travel_m
        return travel_px
    return travel_px


def _effective_ball_travel(
    ball_from: np.ndarray,
    ball_to: np.ndarray,
    *,
    metric: bool,
    transformer_from,
    transformer_to,
    config: PassDetectionConfig,
) -> float | None:
    """Ball travel with pixel fallback when pitch projection is unreliable."""
    travel = _ball_travel(
        ball_from,
        ball_to,
        metric=metric,
        transformer_from=transformer_from,
        transformer_to=transformer_to,
    )
    if travel is None:
        return None
    if metric and travel > config.max_plausible_travel_m:
        travel_px = float(np.linalg.norm(ball_to - ball_from))
        return travel_px
    return travel


def _ball_speed_reference(
    frames_by_idx: dict[int, sv.Detections],
    frame_idx: int,
    *,
    max_lookback: int,
    min_lookback: int = 3,
    transformers: dict[int, object] | None = None,
    metric: bool = False,
) -> tuple[np.ndarray | None, int, object | None]:
    """Best prior ball position for inbound speed (max travel, not nearest gap)."""
    ball = ball_xy(frames_by_idx.get(frame_idx))
    if ball is None:
        return None, 1, None
    ball = np.asarray(ball, dtype=np.float64)
    best: tuple[np.ndarray, int, object | None] | None = None
    best_travel = 0.0
    for gap in range(max(min_lookback, 2), max_lookback + 1):
        older = frames_by_idx.get(frame_idx - gap)
        if older is None:
            continue
        older_ball = ball_xy(older)
        if older_ball is None:
            continue
        older_ball = np.asarray(older_ball, dtype=np.float64)
        travel = float(np.linalg.norm(ball - older_ball))
        if travel <= best_travel:
            continue
        prev_t = transformers.get(frame_idx - gap) if metric and transformers else None
        best = (older_ball, gap, prev_t)
        best_travel = travel
    if best is None:
        return None, 1, None
    return best


def _touch_validation_kwargs(
    frames_by_idx: dict[int, sv.Detections],
    frame_idx: int,
    *,
    transformers: dict[int, object],
    metric: bool,
    fps: float,
    max_lookback: int = 10,
    min_lookback: int = 3,
    release_ball: np.ndarray | None = None,
    release_frame: int | None = None,
) -> dict:
    transformer = transformers.get(frame_idx) if metric else None
    prev = frames_by_idx.get(frame_idx - 1)
    prev_ball = ball_xy(prev) if prev is not None else None
    prev_transformer = (
        transformers.get(frame_idx - 1) if metric and prev_ball is not None else None
    )
    speed_prev_ball = (
        np.asarray(prev_ball, dtype=np.float64) if prev_ball is not None else None
    )
    speed_frame_gap = 1
    speed_prev_transformer = prev_transformer
    if speed_prev_ball is None:
        speed_prev_ball, speed_frame_gap, speed_prev_transformer = _ball_speed_reference(
            frames_by_idx,
            frame_idx,
            max_lookback=max_lookback,
            min_lookback=min_lookback,
            transformers=transformers,
            metric=metric,
        )
    release_gap_frames = None
    if release_ball is not None and release_frame is not None:
        release_gap_frames = max(1, frame_idx - release_frame)
    return {
        "prev_ball": np.asarray(prev_ball, dtype=np.float64) if prev_ball is not None else None,
        "speed_prev_ball": speed_prev_ball,
        "speed_frame_gap": speed_frame_gap,
        "speed_prev_transformer": speed_prev_transformer,
        "fps": fps,
        "transformer": transformer,
        "release_ball": (
            np.asarray(release_ball, dtype=np.float64) if release_ball is not None else None
        ),
        "release_gap_frames": release_gap_frames,
        "frames_by_idx": frames_by_idx,
        "frame_idx": frame_idx,
    }


def _ball_redirected(
    frames_by_idx: dict[int, sv.Detections],
    frame_idx: int,
    *,
    config: PassDetectionConfig,
) -> bool:
    touch_cfg = config.touch_validation_config()
    return ball_redirected_at_touch(
        frames_by_idx,
        frame_idx,
        lookback=touch_cfg.redirect_lookback_frames,
        lookahead=touch_cfg.redirect_lookahead_frames,
        min_angle_deg=touch_cfg.redirect_min_angle_deg,
        min_speed_ratio=touch_cfg.redirect_min_speed_ratio,
        min_segment_px=touch_cfg.redirect_min_segment_px,
    )


def _touch_valid_or_redirect(
    dets: sv.Detections,
    carrier: Carrier,
    *,
    touch_kind: str,
    frames_by_idx: dict[int, sv.Detections],
    frame_idx: int,
    config: PassDetectionConfig,
    transformers: dict[int, object],
    metric: bool,
    fps: float,
    release_ball: np.ndarray | None = None,
    release_frame: int | None = None,
) -> bool:
    """Valid possession touch, or a redirect kick that transit/release fly-by would veto."""
    touch_cfg = config.touch_validation_config()
    kwargs = _touch_validation_kwargs(
        frames_by_idx,
        frame_idx,
        transformers=transformers,
        metric=metric,
        fps=fps,
        max_lookback=config.ball_speed_lookback_frames,
        min_lookback=config.ball_speed_min_lookback_frames,
        release_ball=release_ball,
        release_frame=release_frame,
    )
    if is_valid_possession_touch(
        dets,
        carrier,
        touch_kind=touch_kind,
        config=touch_cfg,
        **kwargs,
    ):
        return True
    release_gap_frames = kwargs.get("release_gap_frames")
    if (
        release_ball is not None
        and release_gap_frames is not None
        and is_release_inbound_flyby(
            release_ball,
            carrier.ball,
            release_gap_frames=release_gap_frames,
            config=touch_cfg,
        )
    ):
        return False
    if (
        release_ball is not None
        and release_gap_frames is not None
        and not redirect_overrides_transit_flyby(
            frames_by_idx,
            frame_idx,
            config=touch_cfg,
            release_gap_frames=release_gap_frames,
        )
    ):
        return False
    elif not _ball_redirected(frames_by_idx, frame_idx, config=config):
        return False
    if touch_kind == "control" and is_aerial_touch(
        dets, carrier, threshold_px=touch_cfg.aerial_dy_threshold_px
    ):
        return False
    if touch_kind == "reception" and is_aerial_flyby_below_feet(
        dets,
        carrier,
        threshold_px=reception_aerial_veto_threshold(touch_cfg),
    ):
        return False
    ball = carrier.ball
    tid = int(dets.tracker_id[carrier.index]) if dets.tracker_id is not None else -1

    nearest_tid = nearest_player_tid(dets, ball)
    return nearest_tid is not None and nearest_tid == tid


def _first_opponent_touch_in_window(
    frames_by_idx: dict[int, sv.Detections],
    *,
    start_frame: int,
    end_frame: int,
    passer_team: int,
    config: PassDetectionConfig,
    transformers: dict[int, object],
    metric: bool,
    require_control: bool = False,
    player_tid: int | None = None,
    fps: float = 25.0,
    release_frame: int | None = None,
    release_ball: np.ndarray | None = None,
) -> tuple[int, int, str] | None:
    """First ``(frame, opponent_tid, touch_kind)`` matching the demo opponent-touch gate."""
    touch_cfg = config.touch_validation_config()
    for frame_idx in range(start_frame + 1, end_frame):
        dets = frames_by_idx.get(frame_idx)
        if dets is None:
            continue
        transformer = transformers.get(frame_idx) if metric else None
        carrier, touch_kind = _active_carrier(
            dets, transformer=transformer, config=config
        )
        touch_kind = touch_kind or "reception"
        if carrier is None or int(carrier.team) == passer_team:
            continue
        if require_control and touch_kind != "control":
            continue
        if dets.tracker_id is None:
            continue
        tid = int(dets.tracker_id[carrier.index])
        if player_tid is not None and tid != player_tid:
            continue
        if not is_valid_possession_touch(
            dets,
            carrier,
            touch_kind=touch_kind,
            config=touch_cfg,
            **_touch_validation_kwargs(
                frames_by_idx,
                frame_idx,
                transformers=transformers,
                metric=metric,
                fps=fps,
                max_lookback=config.ball_speed_lookback_frames,
                min_lookback=config.ball_speed_min_lookback_frames,
                release_ball=release_ball,
                release_frame=release_frame,
            ),
        ):
            continue
        return frame_idx, tid, touch_kind
    return None


def _opponent_control_streak_between(
    frames_by_idx: dict[int, sv.Detections],
    *,
    start_frame: int,
    end_frame: int,
    passer_team: int,
    config: PassDetectionConfig,
    transformers: dict[int, object],
    metric: bool,
    min_streak: int,
    fps: float = 25.0,
    release_frame: int | None = None,
    release_ball: np.ndarray | None = None,
) -> bool:
    """True if an opponent had ``min_streak`` consecutive control frames in window."""
    streak = 0
    for frame_idx in range(start_frame + 1, end_frame):
        dets = frames_by_idx.get(frame_idx)
        if dets is None:
            streak = 0
            continue
        transformer = transformers.get(frame_idx) if metric else None
        carrier, touch_kind = _long_gap_opponent_carrier(
            dets, transformer=transformer, config=config
        )
        touch_kind = touch_kind or "reception"
        if carrier is None or int(carrier.team) == passer_team:
            streak = 0
            continue
        if touch_kind != "control":
            streak = 0
            continue
        if not _touch_valid_or_redirect(
            dets,
            carrier,
            touch_kind=touch_kind,
            frames_by_idx=frames_by_idx,
            frame_idx=frame_idx,
            config=config,
            transformers=transformers,
            metric=metric,
            fps=fps,
            release_ball=release_ball,
            release_frame=release_frame,
        ):
            streak = 0
            continue
        streak += 1
        if streak >= min_streak:
            return True
    return False


def _opponent_active_control_frames(
    frames_by_idx: dict[int, sv.Detections],
    *,
    start_frame: int,
    end_frame: int,
    passer_team: int,
    config: PassDetectionConfig,
    transformers: dict[int, object],
    metric: bool,
    fps: float = 25.0,
    release_frame: int | None = None,
    release_ball: np.ndarray | None = None,
) -> int:
    """Count in-window frames where an opponent holds validated tight control."""
    count = 0
    for frame_idx in range(start_frame + 1, end_frame):
        dets = frames_by_idx.get(frame_idx)
        if dets is None:
            continue
        transformer = transformers.get(frame_idx) if metric else None
        carrier, touch_kind = _long_gap_opponent_carrier(
            dets, transformer=transformer, config=config
        )
        touch_kind = touch_kind or "reception"
        if (
            carrier is None
            or int(carrier.team) == passer_team
            or touch_kind != "control"
        ):
            continue
        if not _touch_valid_or_redirect(
            dets,
            carrier,
            touch_kind=touch_kind,
            frames_by_idx=frames_by_idx,
            frame_idx=frame_idx,
            config=config,
            transformers=transformers,
            metric=metric,
            fps=fps,
            release_ball=release_ball,
            release_frame=release_frame,
        ):
            continue
        count += 1
    return count


def _opponent_secured_control_between(
    frames_by_idx: dict[int, sv.Detections],
    *,
    start_frame: int,
    end_frame: int,
    passer_team: int,
    config: PassDetectionConfig,
    transformers: dict[int, object],
    metric: bool,
    fps: float = 25.0,
) -> bool:
    """True when an opponent secured brief control (not a one-frame fly-by)."""
    first = _first_opponent_touch_in_window(
        frames_by_idx,
        start_frame=start_frame,
        end_frame=end_frame,
        passer_team=passer_team,
        config=config,
        transformers=transformers,
        metric=metric,
        require_control=True,
        fps=fps,
    )
    if first is None:
        return False
    opp_frame, opp_tid, _ = first
    return (
        _interceptor_secured_control_frame(
            frames_by_idx,
            start_frame=opp_frame,
            end_frame=end_frame,
            interceptor_tid=opp_tid,
            config=config,
            transformers=transformers,
            metric=metric,
            fps=fps,
        )
        is not None
    )


def _opponent_blocks_between(
    frames_by_idx: dict[int, sv.Detections],
    *,
    start_frame: int,
    end_frame: int,
    passer_team: int,
    gap_frames: int,
    config: PassDetectionConfig,
    transformers: dict[int, object],
    metric: bool,
    fps: float = 25.0,
    release_frame: int | None = None,
    release_ball: np.ndarray | None = None,
) -> bool:
    """Short gaps: any opponent touch; long gaps: sustained tight opponent control."""
    if gap_frames >= config.adjacent_pass_max_gap_frames:
        # Rolling release anchors can push a contested pickup barely over the
        # adjacent threshold; still apply the short-gap touch veto there.
        near_adjacent = gap_frames <= (
            config.adjacent_pass_max_gap_frames + config.pre_flight_release_window
        )
        if near_adjacent and _opponent_touch_between(
            frames_by_idx,
            start_frame=start_frame,
            end_frame=end_frame,
            passer_team=passer_team,
            config=config,
            transformers=transformers,
            metric=metric,
            require_control=False,
            fps=fps,
            release_frame=release_frame,
            release_ball=release_ball,
        ):
            return True
        if _opponent_control_streak_between(
            frames_by_idx,
            start_frame=start_frame,
            end_frame=end_frame,
            passer_team=passer_team,
            config=config,
            transformers=transformers,
            metric=metric,
            min_streak=config.min_long_gap_opponent_control_streak,
            fps=fps,
            release_frame=release_frame,
            release_ball=release_ball,
        ):
            return True
        return (
            _opponent_active_control_frames(
                frames_by_idx,
                start_frame=start_frame,
                end_frame=end_frame,
                passer_team=passer_team,
                config=config,
                transformers=transformers,
                metric=metric,
                fps=fps,
                release_frame=release_frame,
                release_ball=release_ball,
            )
            >= config.min_long_gap_opponent_control_streak
        )
    return _opponent_touch_between(
        frames_by_idx,
        start_frame=start_frame,
        end_frame=end_frame,
        passer_team=passer_team,
        config=config,
        transformers=transformers,
        metric=metric,
        require_control=False,
        fps=fps,
        release_frame=release_frame,
        release_ball=release_ball,
    )


def _opponent_touch_between(
    frames_by_idx: dict[int, sv.Detections],
    *,
    start_frame: int,
    end_frame: int,
    passer_team: int,
    config: PassDetectionConfig,
    transformers: dict[int, object],
    metric: bool,
    require_control: bool = False,
    fps: float = 25.0,
    release_frame: int | None = None,
    release_ball: np.ndarray | None = None,
) -> bool:
    """True if an opponent had a valid touch between two teammate beats."""
    return (
        _first_opponent_touch_in_window(
            frames_by_idx,
            start_frame=start_frame,
            end_frame=end_frame,
            passer_team=passer_team,
            config=config,
            transformers=transformers,
            metric=metric,
            require_control=require_control,
            fps=fps,
            release_frame=release_frame,
            release_ball=release_ball,
        )
        is not None
    )


def _opponent_control_between(
    frames_by_idx: dict[int, sv.Detections],
    *,
    start_frame: int,
    end_frame: int,
    passer_team: int,
    config: PassDetectionConfig,
    transformers: dict[int, object],
    metric: bool,
    fps: float = 25.0,
) -> bool:
    """True if an opponent had sustained control between two teammate touches."""
    return _opponent_touch_between(
        frames_by_idx,
        start_frame=start_frame,
        end_frame=end_frame,
        passer_team=passer_team,
        config=config,
        transformers=transformers,
        metric=metric,
        require_control=True,
        fps=fps,
    )


def _min_travel_threshold(config: PassDetectionConfig, *, metric: bool) -> float:
    return config.min_ball_travel_m if metric else config.min_ball_travel_px


def _try_emit_pass(
    events: list[InferredPass],
    *,
    release_frame: int,
    release_carrier: Carrier,
    passer_tid: int,
    receiver_tid: int,
    arrival_frame: int,
    arrival_carrier: Carrier,
    touch_kind: str,
    config: PassDetectionConfig,
    metric: bool,
    transformers: dict[int, object],
) -> bool:
    """Append one inferred pass if all gates pass."""
    gap = arrival_frame - release_frame
    if gap < config.min_carrier_gap_frames or gap > config.max_pass_gap_frames:
        return False

    travel = _effective_ball_travel(
        release_carrier.ball,
        arrival_carrier.ball,
        metric=metric,
        transformer_from=transformers.get(release_frame),
        transformer_to=transformers.get(arrival_frame),
        config=config,
    )
    min_travel = _min_travel_threshold(config, metric=metric)
    if travel is None or travel < min_travel:
        return False

    if _recent_duplicate(
        events,
        passer_tid,
        receiver_tid,
        release_frame,
        window=config.dedupe_window_frames,
    ):
        return False

    events.append(
        InferredPass(
            frame_idx=release_frame,
            passer_tid=passer_tid,
            receiver_tid=receiver_tid,
            team=int(release_carrier.team),
            gap_frames=gap,
            pass_length_m=float(travel) if metric else None,
            touch_kind=touch_kind,
        )
    )
    return True


def _active_carrier(
    dets: sv.Detections,
    *,
    transformer,
    config: PassDetectionConfig,
) -> tuple[Carrier | None, str | None]:
    """Possession carrier; reception requires pixel+metric when homography is available."""
    require_both_reception = transformer is not None
    control = find_control_carrier(
        dets,
        transformer=transformer,
        max_distance_px=config.control_max_distance_px,
        max_distance_m=config.control_max_distance_m,
        require_both_spaces=False,
    )
    if control is not None:
        return control, "control"
    reception = find_reception_carrier(
        dets,
        transformer=transformer,
        max_distance_px=config.reception_max_distance_px,
        max_distance_m=config.reception_max_distance_m,
        require_both_spaces=require_both_reception,
    )
    if reception is not None:
        return reception, "reception"
    return None, None


def _long_gap_opponent_carrier(
    dets: sv.Detections,
    *,
    transformer,
    config: PassDetectionConfig,
) -> tuple[Carrier | None, str | None]:
    """Stricter opponent control during long in-flight passes.

    With homography: both pixel and metric must agree (real feet control).
    Without homography: tight pixel-only control (reject wide bbox overlaps).
    """
    if transformer is not None:
        control = find_control_carrier(
            dets,
            transformer=transformer,
            max_distance_px=config.control_max_distance_px,
            max_distance_m=config.control_max_distance_m,
            require_both_spaces=True,
        )
        if control is not None:
            return control, "control"
        return None, None

    control = find_control_carrier(
        dets,
        transformer=None,
        max_distance_px=config.long_gap_opponent_control_max_px,
        max_distance_m=config.control_max_distance_m,
    )
    if control is not None:
        return control, "control"
    return None, None


@dataclass
class _TeamPossessionState:
    release: tuple[int, sv.Detections, Carrier, int] | None = None
    release_is_one_touch: bool = False
    last_touch: tuple[int, sv.Detections, Carrier, int] | None = None
    possession_tid: int = -1
    control_streak: int = 0
    reception_streak: int = 0
    in_flight: bool = False
    arrival_candidate_tid: int = -1
    arrival_streak: int = 0
    arrival_control_streak: int = 0
    turnover_snapshot: tuple[int, int] | None = None
    last_possession_frame: int = -1
    last_possessor_tid: int = -1


def _possession_staleness_anchor(state: _TeamPossessionState) -> int | None:
    """Frame clock for release expiry: latest of pass anchor and last valid team touch."""
    if state.release is None:
        return None
    release_frame = state.release[0]
    if state.last_possession_frame >= 0:
        return max(release_frame, state.last_possession_frame)
    return release_frame


def _pass_flight_window(event: InferredPass) -> tuple[int, int]:
    return (event.frame_idx, event.frame_idx + event.gap_frames)


def _filter_overlapping_cross_team_passes(
    passes: list[InferredPass],
) -> list[InferredPass]:
    """Keep at most one pass per overlapping flight window; prefer the earlier release."""
    if len(passes) < 2:
        return passes
    drop: set[int] = set()
    for i, a in enumerate(passes):
        if i in drop:
            continue
        for j, b in enumerate(passes):
            if j <= i or j in drop or a.team == b.team:
                continue
            a_start, a_end = _pass_flight_window(a)
            b_start, b_end = _pass_flight_window(b)
            if a_start <= b_end and b_start <= a_end:
                drop.add(j if b_start >= a_start else i)
    return [p for idx, p in enumerate(passes) if idx not in drop]


def passes_for_overlay(passes: tuple[InferredPass, ...] | list[InferredPass]) -> tuple[InferredPass, ...]:
    """Passes safe for in-flight overlays (no overlapping cross-team flight windows)."""
    return tuple(_filter_overlapping_cross_team_passes(list(passes)))


@dataclass
class _PendingRedirectTurnover:
    losing_team: int
    interceptor_tid: int
    interceptor_team: int
    redirect_frame: int
    emit_after_frame: int
    expire_after_frame: int


@dataclass
class _PendingTurnoverEmit:
    losing_team: int
    release_frame: int
    passer_tid: int
    interceptor_tid: int
    interceptor_team: int
    interception_frame: int
    emit_after_frame: int


def _is_goalkeeper(dets: sv.Detections, carrier_index: int) -> bool:
    return int(dets.class_id[carrier_index]) == ROLE_GOALKEEPER


def _min_control_frames_for(
    dets: sv.Detections,
    carrier: Carrier,
    *,
    config: PassDetectionConfig,
) -> int:
    if _is_goalkeeper(dets, carrier.index):
        return config.min_gk_control_frames
    return config.min_control_frames


def _player_had_min_control_at(
    frames_by_idx: dict[int, sv.Detections],
    *,
    player_tid: int,
    end_frame: int,
    min_control: int,
    config: PassDetectionConfig,
    transformers: dict[int, object],
    metric: bool,
    fps: float = 25.0,
) -> bool:
    """True when ``player_tid`` had ``min_control`` consecutive control ending at ``end_frame``."""
    touch_cfg = config.touch_validation_config()
    streak = 0
    for frame_idx in range(end_frame, max(0, end_frame - config.max_pass_gap_frames) - 1, -1):
        dets = frames_by_idx.get(frame_idx)
        if dets is None:
            streak = 0
            continue
        transformer = transformers.get(frame_idx) if metric else None
        carrier, touch_kind = _active_carrier(
            dets, transformer=transformer, config=config
        )
        touch_kind = touch_kind or "reception"
        if (
            carrier is None
            or touch_kind != "control"
            or dets.tracker_id is None
            or int(dets.tracker_id[carrier.index]) != player_tid
        ):
            streak = 0
            continue
        if not is_valid_possession_touch(
            dets,
            carrier,
            touch_kind=touch_kind,
            config=touch_cfg,
            **_touch_validation_kwargs(
                frames_by_idx,
                frame_idx,
                transformers=transformers,
                metric=metric,
                fps=fps,
                max_lookback=config.ball_speed_lookback_frames,
                min_lookback=config.ball_speed_min_lookback_frames,
            ),
        ):
            streak = 0
            continue
        streak += 1
        if streak >= min_control:
            return True
    return False


def _player_had_min_reception_at(
    frames_by_idx: dict[int, sv.Detections],
    *,
    player_tid: int,
    end_frame: int,
    min_reception: int,
    config: PassDetectionConfig,
    transformers: dict[int, object],
    metric: bool,
    fps: float = 25.0,
) -> bool:
    """True when ``player_tid`` had ``min_reception`` consecutive receptions ending at ``end_frame``."""
    touch_cfg = config.touch_validation_config()
    streak = 0
    for frame_idx in range(end_frame, max(0, end_frame - config.max_pass_gap_frames) - 1, -1):
        dets = frames_by_idx.get(frame_idx)
        if dets is None:
            streak = 0
            continue
        transformer = transformers.get(frame_idx) if metric else None
        carrier, touch_kind = _active_carrier(
            dets, transformer=transformer, config=config
        )
        touch_kind = touch_kind or "reception"
        if (
            carrier is None
            or touch_kind != "reception"
            or dets.tracker_id is None
            or int(dets.tracker_id[carrier.index]) != player_tid
        ):
            streak = 0
            continue
        if not is_valid_possession_touch(
            dets,
            carrier,
            touch_kind=touch_kind,
            config=touch_cfg,
            **_touch_validation_kwargs(
                frames_by_idx,
                frame_idx,
                transformers=transformers,
                metric=metric,
                fps=fps,
                max_lookback=config.ball_speed_lookback_frames,
                min_lookback=config.ball_speed_min_lookback_frames,
            ),
        ):
            streak = 0
            continue
        streak += 1
        if streak >= min_reception:
            return True
    return False


def _had_confirmed_one_touch_release_at(
    frames_by_idx: dict[int, sv.Detections],
    *,
    player_tid: int,
    end_frame: int,
    config: PassDetectionConfig,
    transformers: dict[int, object],
    metric: bool,
    fps: float = 25.0,
) -> bool:
    """True when ``player_tid`` flicked the ball on a validated one-touch release."""
    if not _player_had_min_reception_at(
        frames_by_idx,
        player_tid=player_tid,
        end_frame=end_frame,
        min_reception=config.min_one_touch_reception_frames,
        config=config,
        transformers=transformers,
        metric=metric,
        fps=fps,
    ):
        return False
    touch_cfg = config.touch_validation_config()
    for frame_idx in range(
        end_frame, max(0, end_frame - config.one_touch_release_window_frames) - 1, -1
    ):
        dets = frames_by_idx.get(frame_idx)
        if dets is None:
            continue
        transformer = transformers.get(frame_idx) if metric else None
        carrier, touch_kind = _active_carrier(
            dets, transformer=transformer, config=config
        )
        touch_kind = touch_kind or "reception"
        if (
            carrier is None
            or touch_kind != "reception"
            or dets.tracker_id is None
            or int(dets.tracker_id[carrier.index]) != player_tid
        ):
            continue
        if not is_valid_possession_touch(
            dets,
            carrier,
            touch_kind=touch_kind,
            config=touch_cfg,
            **_touch_validation_kwargs(
                frames_by_idx,
                frame_idx,
                transformers=transformers,
                metric=metric,
                fps=fps,
                max_lookback=config.ball_speed_lookback_frames,
                min_lookback=config.ball_speed_min_lookback_frames,
            ),
        ):
            continue
        frame_balls: list[tuple[int, np.ndarray]] = []
        for fi in range(
            frame_idx + 1,
            min(frame_idx + config.one_touch_release_window_frames + 1, end_frame + 15),
        ):
            ball = ball_xy(frames_by_idx.get(fi))
            if ball is not None:
                frame_balls.append((fi, ball))
        if ball_departed_for_one_touch(
            carrier.ball,
            ball_xy(frames_by_idx.get(end_frame)),
            touch_frame=frame_idx,
            in_flight_frame=end_frame,
            depart_min_px=config.one_touch_depart_min_px,
            frame_balls=frame_balls or None,
        ):
            return True
        continue
    return False


def _player_had_committed_possession_at(
    frames_by_idx: dict[int, sv.Detections],
    *,
    player_tid: int,
    end_frame: int,
    release_dets: sv.Detections,
    release_carrier: Carrier,
    config: PassDetectionConfig,
    transformers: dict[int, object],
    metric: bool,
    fps: float = 25.0,
) -> bool:
    """Control streak or a confirmed one-touch reception release at ``end_frame``."""
    if _player_had_min_control_at(
        frames_by_idx,
        player_tid=player_tid,
        end_frame=end_frame,
        min_control=_min_turnover_passer_control(
            release_dets, release_carrier, config=config
        ),
        config=config,
        transformers=transformers,
        metric=metric,
        fps=fps,
    ):
        return True
    return _had_confirmed_one_touch_release_at(
        frames_by_idx,
        player_tid=player_tid,
        end_frame=end_frame,
        config=config,
        transformers=transformers,
        metric=metric,
        fps=fps,
    )


def _interceptor_secured_possession_frame(
    frames_by_idx: dict[int, sv.Detections],
    *,
    start_frame: int,
    end_frame: int,
    interceptor_tid: int,
    config: PassDetectionConfig,
    transformers: dict[int, object],
    metric: bool,
    fps: float = 25.0,
) -> int | None:
    """First frame where the interceptor settles the ball (control or brief reception)."""
    secured_control = _interceptor_secured_control_frame(
        frames_by_idx,
        start_frame=start_frame,
        end_frame=end_frame,
        interceptor_tid=interceptor_tid,
        config=config,
        transformers=transformers,
        metric=metric,
        fps=fps,
    )
    if secured_control is not None:
        return secured_control
    touch_cfg = config.touch_validation_config()
    streak = 0
    first_reception: int | None = None
    for frame_idx in range(start_frame, end_frame + 1):
        dets = frames_by_idx.get(frame_idx)
        if dets is None:
            streak = 0
            continue
        transformer = transformers.get(frame_idx) if metric else None
        carrier, touch_kind = _active_carrier(
            dets, transformer=transformer, config=config
        )
        touch_kind = touch_kind or "reception"
        if (
            carrier is None
            or dets.tracker_id is None
            or int(dets.tracker_id[carrier.index]) != interceptor_tid
        ):
            streak = 0
            continue
        if not is_valid_possession_touch(
            dets,
            carrier,
            touch_kind=touch_kind,
            config=touch_cfg,
            **_touch_validation_kwargs(
                frames_by_idx,
                frame_idx,
                transformers=transformers,
                metric=metric,
                fps=fps,
                max_lookback=config.ball_speed_lookback_frames,
                min_lookback=config.ball_speed_min_lookback_frames,
            ),
        ):
            streak = 0
            continue
        if touch_kind == "control":
            return frame_idx
        streak += 1
        if first_reception is None:
            first_reception = frame_idx
        if streak >= config.min_reception_arrival_frames:
            return first_reception
    return None


def _interceptor_min_control(
    dets: sv.Detections,
    carrier: Carrier,
    *,
    config: PassDetectionConfig,
) -> int:
    """Interceptors need less streak than passers; fly-by touches don't count."""
    base = _min_control_frames_for(dets, carrier, config=config)
    return min(base, config.min_long_gap_opponent_control_streak)


def _interceptor_secured_control_frame(
    frames_by_idx: dict[int, sv.Detections],
    *,
    start_frame: int,
    end_frame: int,
    interceptor_tid: int,
    config: PassDetectionConfig,
    transformers: dict[int, object],
    metric: bool,
    fps: float = 25.0,
) -> int | None:
    """First frame in ``[start_frame, end_frame]`` where the interceptor holds control."""
    for frame_idx in range(start_frame, end_frame + 1):
        dets = frames_by_idx.get(frame_idx)
        if dets is None:
            continue
        transformer = transformers.get(frame_idx) if metric else None
        carrier, touch_kind = _active_carrier(
            dets, transformer=transformer, config=config
        )
        touch_kind = touch_kind or "reception"
        if (
            carrier is None
            or touch_kind != "control"
            or dets.tracker_id is None
            or int(dets.tracker_id[carrier.index]) != interceptor_tid
        ):
            continue
        min_control = _interceptor_min_control(dets, carrier, config=config)
        if _player_had_min_control_at(
            frames_by_idx,
            player_tid=interceptor_tid,
            end_frame=frame_idx,
            min_control=min_control,
            config=config,
            transformers=transformers,
            metric=metric,
            fps=fps,
        ):
            return frame_idx
    return None


def _bridge_missing_ball(
    team_states: dict[int, _TeamPossessionState],
    *,
    frame_idx: int,
    config: PassDetectionConfig,
    frames_by_idx: dict[int, sv.Detections],
    transformers: dict[int, object],
    metric: bool,
    fps: float = 25.0,
) -> None:
    """Brief ball dropout: keep release anchor and arrival streak (like in-flight)."""
    for team_id, state in team_states.items():
        if state.release is None and state.last_touch is None:
            continue
        if not state.in_flight:
            _promote_in_flight_release(
                state,
                frame_idx,
                config=config,
                team=team_id,
                other_state=team_states[1 - team_id],
                frames_by_idx=frames_by_idx,
                transformers=transformers,
                metric=metric,
                fps=fps,
            )
        state.in_flight = True


def _end_missing_ball_bridge(team_states: dict[int, _TeamPossessionState]) -> None:
    """Ball gone too long — stop bridging unless a release anchor is still live."""
    for state in team_states.values():
        state.arrival_candidate_tid = -1
        state.arrival_streak = 0
        state.arrival_control_streak = 0
        if state.release is not None:
            state.in_flight = True
            continue
        state.in_flight = False


def _promote_pre_flight_release(
    state: _TeamPossessionState,
    frame_idx: int,
    *,
    config: PassDetectionConfig,
    team: int | None = None,
    other_state: _TeamPossessionState | None = None,
    frames_by_idx: dict[int, sv.Detections] | None = None,
    transformers: dict[int, object] | None = None,
    metric: bool = False,
    fps: float = 25.0,
) -> None:
    """Credit a brief touch as the passer when the ball immediately goes in-flight."""
    if state.last_touch is None:
        return
    touch_frame, touch_dets, touch_carrier, touch_tid = state.last_touch
    if frame_idx - touch_frame > config.pre_flight_release_window:
        return
    touch_cfg = config.touch_validation_config()
    transformer = (
        transformers.get(touch_frame) if metric and transformers is not None else None
    )
    carrier_now, touch_kind = _active_carrier(
        touch_dets, transformer=transformer, config=config
    )
    if (
        carrier_now is None
        or touch_kind != "control"
        or touch_dets.tracker_id is None
        or int(touch_dets.tracker_id[carrier_now.index]) != touch_tid
    ):
        return
    if not is_valid_possession_touch(
        touch_dets,
        carrier_now,
        touch_kind="control",
        config=touch_cfg,
        **_touch_validation_kwargs(
            frames_by_idx or {},
            touch_frame,
            transformers=transformers or {},
            metric=metric,
            fps=fps,
            max_lookback=config.ball_speed_lookback_frames,
            min_lookback=config.ball_speed_min_lookback_frames,
        ),
    ):
        return
    if is_aerial_flyby_below_feet(
        touch_dets,
        touch_carrier,
        threshold_px=reception_aerial_veto_threshold(touch_cfg),
    ):
        return
    if (
        frames_by_idx is not None
        and transformers is not None
        and not _player_had_min_control_at(
            frames_by_idx,
            player_tid=touch_tid,
            end_frame=touch_frame,
            min_control=_min_turnover_passer_control(
                touch_dets, touch_carrier, config=config
            ),
            config=config,
            transformers=transformers,
            metric=metric,
            fps=fps,
        )
    ):
        return
    if state.release is not None and state.release[3] == touch_tid:
        return
    if state.release is not None and state.release[3] != touch_tid:
        release_frame = state.release[0]
        opponent_between = False
        if (
            team is not None
            and frames_by_idx is not None
            and transformers is not None
        ):
            opponent_between = _opponent_blocks_between(
                frames_by_idx,
                start_frame=release_frame,
                end_frame=touch_frame,
                passer_team=team,
                gap_frames=touch_frame - release_frame,
                config=config,
                transformers=transformers,
                metric=metric,
                fps=fps,
            )
        other_in_play = (
            other_state is not None
            and other_state.release is not None
            and touch_frame > other_state.release[0]
        )
        if (
            other_in_play
            and not opponent_between
            and touch_tid == state.arrival_candidate_tid
            and state.arrival_streak > 0
        ):
            # Opponent team has possession elsewhere; keep the in-flight pass
            # anchor while the intended receiver builds an arrival streak.
            return
        if opponent_between or other_in_play:
            # Opponent has the ball or a teammate deflected en route — drop the
            # stale passer anchor; a fly-by near another teammate is not a new pass.
            state.release = None
            state.arrival_candidate_tid = -1
            state.arrival_streak = 0
            state.arrival_control_streak = 0
        return
    if state.release is not None:
        return
    if (
        other_state is not None
        and other_state.release is not None
        and touch_frame > other_state.release[0]
    ):
        # Opponent still owns a pending release — never steal from a skim/fly-by,
        # even when the ball passes inside the control radius.
        return
    state.release = (touch_frame, touch_dets, touch_carrier, touch_tid)
    state.release_is_one_touch = False


def _promote_in_flight_release(
    state: _TeamPossessionState,
    frame_idx: int,
    *,
    config: PassDetectionConfig,
    team: int,
    other_state: _TeamPossessionState,
    frames_by_idx: dict[int, sv.Detections],
    transformers: dict[int, object],
    metric: bool,
    fps: float,
) -> None:
    """Try control-based and one-touch reception release anchors."""
    kwargs = dict(
        config=config,
        team=team,
        other_state=other_state,
        frames_by_idx=frames_by_idx,
        transformers=transformers,
        metric=metric,
        fps=fps,
    )
    _promote_pre_flight_release(state, frame_idx, **kwargs)
    _promote_one_touch_reception_release(state, frame_idx, **kwargs)


def _promote_one_touch_reception_release(
    state: _TeamPossessionState,
    frame_idx: int,
    *,
    config: PassDetectionConfig,
    team: int | None = None,
    other_state: _TeamPossessionState | None = None,
    frames_by_idx: dict[int, sv.Detections] | None = None,
    transformers: dict[int, object] | None = None,
    metric: bool = False,
    fps: float = 25.0,
) -> None:
    """Credit a give-and-go when reception ends and the ball leaves immediately."""
    if state.last_touch is None or state.release is not None:
        return
    touch_frame, touch_dets, touch_carrier, touch_tid = state.last_touch
    if frame_idx - touch_frame > config.one_touch_release_window_frames:
        return
    touch_cfg = config.touch_validation_config()
    transformer = (
        transformers.get(touch_frame) if metric and transformers is not None else None
    )
    carrier_now, touch_kind = _active_carrier(
        touch_dets, transformer=transformer, config=config
    )
    if (
        carrier_now is None
        or touch_kind != "reception"
        or touch_dets.tracker_id is None
        or int(touch_dets.tracker_id[carrier_now.index]) != touch_tid
    ):
        return
    if not is_valid_possession_touch(
        touch_dets,
        carrier_now,
        touch_kind="reception",
        config=touch_cfg,
        **_touch_validation_kwargs(
            frames_by_idx or {},
            touch_frame,
            transformers=transformers or {},
            metric=metric,
            fps=fps,
            max_lookback=config.ball_speed_lookback_frames,
            min_lookback=config.ball_speed_min_lookback_frames,
        ),
    ):
        return
    if is_aerial_flyby_below_feet(
        touch_dets,
        touch_carrier,
        threshold_px=reception_aerial_veto_threshold(touch_cfg),
    ):
        return
    if (
        frames_by_idx is not None
        and transformers is not None
        and not _player_had_min_reception_at(
            frames_by_idx,
            player_tid=touch_tid,
            end_frame=touch_frame,
            min_reception=config.min_one_touch_reception_frames,
            config=config,
            transformers=transformers,
            metric=metric,
            fps=fps,
        )
    ):
        return
    in_flight_ball = (
        ball_xy(frames_by_idx[frame_idx])
        if frames_by_idx is not None and frame_idx in frames_by_idx
        else None
    )
    frame_balls: list[tuple[int, np.ndarray]] = []
    if frames_by_idx is not None:
        for fi in range(touch_frame + 1, frame_idx):
            ball = ball_xy(frames_by_idx.get(fi))
            if ball is not None:
                frame_balls.append((fi, ball))
    if not ball_departed_for_one_touch(
        touch_carrier.ball,
        in_flight_ball,
        touch_frame=touch_frame,
        in_flight_frame=frame_idx,
        depart_min_px=config.one_touch_depart_min_px,
        frame_balls=frame_balls or None,
    ):
        return
    if (
        other_state is not None
        and other_state.release is not None
        and touch_frame > other_state.release[0]
    ):
        # Opponent still owns a pending release — one-touch promote would mint a
        # false passer from a fly-by.
        return
    state.release = (touch_frame, touch_dets, touch_carrier, touch_tid)
    state.release_is_one_touch = True


def _recent_duplicate(
    events: list[InferredPass],
    passer_tid: int,
    receiver_tid: int,
    frame_idx: int,
    *,
    window: int,
) -> bool:
    for event in reversed(events):
        if frame_idx - event.frame_idx > window:
            break
        if event.passer_tid == passer_tid and event.receiver_tid == receiver_tid:
            return True
    return False


def _confirm_release(
    state: _TeamPossessionState,
    frame_idx: int,
    dets: sv.Detections,
    carrier: Carrier,
    tid: int,
) -> None:
    state.release = (frame_idx, dets, carrier, tid)
    state.release_is_one_touch = False
    if state.turnover_snapshot is not None:
        state.turnover_snapshot = (frame_idx, tid)


def _recent_turnover_duplicate(
    events: list[InferredTurnover],
    passer_tid: int,
    interceptor_tid: int,
    release_frame: int,
    *,
    window: int,
) -> bool:
    for event in reversed(events):
        if release_frame - event.release_frame > window:
            break
        if (
            event.passer_tid == passer_tid
            and event.interceptor_tid == interceptor_tid
        ):
            return True
    return False


def _min_turnover_passer_control(
    dets: sv.Detections,
    carrier: Carrier,
    *,
    config: PassDetectionConfig,
) -> int:
    """Turnover passer needs sustained control; fly-by single frames do not count."""
    if _is_goalkeeper(dets, carrier.index):
        return config.min_gk_control_frames
    return config.min_turnover_passer_control_frames


def _interceptor_follow_through_after_redirect(
    frames_by_idx: dict[int, sv.Detections],
    *,
    interceptor_tid: int,
    interceptor_team: int,
    redirect_frame: int,
    config: PassDetectionConfig,
    transformers: dict[int, object],
    metric: bool,
    fps: float,
) -> bool:
    """True when the interceptor keeps the ball (control, one-touch, or pass to teammate)."""
    short_end = redirect_frame + config.turnover_recovery_window_frames
    long_end = redirect_frame + min(config.max_pass_gap_frames, 60)
    if _player_had_min_control_at(
        frames_by_idx,
        player_tid=interceptor_tid,
        end_frame=short_end,
        min_control=config.min_control_frames,
        config=config,
        transformers=transformers,
        metric=metric,
        fps=fps,
    ):
        return True
    if _had_confirmed_one_touch_release_at(
        frames_by_idx,
        player_tid=interceptor_tid,
        end_frame=short_end,
        config=config,
        transformers=transformers,
        metric=metric,
        fps=fps,
    ):
        return True
    for frame_idx in range(redirect_frame, long_end + 1):
        dets = frames_by_idx.get(frame_idx)
        if dets is None:
            continue
        transformer = transformers.get(frame_idx) if metric else None
        carrier, touch_kind = _active_carrier(
            dets, transformer=transformer, config=config
        )
        touch_kind = touch_kind or "reception"
        if carrier is None or int(carrier.team) != interceptor_team:
            continue
        if dets.tracker_id is None:
            continue
        tid = int(dets.tracker_id[carrier.index])
        if tid == interceptor_tid or tid < 0:
            continue
        if not _touch_valid_or_redirect(
            dets,
            carrier,
            touch_kind=touch_kind,
            frames_by_idx=frames_by_idx,
            frame_idx=frame_idx,
            config=config,
            transformers=transformers,
            metric=metric,
            fps=fps,
        ):
            continue
        gap = frame_idx - redirect_frame
        if (
            gap >= config.min_carrier_gap_frames
            and gap <= config.redirect_instant_pass_max_gap_frames
        ):
            return True
    return False


def _opponent_redirect_in_gap(
    frames_by_idx: dict[int, sv.Detections],
    *,
    start_frame: int,
    end_frame: int,
    passer_team: int,
    config: PassDetectionConfig,
    transformers: dict[int, object],
    metric: bool,
    fps: float,
) -> bool:
    """True when an opponent redirect touch occurs between two teammate beats."""
    for frame_idx in range(start_frame + 1, end_frame):
        dets = frames_by_idx.get(frame_idx)
        if dets is None:
            continue
        transformer = transformers.get(frame_idx) if metric else None
        carrier, touch_kind = _active_carrier(
            dets, transformer=transformer, config=config
        )
        touch_kind = touch_kind or "reception"
        if carrier is None or int(carrier.team) == passer_team:
            continue
        if not _ball_redirected(frames_by_idx, frame_idx, config=config):
            continue
        if not _touch_valid_or_redirect(
            dets,
            carrier,
            touch_kind=touch_kind,
            frames_by_idx=frames_by_idx,
            frame_idx=frame_idx,
            config=config,
            transformers=transformers,
            metric=metric,
            fps=fps,
        ):
            continue
        return True
    return False


def _invalidate_passes_for_turnover(
    passes: list[InferredPass],
    turnover: InferredTurnover,
    frames_by_idx: dict[int, sv.Detections],
    *,
    config: PassDetectionConfig,
    transformers: dict[int, object],
    metric: bool,
    fps: float,
) -> None:
    """Drop passes superseded by a confirmed turnover."""
    origin_lo = max(0, turnover.release_frame - config.dedupe_window_frames)

    def _drop(p: InferredPass) -> bool:
        arrival = p.frame_idx + p.gap_frames
        if (
            p.passer_tid == turnover.passer_tid
            and p.team == turnover.passer_team
            and p.frame_idx == turnover.release_frame
            and arrival < turnover.interception_frame
        ):
            return True
        if (
            p.passer_tid == turnover.passer_tid
            and p.team == turnover.passer_team
            and p.frame_idx >= origin_lo
            and arrival <= turnover.interception_frame
            and _opponent_blocks_between(
                frames_by_idx,
                start_frame=p.frame_idx,
                end_frame=arrival,
                passer_team=p.team,
                gap_frames=p.gap_frames,
                config=config,
                transformers=transformers,
                metric=metric,
                fps=fps,
            )
        ):
            return True
        # Brief redirect anchors on the wrong intercepting player (tracker noise).
        if (
            p.team == turnover.interceptor_team
            and p.passer_tid != turnover.interceptor_tid
            and p.frame_idx >= turnover.release_frame
            and abs(arrival - turnover.interception_frame)
            <= config.dedupe_window_frames
        ):
            return True
        return False

    passes[:] = [p for p in passes if not _drop(p)]


def _queue_redirect_turnover(
    pending: list[_PendingRedirectTurnover],
    *,
    losing_team: int,
    interceptor_tid: int,
    interceptor_team: int,
    redirect_frame: int,
    config: PassDetectionConfig,
) -> None:
    pending.append(
        _PendingRedirectTurnover(
            losing_team=losing_team,
            interceptor_tid=interceptor_tid,
            interceptor_team=interceptor_team,
            redirect_frame=redirect_frame,
            emit_after_frame=redirect_frame + config.turnover_recovery_window_frames,
            expire_after_frame=redirect_frame
            + min(config.max_pass_gap_frames, 60),
        )
    )


def _flush_pending_redirect_turnovers(
    pending: list[_PendingRedirectTurnover],
    *,
    frame_idx: int,
    team_states: dict[int, _TeamPossessionState],
    turnovers: list[InferredTurnover],
    passes: list[InferredPass],
    frames_by_idx: dict[int, sv.Detections],
    config: PassDetectionConfig,
    transformers: dict[int, object],
    metric: bool,
    fps: float,
) -> None:
    remaining: list[_PendingRedirectTurnover] = []
    for item in pending:
        if frame_idx < item.emit_after_frame:
            remaining.append(item)
            continue
        if frame_idx > item.expire_after_frame:
            continue
        if not _interceptor_follow_through_after_redirect(
            frames_by_idx,
            interceptor_tid=item.interceptor_tid,
            interceptor_team=item.interceptor_team,
            redirect_frame=item.redirect_frame,
            config=config,
            transformers=transformers,
            metric=metric,
            fps=fps,
        ):
            remaining.append(item)
            continue
        losing_state = team_states[item.losing_team]
        _try_emit_possession_loss_turnover(
            losing_state,
            losing_team=item.losing_team,
            interceptor_tid=item.interceptor_tid,
            interceptor_team=item.interceptor_team,
            interception_frame=item.redirect_frame,
            turnovers=turnovers,
            passes=passes,
            config=config,
            frames_by_idx=frames_by_idx,
            transformers=transformers,
            metric=metric,
            fps=fps,
            redirect=True,
        )
    pending[:] = remaining


def _best_possession_loss_origin(
    frames_by_idx: dict[int, sv.Detections],
    *,
    losing_team: int,
    interception_frame: int,
    config: PassDetectionConfig,
    transformers: dict[int, object],
    metric: bool,
    fps: float,
) -> tuple[int, int] | None:
    """Most recent losing-team holder with min control before the intercept."""
    touch_cfg = config.touch_validation_config()
    min_gap = config.min_turnover_gap_frames
    for frame_idx in range(
        interception_frame - 1,
        max(0, interception_frame - config.max_pass_gap_frames) - 1,
        -1,
    ):
        if interception_frame - frame_idx < min_gap:
            continue
        dets = frames_by_idx.get(frame_idx)
        if dets is None:
            continue
        transformer = transformers.get(frame_idx) if metric else None
        carrier, touch_kind = _active_carrier(
            dets, transformer=transformer, config=config
        )
        touch_kind = touch_kind or "reception"
        if carrier is None or int(carrier.team) != losing_team:
            continue
        if dets.tracker_id is None:
            continue
        tid = int(dets.tracker_id[carrier.index])
        if tid < 0:
            continue
        if not _touch_valid_or_redirect(
            dets,
            carrier,
            touch_kind=touch_kind,
            frames_by_idx=frames_by_idx,
            frame_idx=frame_idx,
            config=config,
            transformers=transformers,
            metric=metric,
            fps=fps,
        ):
            continue
        if not _player_had_min_control_at(
            frames_by_idx,
            player_tid=tid,
            end_frame=frame_idx,
            min_control=config.min_turnover_passer_control_frames,
            config=config,
            transformers=transformers,
            metric=metric,
            fps=fps,
        ):
            continue
        return frame_idx, tid
    return None


def _ball_dropout_exceeds_between(
    frames_by_idx: dict[int, sv.Detections],
    *,
    start_frame: int,
    end_frame: int,
    max_dropout: int,
) -> bool:
    """True when ball detections are missing for longer than ``max_dropout`` frames."""
    streak = 0
    for frame_idx in range(start_frame + 1, end_frame):
        dets = frames_by_idx.get(frame_idx)
        if dets is None or ball_xy(dets) is None:
            streak += 1
            if streak > max_dropout:
                return True
        else:
            streak = 0
    return False


def _promote_redirect_one_touch_release(
    state: _TeamPossessionState,
    frame_idx: int,
    dets: sv.Detections,
    carrier: Carrier,
    tid: int,
) -> None:
    """Credit an instant redirect touch as a one-touch release anchor."""
    if state.release is not None and state.release[3] != tid:
        return
    state.release = (frame_idx, dets, carrier, tid)
    state.release_is_one_touch = True


def _try_emit_possession_loss_turnover(
    losing_state: _TeamPossessionState,
    *,
    losing_team: int,
    interceptor_tid: int,
    interceptor_team: int,
    interception_frame: int,
    turnovers: list[InferredTurnover],
    passes: list[InferredPass],
    config: PassDetectionConfig,
    frames_by_idx: dict[int, sv.Detections],
    transformers: dict[int, object],
    metric: bool,
    fps: float,
    redirect: bool = False,
) -> bool:
    """Turnover from the last committed possessor when possession changes teams."""
    origin = _best_possession_loss_origin(
        frames_by_idx,
        losing_team=losing_team,
        interception_frame=interception_frame,
        config=config,
        transformers=transformers,
        metric=metric,
        fps=fps,
    )
    if origin is None:
        return False
    release_frame, passer_tid = origin
    secured = _interceptor_secured_possession_frame(
        frames_by_idx,
        start_frame=interception_frame,
        end_frame=interception_frame + config.turnover_recovery_window_frames,
        interceptor_tid=interceptor_tid,
        config=config,
        transformers=transformers,
        metric=metric,
        fps=fps,
    )
    if secured is None:
        return False
    return _try_emit_turnover(
        losing_state,
        interceptor_tid=interceptor_tid,
        interceptor_team=interceptor_team,
        interception_frame=secured,
        turnovers=turnovers,
        config=config,
        release_frame=release_frame,
        passer_tid=passer_tid,
        frames_by_idx=frames_by_idx,
        transformers=transformers,
        metric=metric,
        fps=fps,
        skip_interceptor_control_check=True,
        passes=passes,
    )


def _try_emit_turnover(
    losing_state: _TeamPossessionState,
    *,
    interceptor_tid: int,
    interceptor_team: int,
    interception_frame: int,
    turnovers: list[InferredTurnover],
    config: PassDetectionConfig,
    release_frame: int | None = None,
    passer_tid: int | None = None,
    frames_by_idx: dict[int, sv.Detections] | None = None,
    transformers: dict[int, object] | None = None,
    metric: bool = True,
    fps: float = 25.0,
    skip_interceptor_control_check: bool = False,
    passes: list[InferredPass] | None = None,
) -> bool:
    """Rule 5: opponent takes the ball while our release is still pending."""
    release_dets: sv.Detections | None = None
    release_carrier: Carrier | None = None
    if release_frame is None or passer_tid is None:
        release = losing_state.release
        if release is None:
            return False
        release_frame, release_dets, release_carrier, passer_tid = release
    elif losing_state.release is not None:
        release_dets = losing_state.release[1]
        release_carrier = losing_state.release[2]
    elif frames_by_idx is not None and release_frame is not None and passer_tid is not None:
        release_dets = frames_by_idx.get(release_frame)
        if release_dets is not None:
            transformer = (
                transformers.get(release_frame) if metric and transformers else None
            )
            carrier, _ = _active_carrier(
                release_dets, transformer=transformer, config=config
            )
            if (
                carrier is not None
                and release_dets.tracker_id is not None
                and int(release_dets.tracker_id[carrier.index]) == passer_tid
            ):
                release_carrier = carrier

    if (
        frames_by_idx is not None
        and release_dets is not None
        and release_carrier is not None
        and not _player_had_committed_possession_at(
            frames_by_idx,
            player_tid=passer_tid,
            end_frame=release_frame,
            release_dets=release_dets,
            release_carrier=release_carrier,
            config=config,
            transformers=transformers or {},
            metric=metric,
            fps=fps,
        )
    ):
        return False

    if frames_by_idx is not None and not skip_interceptor_control_check:
        secured = _interceptor_secured_possession_frame(
            frames_by_idx,
            start_frame=interception_frame,
            end_frame=interception_frame + config.turnover_recovery_window_frames,
            interceptor_tid=interceptor_tid,
            config=config,
            transformers=transformers or {},
            metric=metric,
            fps=fps,
        )
        if secured is None:
            return False
        interception_frame = secured
        int_dets = frames_by_idx.get(interception_frame)
        if int_dets is None:
            return False
        transformer = transformers.get(interception_frame) if metric else None
        int_carrier, int_kind = _active_carrier(
            int_dets, transformer=transformer, config=config
        )
        if (
            int_carrier is None
            or int_dets.tracker_id is None
            or int(int_dets.tracker_id[int_carrier.index]) != interceptor_tid
        ):
            return False

    gap = interception_frame - release_frame
    if gap < config.min_turnover_gap_frames or gap > config.max_pass_gap_frames:
        return False

    if (
        frames_by_idx is not None
        and release_carrier is not None
        and release_frame is not None
    ):
        # Prefer a frame where the interceptor is valid under the release fly-by
        # gates (settled ball), not merely the first secured proximity frame.
        chosen: tuple[int, sv.Detections, Carrier, str] | None = None
        search_end = interception_frame + config.turnover_recovery_window_frames
        for fi in range(interception_frame, search_end + 1):
            int_dets = frames_by_idx.get(fi)
            if int_dets is None:
                continue
            transformer = (
                transformers.get(fi) if metric and transformers else None
            )
            int_carrier, int_kind = _active_carrier(
                int_dets, transformer=transformer, config=config
            )
            int_kind = int_kind or "reception"
            if (
                int_carrier is None
                or int_dets.tracker_id is None
                or int(int_dets.tracker_id[int_carrier.index]) != interceptor_tid
            ):
                continue
            if _touch_valid_or_redirect(
                int_dets,
                int_carrier,
                touch_kind=int_kind,
                frames_by_idx=frames_by_idx,
                frame_idx=fi,
                config=config,
                transformers=transformers or {},
                metric=metric,
                fps=fps,
                release_ball=release_carrier.ball,
                release_frame=release_frame,
            ):
                chosen = (fi, int_dets, int_carrier, int_kind)
                break
        if chosen is None:
            return False
        interception_frame = chosen[0]
        gap = interception_frame - release_frame
        if gap < config.min_turnover_gap_frames or gap > config.max_pass_gap_frames:
            return False

    if _recent_turnover_duplicate(
        turnovers,
        passer_tid,
        interceptor_tid,
        release_frame,
        window=config.dedupe_window_frames,
    ):
        return False

    passer_team = 1 - interceptor_team
    turnovers[:] = [
        t
        for t in turnovers
        if not (
            t.passer_tid == passer_tid
            and t.passer_team == passer_team
            and t.interception_frame < interception_frame
        )
    ]

    turnovers.append(
        InferredTurnover(
            release_frame=release_frame,
            interception_frame=interception_frame,
            passer_tid=passer_tid,
            passer_team=1 - interceptor_team,
            interceptor_tid=interceptor_tid,
            interceptor_team=interceptor_team,
            gap_frames=gap,
        )
    )
    if frames_by_idx is not None and passes is not None:
        _invalidate_passes_for_turnover(
            passes,
            turnovers[-1],
            frames_by_idx,
            config=config,
            transformers=transformers or {},
            metric=metric,
            fps=fps,
        )
    losing_state.release = None
    losing_state.release_is_one_touch = False
    losing_state.in_flight = False
    losing_state.arrival_candidate_tid = -1
    losing_state.arrival_streak = 0
    losing_state.arrival_control_streak = 0
    losing_state.turnover_snapshot = None
    return True


def _min_arrival_frames_for(
    *,
    touch_kind: str,
    gap_frames: int,
    config: PassDetectionConfig,
) -> int:
    """Reception-only paths on longer gaps need fewer consecutive touches."""
    if (
        touch_kind == "reception"
        and gap_frames >= config.adjacent_pass_max_gap_frames
    ):
        return config.min_reception_arrival_frames
    return config.min_arrival_frames


def _arrival_ready_for_pass(
    *,
    arrival_streak: int,
    arrival_control_streak: int,
    touch_kind: str,
    gap_frames: int,
    config: PassDetectionConfig,
    redirect_at_arrival: bool = False,
) -> bool:
    """Streak length + control-at-feet gate (stricter on quick adjacent plays)."""
    min_arrival = _min_arrival_frames_for(
        touch_kind=touch_kind, gap_frames=gap_frames, config=config
    )
    if (
        redirect_at_arrival
        and gap_frames >= config.adjacent_pass_max_gap_frames
    ):
        min_arrival = 1
    if arrival_streak < min_arrival:
        return False
    adjacent = gap_frames < config.adjacent_pass_max_gap_frames
    if adjacent:
        return arrival_control_streak >= config.min_control_frames
    if touch_kind == "reception":
        return True
    return arrival_control_streak >= config.min_arrival_control_frames


def _try_emit_intermediate_hop(
    passes: list[InferredPass],
    state: _TeamPossessionState,
    *,
    frame_idx: int,
    dets: sv.Detections,
    carrier: Carrier,
    tid: int,
    touch_kind: str,
    config: PassDetectionConfig,
    metric: bool,
    transformers: dict[int, object],
) -> bool:
    """Emit a short teammate relay when a new carrier settles during an in-flight pass."""
    release = state.release
    if release is None or touch_kind != "control":
        return False
    release_frame, _release_dets, release_carrier, release_tid = release
    if release_tid == tid:
        return False
    if (
        state.arrival_candidate_tid >= 0
        and state.arrival_candidate_tid != tid
        and state.arrival_streak > 0
    ):
        return False
    return _try_emit_pass(
        passes,
        release_frame=release_frame,
        release_carrier=release_carrier,
        passer_tid=release_tid,
        receiver_tid=tid,
        arrival_frame=frame_idx,
        arrival_carrier=carrier,
        touch_kind=touch_kind,
        config=config,
        metric=metric,
        transformers=transformers,
    )


def _on_confirmed_possession(
    state: _TeamPossessionState,
    frame_idx: int,
    dets: sv.Detections,
    carrier: Carrier,
    tid: int,
) -> None:
    """Credit sustained possession on this team."""
    if state.release is not None and state.release[3] != tid:
        return
    _confirm_release(state, frame_idx, dets, carrier, tid)


def _losing_team_pass_still_pending(state: _TeamPossessionState) -> bool:
    """True while an in-flight teammate arrival is still building."""
    return (
        state.in_flight
        and state.arrival_candidate_tid >= 0
        and state.arrival_streak > 0
    )


def _same_team_pass_arrival_in_progress(
    possessing_state: _TeamPossessionState,
    receiver_tid: int,
) -> bool:
    """True when ``receiver_tid`` is receiving from a teammate's active release."""
    release = possessing_state.release
    if release is None:
        return False
    if release[3] == receiver_tid:
        return False
    return (
        possessing_state.arrival_candidate_tid == receiver_tid
        and possessing_state.arrival_streak > 0
    )


def _should_credit_team_possession(
    *,
    other_state: _TeamPossessionState,
    touch_kind: str,
    redirect_touch: bool,
    dets: sv.Detections,
    carrier: Carrier,
    frames_by_idx: dict[int, sv.Detections],
    frame_idx: int,
    config: PassDetectionConfig,
    transformers: dict[int, object],
    metric: bool,
    fps: float,
) -> bool:
    """Whether a touch updates last-possession during the opponent's pending release.

    Fly-by receptions while the other team still owns a release must not anchor
    turnover snapshots or mint a new passer — only redirects or real control at
    the feet count. Key off ``release`` (not ``in_flight``): in_flight is only
    set on frames where *this* team has no valid carrier, so it stays False while
    an opponent skim is the active carrier.
    """
    if other_state.release is None:
        return True
    release_frame, _, release_carrier, _ = other_state.release
    if frame_idx <= release_frame:
        return True
    if redirect_touch:
        return True
    if touch_kind != "control":
        return False
    return _touch_valid_or_redirect(
        dets,
        carrier,
        touch_kind=touch_kind,
        frames_by_idx=frames_by_idx,
        frame_idx=frame_idx,
        config=config,
        transformers=transformers,
        metric=metric,
        fps=fps,
        release_ball=release_carrier.ball,
        release_frame=release_frame,
    )


def _teammate_recovered_after_intercept(
    frames_by_idx: dict[int, sv.Detections],
    *,
    passer_team: int,
    passer_tid: int,
    intercept_frame: int,
    config: PassDetectionConfig,
    transformers: dict[int, object],
    metric: bool,
    fps: float,
    window_frames: int = 15,
) -> bool:
    """True when a teammate regains the ball soon after a contested touch."""
    touch_cfg = config.touch_validation_config()
    end_frame = intercept_frame + window_frames
    for frame_idx in range(intercept_frame + 1, end_frame):
        dets = frames_by_idx.get(frame_idx)
        if dets is None:
            continue
        transformer = transformers.get(frame_idx) if metric else None
        carrier, touch_kind = _active_carrier(
            dets, transformer=transformer, config=config
        )
        touch_kind = touch_kind or "reception"
        if carrier is None or int(carrier.team) != passer_team:
            continue
        if dets.tracker_id is None:
            continue
        tid = int(dets.tracker_id[carrier.index])
        if tid == passer_tid:
            continue
        if not is_valid_possession_touch(
            dets,
            carrier,
            touch_kind=touch_kind,
            config=touch_cfg,
            **_touch_validation_kwargs(
                frames_by_idx,
                frame_idx,
                transformers=transformers,
                metric=metric,
                fps=fps,
                max_lookback=config.ball_speed_lookback_frames,
                min_lookback=config.ball_speed_min_lookback_frames,
            ),
        ):
            continue
        return True
    return False


def _try_emit_snapshot_turnover(
    losing_state: _TeamPossessionState,
    *,
    interceptor_tid: int,
    interceptor_team: int,
    interception_frame: int,
    turnovers: list[InferredTurnover],
    passes: list[InferredPass] | None,
    frames_by_idx: dict[int, sv.Detections],
    config: PassDetectionConfig,
    transformers: dict[int, object],
    metric: bool,
    fps: float,
) -> bool:
    """Emit a deferred turnover once the interceptor has secured possession."""
    snapshot = losing_state.turnover_snapshot
    if snapshot is None or _losing_team_pass_still_pending(losing_state):
        return False
    losing_release_frame, losing_passer_tid = snapshot
    intercept_frame = _first_valid_touch_frame(
        frames_by_idx,
        player_tid=interceptor_tid,
        start_frame=losing_release_frame,
        end_frame=interception_frame,
        config=config,
        transformers=transformers,
        metric=metric,
        fps=fps,
    )
    if intercept_frame is None:
        intercept_frame = interception_frame
    if _teammate_recovered_after_intercept(
        frames_by_idx,
        passer_team=1 - interceptor_team,
        passer_tid=losing_passer_tid,
        intercept_frame=intercept_frame,
        config=config,
        transformers=transformers,
        metric=metric,
        fps=fps,
    ):
        return False
    control_end = max(intercept_frame, interception_frame) + 1
    if not _opponent_control_between(
        frames_by_idx,
        start_frame=losing_release_frame,
        end_frame=control_end,
        passer_team=1 - interceptor_team,
        config=config,
        transformers=transformers,
        metric=metric,
        fps=fps,
    ):
        return False
    return _try_emit_turnover(
        losing_state,
        interceptor_tid=interceptor_tid,
        interceptor_team=interceptor_team,
        interception_frame=intercept_frame,
        turnovers=turnovers,
        config=config,
        release_frame=losing_release_frame,
        passer_tid=losing_passer_tid,
        frames_by_idx=frames_by_idx,
        transformers=transformers,
        metric=metric,
        fps=fps,
        passes=passes,
    )


def _queue_turnover_emit(
    pending: list[_PendingTurnoverEmit],
    *,
    losing_team: int,
    losing_state: _TeamPossessionState,
    possessing_state: _TeamPossessionState,
    interceptor_tid: int,
    interceptor_team: int,
    interception_frame: int,
    frames_by_idx: dict[int, sv.Detections],
    config: PassDetectionConfig,
    transformers: dict[int, object],
    metric: bool,
    fps: float,
) -> None:
    snapshot = losing_state.turnover_snapshot
    if snapshot is None or _losing_team_pass_still_pending(losing_state):
        return
    if _same_team_pass_arrival_in_progress(possessing_state, interceptor_tid):
        return
    release_frame, passer_tid = snapshot
    if interception_frame - release_frame < config.adjacent_pass_max_gap_frames:
        return
    release_ball: np.ndarray | None = None
    release_dets = frames_by_idx.get(release_frame)
    if release_dets is not None:
        rel_transformer = transformers.get(release_frame) if metric else None
        rel_carrier, _ = _active_carrier(
            release_dets, transformer=rel_transformer, config=config
        )
        if (
            rel_carrier is not None
            and release_dets.tracker_id is not None
            and int(release_dets.tracker_id[rel_carrier.index]) == passer_tid
        ):
            release_ball = rel_carrier.ball
    first_opp = _first_opponent_touch_in_window(
        frames_by_idx,
        start_frame=release_frame,
        end_frame=interception_frame + 1,
        passer_team=losing_team,
        config=config,
        transformers=transformers,
        metric=metric,
        require_control=True,
        player_tid=interceptor_tid,
        fps=fps,
        release_frame=release_frame,
        release_ball=release_ball,
    )
    if first_opp is None:
        return
    pending.append(
        _PendingTurnoverEmit(
            losing_team=losing_team,
            release_frame=release_frame,
            passer_tid=passer_tid,
            interceptor_tid=interceptor_tid,
            interceptor_team=interceptor_team,
            interception_frame=interception_frame,
            emit_after_frame=interception_frame
            + config.turnover_recovery_window_frames,
        )
    )


def _flush_pending_turnovers(
    pending: list[_PendingTurnoverEmit],
    *,
    frame_idx: int,
    team_states: dict[int, _TeamPossessionState],
    turnovers: list[InferredTurnover],
    passes: list[InferredPass],
    frames_by_idx: dict[int, sv.Detections],
    config: PassDetectionConfig,
    transformers: dict[int, object],
    metric: bool,
    fps: float,
) -> None:
    """Finalize queued turnovers once the recovery window has elapsed."""
    remaining: list[_PendingTurnoverEmit] = []
    for item in pending:
        if frame_idx < item.emit_after_frame:
            remaining.append(item)
            continue
        losing_state = team_states[item.losing_team]
        losing_state.turnover_snapshot = (item.release_frame, item.passer_tid)
        losing_state.arrival_candidate_tid = -1
        losing_state.arrival_streak = 0
        losing_state.arrival_control_streak = 0
        _try_emit_snapshot_turnover(
            losing_state,
            interceptor_tid=item.interceptor_tid,
            interceptor_team=item.interceptor_team,
            interception_frame=item.interception_frame,
            turnovers=turnovers,
            passes=passes,
            frames_by_idx=frames_by_idx,
            config=config,
            transformers=transformers,
            metric=metric,
            fps=fps,
        )
    pending[:] = remaining


def _stale_anchor_during_opponent_attack(
    state: _TeamPossessionState,
    other_state: _TeamPossessionState | None,
    frame_idx: int,
    *,
    release_frame: int,
) -> bool:
    """True when our in-flight anchor predates the opponent's current possession."""
    if other_state is None or other_state.release is None:
        return False
    if release_frame >= other_state.release[0]:
        return False
    return frame_idx > other_state.release[0]


def _first_valid_touch_frame(
    frames_by_idx: dict[int, sv.Detections],
    *,
    player_tid: int,
    start_frame: int,
    end_frame: int,
    config: PassDetectionConfig,
    transformers: dict[int, object],
    metric: bool,
    fps: float = 25.0,
) -> int | None:
    """First frame in ``(start_frame, end_frame)`` where ``player_tid`` has a valid touch."""
    touch_cfg = config.touch_validation_config()
    for frame_idx in range(start_frame + 1, end_frame):
        dets = frames_by_idx.get(frame_idx)
        if dets is None:
            continue
        transformer = transformers.get(frame_idx) if metric else None
        carrier, touch_kind = _active_carrier(
            dets, transformer=transformer, config=config
        )
        touch_kind = touch_kind or "reception"
        if carrier is None or not is_valid_possession_touch(
            dets,
            carrier,
            touch_kind=touch_kind,
            config=touch_cfg,
            **_touch_validation_kwargs(
                frames_by_idx,
                frame_idx,
                transformers=transformers,
                metric=metric,
                fps=fps,
                max_lookback=config.ball_speed_lookback_frames,
                min_lookback=config.ball_speed_min_lookback_frames,
            ),
        ):
            continue
        if dets.tracker_id is None:
            continue
        if int(dets.tracker_id[carrier.index]) == player_tid:
            return frame_idx
    return None


def scan_possession_events(
    detections_iter: DetectionIterator,
    *,
    config: PassDetectionConfig = PassDetectionConfig(),
    metric: bool = True,
    transformers: dict[int, object] | None = None,
    fps: float = 25.0,
) -> PossessionScanResult:
    """Scan tracked detections for passes (rule 4) and turnovers (rule 5)."""
    transformers = transformers or {}
    frames = list(detections_iter)
    frames_by_idx: dict[int, sv.Detections] = {
        frame_idx: dets for frame_idx, dets in frames
    }
    passes: list[InferredPass] = []
    turnovers: list[InferredTurnover] = []
    team_states: dict[int, _TeamPossessionState] = {
        0: _TeamPossessionState(),
        1: _TeamPossessionState(),
    }
    missing_ball_streak = 0
    pending_turnovers: list[_PendingTurnoverEmit] = []
    pending_redirect_turnovers: list[_PendingRedirectTurnover] = []
    last_good_ball: np.ndarray | None = None
    last_good_ball_frame: int | None = None

    def _expire_stale_releases(frame_idx: int) -> None:
        for state in team_states.values():
            if state.release is None or state.in_flight:
                continue
            anchor = _possession_staleness_anchor(state)
            if anchor is None:
                continue
            if frame_idx - anchor > config.max_pass_gap_frames:
                state.release = None
                state.arrival_candidate_tid = -1
                state.arrival_streak = 0
                state.arrival_control_streak = 0

    def _update_control_streak(state: _TeamPossessionState, tid: int) -> None:
        if tid == state.possession_tid:
            state.control_streak += 1
        else:
            state.possession_tid = tid
            state.control_streak = 1

    for frame_idx, dets in frames:
        _flush_pending_turnovers(
            pending_turnovers,
            frame_idx=frame_idx,
            team_states=team_states,
            turnovers=turnovers,
            passes=passes,
            frames_by_idx=frames_by_idx,
            config=config,
            transformers=transformers,
            metric=metric,
            fps=fps,
        )
        _flush_pending_redirect_turnovers(
            pending_redirect_turnovers,
            frame_idx=frame_idx,
            team_states=team_states,
            turnovers=turnovers,
            passes=passes,
            frames_by_idx=frames_by_idx,
            config=config,
            transformers=transformers,
            metric=metric,
            fps=fps,
        )
        _expire_stale_releases(frame_idx)
        ball = ball_xy(dets)
        ball_is_teleport = False
        if ball is not None and last_good_ball is not None and last_good_ball_frame is not None:
            gap = frame_idx - last_good_ball_frame
            if 0 < gap <= config.missing_ball_tolerance + 1:
                jump = float(
                    np.hypot(ball[0] - last_good_ball[0], ball[1] - last_good_ball[1])
                )
                if jump > config.max_ball_teleport_px_per_frame * gap:
                    ball_is_teleport = True
        if ball is None:
            missing_ball_streak += 1
        else:
            missing_ball_streak = 0
            if not ball_is_teleport:
                last_good_ball = ball
                last_good_ball_frame = frame_idx

        transformer = transformers.get(frame_idx) if metric else None
        carrier, touch_kind = _active_carrier(
            dets, transformer=transformer, config=config
        )
        if ball_is_teleport:
            # One-frame ball detection outlier: distrust possession this frame.
            carrier = None

        touch_kind = touch_kind or "reception"
        release_for_speed: np.ndarray | None = None
        release_frame_for_speed: int | None = None
        if carrier is not None:
            carrier_team = int(carrier.team)
            for poss_state in team_states.values():
                if poss_state.release is None:
                    continue
                rel_f, _, rel_c, _ = poss_state.release
                if int(rel_c.team) == carrier_team:
                    continue
                release_for_speed = rel_c.ball
                release_frame_for_speed = rel_f
                break
        if carrier is None or not _touch_valid_or_redirect(
            dets,
            carrier,
            touch_kind=touch_kind,
            frames_by_idx=frames_by_idx,
            frame_idx=frame_idx,
            config=config,
            transformers=transformers,
            metric=metric,
            fps=fps,
            release_ball=release_for_speed,
            release_frame=release_frame_for_speed,
        ):
            if ball is not None:
                for team_id, state in team_states.items():
                    if state.release is None:
                        _promote_in_flight_release(
                            state,
                            frame_idx,
                            config=config,
                            team=team_id,
                            other_state=team_states[1 - team_id],
                            frames_by_idx=frames_by_idx,
                            transformers=transformers,
                            metric=metric,
                            fps=fps,
                        )
                    state.in_flight = True
                    if state.release is None:
                        state.arrival_candidate_tid = -1
                        state.arrival_streak = 0
                        state.arrival_control_streak = 0
            elif missing_ball_streak <= config.missing_ball_tolerance:
                _bridge_missing_ball(
                    team_states,
                    frame_idx=frame_idx,
                    config=config,
                    frames_by_idx=frames_by_idx,
                    transformers=transformers,
                    metric=metric,
                    fps=fps,
                )
            else:
                _end_missing_ball_bridge(team_states)
            continue

        tid = int(dets.tracker_id[carrier.index]) if dets.tracker_id is not None else -1
        team = int(carrier.team)
        if tid < 0 or team not in (0, 1):
            continue

        state = team_states[team]
        other_state = team_states[1 - team]
        redirect_touch = _ball_redirected(
            frames_by_idx, frame_idx, config=config
        )
        credit_possession = _should_credit_team_possession(
            other_state=other_state,
            touch_kind=touch_kind,
            redirect_touch=redirect_touch,
            dets=dets,
            carrier=carrier,
            frames_by_idx=frames_by_idx,
            frame_idx=frame_idx,
            config=config,
            transformers=transformers,
            metric=metric,
            fps=fps,
        )
        if credit_possession:
            state.last_possession_frame = frame_idx
            state.last_possessor_tid = tid
        if redirect_touch:
            _promote_redirect_one_touch_release(
                state, frame_idx, dets, carrier, tid
            )
        losing_origin_frame = other_state.last_possession_frame
        losing_origin_tid = other_state.last_possessor_tid
        if losing_origin_frame < 0 and other_state.release is not None:
            losing_origin_frame = other_state.release[0]
            losing_origin_tid = other_state.release[3]
        if (
            credit_possession
            and not _same_team_pass_arrival_in_progress(state, tid)
            and losing_origin_frame >= 0
            and losing_origin_tid >= 0
            and frame_idx > losing_origin_frame
        ):
            other_state.turnover_snapshot = (losing_origin_frame, losing_origin_tid)
            if redirect_touch:
                _queue_redirect_turnover(
                    pending_redirect_turnovers,
                    losing_team=1 - team,
                    interceptor_tid=tid,
                    interceptor_team=team,
                    redirect_frame=frame_idx,
                    config=config,
                )

        # Fly-by during the opponent's in-flight release must not mint a new
        # passer via last_touch / control streak / confirm (false #6→#3).
        # Still fall through so an in-progress teammate arrival can complete.
        if credit_possession:
            state.in_flight = False
            min_control = _min_control_frames_for(dets, carrier, config=config)

            if touch_kind == "control":
                state.reception_streak = 0
                _update_control_streak(state, tid)
                skip_last_touch_update = False
                if state.last_touch is not None:
                    prev_frame, _, _, prev_tid = state.last_touch
                    if (
                        prev_tid != tid
                        and state.control_streak < min_control
                        and frame_idx - prev_frame
                        <= config.one_touch_release_window_frames
                    ):
                        skip_last_touch_update = True
                if not skip_last_touch_update:
                    state.last_touch = (frame_idx, dets, carrier, tid)
                if state.control_streak >= min_control:
                    if (
                        state.control_streak == min_control
                        and state.release is not None
                        and state.release[3] != tid
                    ):
                        _try_emit_intermediate_hop(
                            passes,
                            state,
                            frame_idx=frame_idx,
                            dets=dets,
                            carrier=carrier,
                            tid=tid,
                            touch_kind=touch_kind,
                            config=config,
                            metric=metric,
                            transformers=transformers,
                        )
                        _confirm_release(state, frame_idx, dets, carrier, tid)
                        state.arrival_candidate_tid = -1
                        state.arrival_streak = 0
                        state.arrival_control_streak = 0
                    else:
                        _on_confirmed_possession(
                            state, frame_idx, dets, carrier, tid
                        )
                    if state.control_streak == min_control:
                        _queue_turnover_emit(
                            pending_turnovers,
                            losing_team=1 - team,
                            losing_state=other_state,
                            possessing_state=state,
                            interceptor_tid=tid,
                            interceptor_team=team,
                            interception_frame=frame_idx,
                            frames_by_idx=frames_by_idx,
                            config=config,
                            transformers=transformers,
                            metric=metric,
                            fps=fps,
                        )
            else:
                if tid == state.possession_tid:
                    state.reception_streak += 1
                else:
                    state.possession_tid = tid
                    state.reception_streak = 1
                state.control_streak = 0
                state.last_touch = (frame_idx, dets, carrier, tid)
                if _is_goalkeeper(dets, carrier.index):
                    _on_confirmed_possession(state, frame_idx, dets, carrier, tid)
        else:
            state.control_streak = 0

        release = state.release
        if release is None:
            continue

        release_frame, release_dets, release_carrier, release_tid = release

        if tid == release_tid:
            if touch_kind == "control":
                state.release = (frame_idx, dets, carrier, tid)
                state.release_is_one_touch = False
            elif not state.release_is_one_touch:
                state.release = (frame_idx, dets, carrier, tid)
            state.arrival_candidate_tid = -1
            state.arrival_streak = 0
            state.arrival_control_streak = 0
            continue

        if tid == state.arrival_candidate_tid:
            state.arrival_streak += 1
            if touch_kind == "control":
                state.arrival_control_streak += 1
            else:
                state.arrival_control_streak = 0
        else:
            state.arrival_candidate_tid = tid
            state.arrival_streak = 1
            state.arrival_control_streak = 1 if touch_kind == "control" else 0

        gap_frames = frame_idx - release_frame
        redirect_at_arrival = _ball_redirected(
            frames_by_idx, frame_idx, config=config
        ) and not _ball_dropout_exceeds_between(
            frames_by_idx,
            start_frame=release_frame,
            end_frame=frame_idx,
            max_dropout=config.missing_ball_tolerance,
        )
        arrival_ready = _arrival_ready_for_pass(
            arrival_streak=state.arrival_streak,
            arrival_control_streak=state.arrival_control_streak,
            touch_kind=touch_kind,
            gap_frames=gap_frames,
            config=config,
            redirect_at_arrival=redirect_at_arrival,
        )

        if not arrival_ready:
            continue

        opponent_blocked = _opponent_blocks_between(
            frames_by_idx,
            start_frame=release_frame,
            end_frame=frame_idx,
            passer_team=team,
            gap_frames=gap_frames,
            config=config,
            transformers=transformers,
            metric=metric,
            fps=fps,
            release_frame=release_frame,
            release_ball=release_carrier.ball,
        )
        if opponent_blocked:
            turnover_emitted = False
            first_opp = _first_opponent_touch_in_window(
                frames_by_idx,
                start_frame=release_frame,
                end_frame=frame_idx,
                passer_team=team,
                config=config,
                transformers=transformers,
                metric=metric,
                require_control=True,
                fps=fps,
                release_frame=release_frame,
                release_ball=release_carrier.ball,
            )
            if first_opp is not None:
                opp_frame, opp_tid, _ = first_opp
                secured = _interceptor_secured_possession_frame(
                    frames_by_idx,
                    start_frame=opp_frame,
                    end_frame=frame_idx,
                    interceptor_tid=opp_tid,
                    config=config,
                    transformers=transformers,
                    metric=metric,
                    fps=fps,
                )
                if secured is not None:
                    turnover_emitted = _try_emit_turnover(
                        state,
                        interceptor_tid=opp_tid,
                        interceptor_team=1 - team,
                        interception_frame=secured,
                        turnovers=turnovers,
                        config=config,
                        release_frame=release_frame,
                        passer_tid=release_tid,
                        frames_by_idx=frames_by_idx,
                        transformers=transformers,
                        metric=metric,
                        fps=fps,
                        skip_interceptor_control_check=True,
                        passes=passes,
                    )
                elif (
                    _opponent_active_control_frames(
                        frames_by_idx,
                        start_frame=release_frame,
                        end_frame=frame_idx,
                        passer_team=team,
                        config=config,
                        transformers=transformers,
                        metric=metric,
                        fps=fps,
                        release_frame=release_frame,
                        release_ball=release_carrier.ball,
                    )
                    >= config.min_long_gap_opponent_control_streak
                ):
                    turnover_emitted = _try_emit_turnover(
                        state,
                        interceptor_tid=opp_tid,
                        interceptor_team=1 - team,
                        interception_frame=opp_frame,
                        turnovers=turnovers,
                        config=config,
                        release_frame=release_frame,
                        passer_tid=release_tid,
                        frames_by_idx=frames_by_idx,
                        transformers=transformers,
                        metric=metric,
                        fps=fps,
                        skip_interceptor_control_check=True,
                        passes=passes,
                    )
            if _stale_anchor_during_opponent_attack(
                state, other_state, frame_idx, release_frame=release_frame
            ):
                # Fly-by near a teammate while the opponent is on the ball.
                state.arrival_candidate_tid = -1
                state.arrival_streak = 0
                state.arrival_control_streak = 0
                continue
            if not turnover_emitted and tid == state.arrival_candidate_tid and _try_emit_pass(
                passes,
                release_frame=release_frame,
                release_carrier=release_carrier,
                passer_tid=release_tid,
                receiver_tid=tid,
                arrival_frame=frame_idx,
                arrival_carrier=carrier,
                touch_kind=touch_kind,
                config=config,
                metric=metric,
                transformers=transformers,
            ):
                _try_emit_snapshot_turnover(
                    other_state,
                    interceptor_tid=release_tid,
                    interceptor_team=team,
                    interception_frame=frame_idx,
                    turnovers=turnovers,
                    passes=passes,
                    frames_by_idx=frames_by_idx,
                    config=config,
                    transformers=transformers,
                    metric=metric,
                    fps=fps,
                )
            _confirm_release(state, frame_idx, dets, carrier, tid)
            state.arrival_candidate_tid = -1
            state.arrival_streak = 0
            state.arrival_control_streak = 0
            continue

        if _try_emit_pass(
            passes,
            release_frame=release_frame,
            release_carrier=release_carrier,
            passer_tid=release_tid,
            receiver_tid=tid,
            arrival_frame=frame_idx,
            arrival_carrier=carrier,
            touch_kind=touch_kind,
            config=config,
            metric=metric,
            transformers=transformers,
        ):
            _try_emit_snapshot_turnover(
                other_state,
                interceptor_tid=release_tid,
                interceptor_team=team,
                interception_frame=frame_idx,
                turnovers=turnovers,
                passes=passes,
                frames_by_idx=frames_by_idx,
                config=config,
                transformers=transformers,
                metric=metric,
                fps=fps,
            )
        _confirm_release(state, frame_idx, dets, carrier, tid)
        if not _opponent_control_between(
            frames_by_idx,
            start_frame=release_frame,
            end_frame=frame_idx,
            passer_team=team,
            config=config,
            transformers=transformers,
            metric=metric,
        ):
            state.turnover_snapshot = None
        state.arrival_candidate_tid = -1
        state.arrival_streak = 0
        state.arrival_control_streak = 0

    _flush_pending_turnovers(
        pending_turnovers,
        frame_idx=max(frames_by_idx) + config.turnover_recovery_window_frames
        if frames_by_idx
        else config.turnover_recovery_window_frames,
        team_states=team_states,
        turnovers=turnovers,
        passes=passes,
        frames_by_idx=frames_by_idx,
        config=config,
        transformers=transformers,
        metric=metric,
        fps=fps,
    )
    _flush_pending_redirect_turnovers(
        pending_redirect_turnovers,
        frame_idx=max(frames_by_idx) + min(config.max_pass_gap_frames, 60)
        if frames_by_idx
        else min(config.max_pass_gap_frames, 60),
        team_states=team_states,
        turnovers=turnovers,
        passes=passes,
        frames_by_idx=frames_by_idx,
        config=config,
        transformers=transformers,
        metric=metric,
        fps=fps,
    )

    return PossessionScanResult(
        passes=tuple(_filter_overlapping_cross_team_passes(passes)),
        turnovers=tuple(turnovers),
    )
