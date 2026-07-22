"""Freeze-moment planning for PASS_ALTERNATIVES."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import supervision as sv

from sports.common.pass_options import PassOption, PassQualityScorer, PassWeights
from sports.common.pass_pitch import lane_scoring_transformer_for_frame
from sports.common.possession import (
    BallPositionHistory,
    Carrier,
    ball_xy,
    find_control_carrier,
)

__all__ = [
    "PassEvent",
    "plan_pass_events",
    "DEFAULT_SLOWDOWN_RAMP_SECONDS",
    "DEFAULT_OPTION_REVEAL_SECONDS",
    "DEFAULT_FREEZE_SECONDS",
    "DEFAULT_FINAL_OPTION_EXTRA_SECONDS",
    "MAX_RAMP_HOLD",
    "slowdown_hold_count",
]


@dataclass(frozen=True)
class PassEvent:
    frame_idx: int
    carrier: Carrier
    options: list[PassOption]
    top_score: float


MAX_RAMP_HOLD = 6
DEFAULT_SLOWDOWN_RAMP_SECONDS = 0.72
DEFAULT_OPTION_REVEAL_SECONDS = 0.6
DEFAULT_FREEZE_SECONDS = 2.5
DEFAULT_FINAL_OPTION_EXTRA_SECONDS = 1.0


def _freeze_moment_score(
    pass_top: float,
    carrier: Carrier,
    *,
    ball_speed: float | None,
    weights: PassWeights,
    metric: bool,
) -> float | None:
    if weights.use_ball_control_gate and ball_speed is not None:
        if metric:
            if ball_speed >= weights.ball_speed_skip_m:
                return None
            ref, cap = weights.ball_speed_ref_m, weights.ball_speed_max_m
            tight_ref = weights.carrier_tight_ref_m
        else:
            if ball_speed >= weights.ball_speed_skip_px_s:
                return None
            ref, cap = weights.ball_speed_ref_px_s, weights.ball_speed_max_px_s
            tight_ref = weights.carrier_tight_ref_px

        score = pass_top
        if ball_speed <= ref:
            score += 0.05
        elif ball_speed < cap:
            t = (ball_speed - ref) / (cap - ref)
            score -= weights.ball_speed_penalty * t
        else:
            score -= weights.ball_speed_penalty

        if carrier.distance < tight_ref:
            score += weights.carrier_tight_bonus * (
                1.0 - carrier.distance / tight_ref
            )
        return score

    score = pass_top
    tight_ref = (
        weights.carrier_tight_ref_m if metric else weights.carrier_tight_ref_px
    )
    if carrier.distance < tight_ref:
        score += weights.carrier_tight_bonus * (1.0 - carrier.distance / tight_ref)
    return score


def _resolve_freeze_frame_earlier(
    event: PassEvent,
    by_frame: dict[int, PassEvent],
    *,
    weights: PassWeights,
    metric: bool,
    instant_speed_by_frame: dict[int, float],
) -> PassEvent:
    if not weights.freeze_nudge_earlier:
        return event
    slack = weights.freeze_nudge_score_slack
    eps = (
        weights.freeze_separation_eps_m
        if metric
        else weights.freeze_separation_eps_px
    )
    best = event

    while True:
        prev = by_frame.get(best.frame_idx - 1)
        if prev is None or prev.top_score < best.top_score - slack:
            break
        separating = best.carrier.distance > prev.carrier.distance + eps
        instant = instant_speed_by_frame.get(best.frame_idx)
        fast_ball = instant is not None and (
            (metric and instant >= weights.freeze_release_ball_speed_skip_m)
            or (not metric and instant >= weights.freeze_release_ball_speed_skip_px_s)
        )
        if not (separating or fast_ball):
            break
        best = prev

    prev = by_frame.get(best.frame_idx - 1)
    if (
        prev is not None
        and best.top_score > prev.top_score
        and prev.top_score >= best.top_score - slack
    ):
        best = prev

    return best


def _apply_freeze_frame_nudges(
    candidates: list[PassEvent],
    *,
    weights: PassWeights,
    metric: bool,
    instant_speed_by_frame: dict[int, float],
) -> list[PassEvent]:
    by_frame = {e.frame_idx: e for e in candidates}
    resolved: dict[int, PassEvent] = {}
    for event in candidates:
        nudged = _resolve_freeze_frame_earlier(
            event,
            by_frame,
            weights=weights,
            metric=metric,
            instant_speed_by_frame=instant_speed_by_frame,
        )
        prev = resolved.get(nudged.frame_idx)
        if prev is None or nudged.top_score > prev.top_score:
            resolved[nudged.frame_idx] = nudged
    return sorted(resolved.values(), key=lambda e: e.frame_idx)


def _select_pass_moments(
    candidates: list[PassEvent],
    *,
    weights: PassWeights,
    min_gap_frames: int,
    max_events: int | None,
) -> list[PassEvent]:
    if not candidates:
        return []

    by_frame = sorted(candidates, key=lambda e: e.frame_idx)
    score_at = {e.frame_idx: e.top_score for e in by_frame}
    half = weights.freeze_local_peak_half_window

    peaks: list[PassEvent] = []
    for event in by_frame:
        if event.top_score < weights.freeze_min_pick_score:
            continue
        if event.options[0].score < weights.freeze_min_pass_score:
            continue
        if weights.freeze_detect_local_peaks:
            f = event.frame_idx
            neighbor_scores = [
                score_at.get(f + d, -1.0)
                for d in range(-half, half + 1)
                if d != 0 and (f + d) in score_at
            ]
            if neighbor_scores and event.top_score <= max(neighbor_scores):
                continue
        peaks.append(event)

    peaks.sort(key=lambda e: e.top_score, reverse=True)
    chosen: list[PassEvent] = []
    for event in peaks:
        if all(abs(event.frame_idx - c.frame_idx) >= min_gap_frames for c in chosen):
            chosen.append(event)
        if max_events is not None and len(chosen) >= max_events:
            break
    chosen.sort(key=lambda e: e.frame_idx)
    return chosen


def slowdown_hold_count(
    frames_until_event: int,
    *,
    ramp_frames: int,
    max_extra_holds: int = MAX_RAMP_HOLD,
) -> int:
    """Repeat count for live frames as playback eases into a freeze."""
    if frames_until_event <= 0 or frames_until_event > ramp_frames:
        return 1
    t = 1.0 - frames_until_event / ramp_frames
    return 1 + int((t * t) * max(0, max_extra_holds - 1))


def plan_pass_events(
    pass_frames: list[tuple[int, sv.Detections]],
    *,
    fps: float,
    frame_transforms: dict[int, Any | None],
    keypoints_by_frame: dict[int, Any | None] | None = None,
    scorer: PassQualityScorer | None = None,
    weights: PassWeights | None = None,
    max_events: int | None = None,
    min_gap_frames: int = 90,
    pitch_confidence: float = 0.9,
    min_frame_idx: int = 30,
) -> list[PassEvent]:
    """Detect cinematic pass freeze moments from pass-ready frames.

    Lane scoring goes through :class:`PassQualityScorer` for ranked alternatives.
    """
    weights = weights or PassWeights.metric()
    keypoints_by_frame = keypoints_by_frame or {}
    if scorer is None:
        scorer = PassQualityScorer(
            weights=weights,
            transformers=frame_transforms,
            keypoints_by_frame=keypoints_by_frame,
            pitch_confidence=pitch_confidence,
        )
    max_px = weights.freeze_carrier_max_distance_px
    max_m = weights.freeze_carrier_max_distance_m
    require_both = weights.freeze_require_both_spaces

    ball_history = BallPositionHistory()
    candidates: list[PassEvent] = []
    instant_speed_by_frame: dict[int, float] = {}

    for frame_idx, dets in pass_frames:
        kps = keypoints_by_frame.get(frame_idx)
        transformer = lane_scoring_transformer_for_frame(
            frame_transforms, frame_idx, kps, pitch_confidence=pitch_confidence
        )
        ball_history.record(frame_idx, ball_xy(dets))

        carrier = find_control_carrier(
            dets,
            max_distance_px=max_px,
            transformer=transformer,
            max_distance_m=max_m,
            require_both_spaces=require_both and transformer is not None,
        )
        if carrier is None or frame_idx < min_frame_idx:
            continue
        options = scorer.top_options(frame_idx, dets, carrier, k=3)
        if len(options) < 2:
            continue
        top = options[0]
        if top.length < weights.min_length or top.length > weights.max_length:
            continue
        ball_speed = ball_history.speed(
            frame_idx,
            lookback_frames=weights.ball_speed_lookback_frames,
            fps=fps,
            transformer=transformer,
        )
        pick_score = _freeze_moment_score(
            top.score, carrier, ball_speed=ball_speed, weights=weights, metric=True
        )
        if pick_score is None:
            continue
        instant = ball_history.speed(
            frame_idx, lookback_frames=1, fps=fps, transformer=transformer
        )
        if instant is not None:
            instant_speed_by_frame[frame_idx] = instant
        candidates.append(PassEvent(frame_idx, carrier, options, pick_score))

    candidates = _apply_freeze_frame_nudges(
        candidates,
        weights=weights,
        metric=True,
        instant_speed_by_frame=instant_speed_by_frame,
    )
    return _select_pass_moments(
        candidates,
        weights=weights,
        min_gap_frames=min_gap_frames,
        max_events=max_events,
    )
