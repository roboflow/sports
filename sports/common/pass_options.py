"""Score teammate passing lanes for a ball carrier.

Each candidate lane (carrier -> teammate) is scored on three football-sense axes:

* **openness** - nearest **rival** in the pass corridor (``lane_width``, strict; widens
  in fixed length tiers on long passes, not proportional to distance).
* **teammate lane** - optional light penalty if a teammate blocks the corridor (narrower
  width than rivals; they can let the ball through so we only ding obvious obstacles).
* **forward progress** - gain toward the attacking direction.
* **carrier motion** - passes behind the carrier's Kalman-predicted run direction are penalized.
* **backward attack** - passes toward own goal (negative forward gain) are penalized when
  the carrier is running forward (optional gate).
* **receiver space** - how much room the receiver has from the nearest opponent.

A short-range/very-long-range penalty keeps the suggestions realistic. Scores are
normalized to ``[0, 1]`` against tunable reference distances (in the same units as the
input coordinates: pixels for v1, meters for v2), then combined with weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import supervision as sv

from sports.common.geometry import (
    count_lane_blockers,
    count_lane_blockers_body,
    lane_segment_clearance,
    lane_segment_clearance_body,
    point_to_segment_distance,
    unit,
)
from sports.common.kinematics import feet_xy, player_mask
from sports.common.pass_pitch import attack_direction
from sports.common.possession import (
    CONTROL_MAX_DISTANCE_M,
    CONTROL_MAX_DISTANCE_PX,
    Carrier,
    bbox_center_xy,
)


@dataclass(frozen=True)
class PassWeights:
    openness: float = 0.45
    forward: float = 0.30
    space: float = 0.25
    # reference scales for normalization (pixels for v1)
    open_ref: float = 120.0
    space_ref: float = 120.0
    forward_ref: float = 600.0
    min_length: float = 60.0
    max_length: float = 1100.0
    use_lane_openness: bool = True
    lane_t_min: float = 0.0
    lane_t_max: float = 1.0
    lane_width: float | None = None  # base rival corridor width in m (pitch) or px (image)
    lane_width_mid_threshold_m: float = 18.0  # stepped boosts below (metric only)
    lane_width_long_threshold_m: float = 28.0
    lane_width_mid_boost_m: float = 0.75
    lane_width_long_boost_m: float = 1.5
    lane_width_max_m: float = 4.5
    lane_in_image_space: bool = False  # metric default: pitch/radar corridor (meters)
    lane_use_body_center: bool = True  # min(feet, bbox center) for rival lane distance
    teammate_lane_width: float | None = None  # None -> 0.5 * lane_width when lane_width set
    teammate_open_ref: float | None = None  # clearance scale for teammate penalty; None -> open_ref
    teammate_lane_penalty: float = 0.0  # max score deduction when a teammate blocks (0 = off)
    openness_rivals_only: bool = True  # openness field = rivals only; teammate uses penalty
    use_carrier_motion: bool = True
    motion_lookback_frames: int = 5
    motion_min_displacement_px: float = 5.0
    motion_min_displacement_m: float = 0.35
    backward_penalty: float = 0.22
    backward_cos_threshold: float = -0.15  # pass vs run; below => backward
    backward_attack_penalty: float = 0.18
    backward_attack_only_when_running_forward: bool = True
    backward_attack_motion_cos_threshold: float = 0.15  # run vs attack; above => gate on
    # Freeze-moment selection (plan_events): same in-control gate as pass detection
    freeze_carrier_max_distance_m: float = CONTROL_MAX_DISTANCE_M
    freeze_carrier_max_distance_px: float = CONTROL_MAX_DISTANCE_PX
    freeze_require_both_spaces: bool = False
    freeze_nudge_earlier: bool = True
    freeze_nudge_score_slack: float = 0.08
    freeze_separation_eps_m: float = 0.03
    freeze_separation_eps_px: float = 2.5
    freeze_release_ball_speed_skip_m: float = 2.5
    freeze_release_ball_speed_skip_px_s: float = 120.0
    use_ball_control_gate: bool = True
    ball_speed_lookback_frames: int = 4
    ball_speed_ref_m: float = 1.5
    ball_speed_max_m: float = 4.5
    ball_speed_skip_m: float = 8.0
    ball_speed_penalty: float = 0.40
    ball_speed_ref_px_s: float = 90.0
    ball_speed_max_px_s: float = 320.0
    ball_speed_skip_px_s: float = 500.0
    carrier_tight_ref_m: float = 0.45
    carrier_tight_ref_px: float = 40.0
    carrier_tight_bonus: float = 0.12
    # Good-pass-moment detection (plan_events): threshold + local score peaks
    freeze_min_pick_score: float = 0.68
    freeze_min_pass_score: float = 0.48
    freeze_local_peak_half_window: int = 12
    freeze_detect_local_peaks: bool = True

    @classmethod
    def metric(cls) -> PassWeights:
        """Reference distances in pitch meters (homography-calibrated scoring).

        Tuned so lane scores spread across [0, 1]: a *very* open lane keeps the
        nearest opponent ~8 m off the pass line, a tight one ~1-2 m; a strong
        forward pass advances ~25 m toward the opponent goal.
        """
        return cls(
            open_ref=8.0,
            space_ref=8.0,
            forward_ref=25.0,
            min_length=2.0,
            max_length=45.0,
            lane_width=2.5,
            lane_in_image_space=False,
            lane_use_body_center=True,
            teammate_lane_width=2.0,
            teammate_open_ref=0.5,
            teammate_lane_penalty=0.10,
            use_carrier_motion=True,
            motion_min_displacement_m=0.35,
            backward_penalty=0.22,
            backward_attack_penalty=0.18,
            ball_speed_ref_m=1.2,
            ball_speed_max_m=3.5,
            ball_speed_skip_m=6.5,
            carrier_tight_ref_m=0.40,
            freeze_min_pick_score=0.66,
            freeze_min_pass_score=0.45,
        )


@dataclass(frozen=True)
class PassOption:
    receiver_index: int
    receiver_xy: np.ndarray
    score: float
    openness: float        # rival lane clearance used for scoring (m)
    opponent_openness: float
    teammate_openness: float
    rivals_in_lane: int
    teammates_in_lane: int
    segment_opponent_openness: float  # legacy: min dist without lane filter
    segment_teammate_openness: float
    forward_gain: float    # raw projection onto attack direction
    motion_alignment: float  # cos(pass, run); 0 if run unknown
    receiver_space: float  # raw distance, receiver to nearest opponent
    length: float


def _rival_lane_width_steps(
    base: float,
    pass_length_m: float,
    weights: PassWeights,
) -> float:
    """Widen rival corridor on longer passes using fixed tiers (not proportional to distance)."""
    if pass_length_m <= weights.lane_width_mid_threshold_m:
        return base
    if pass_length_m <= weights.lane_width_long_threshold_m:
        return min(weights.lane_width_max_m, base + weights.lane_width_mid_boost_m)
    return min(weights.lane_width_max_m, base + weights.lane_width_long_boost_m)


def _lane_width_for_pass(
    weights: PassWeights,
    *,
    pass_length: float,
    pass_length_px: float | None,
) -> float | None:
    """Corridor width in the units used for lane geometry (px or m)."""
    if weights.lane_width is None or weights.lane_width <= 0:
        return None
    if weights.lane_in_image_space and pass_length_px is not None and pass_length > 1e-3:
        return weights.lane_width * (pass_length_px / pass_length)
    if not weights.lane_in_image_space:
        return _rival_lane_width_steps(weights.lane_width, pass_length, weights)
    return weights.lane_width


def _teammate_lane_width(
    weights: PassWeights, *, pass_length: float, pass_length_px: float | None
) -> float | None:
    base = _lane_width_for_pass(
        weights, pass_length=pass_length, pass_length_px=pass_length_px
    )
    if weights.teammate_lane_width is not None:
        if weights.lane_in_image_space and pass_length_px is not None and pass_length > 1e-3:
            return weights.teammate_lane_width * (pass_length_px / pass_length)
        return weights.teammate_lane_width
    if base is not None:
        return base * 0.5
    return None


def _open_ref_for_lane(weights: PassWeights, lane_width: float | None) -> float:
    if weights.lane_in_image_space and lane_width is not None and lane_width > 0:
        return lane_width
    return weights.open_ref


def _backward_motion_penalty_from_align(
    align: float,
    weights: PassWeights,
) -> float:
    """Run-direction penalty from pass·run cosine (same formula as main demo)."""
    if weights.backward_penalty <= 0:
        return 0.0
    if align >= weights.backward_cos_threshold:
        return 0.0
    span = 1.0 - weights.backward_cos_threshold
    severity = min(1.0, (weights.backward_cos_threshold - align) / span)
    return weights.backward_penalty * severity


def _backward_motion_penalty(
    pass_delta: np.ndarray,
    motion_dir: np.ndarray | None,
    weights: PassWeights,
) -> tuple[float, float]:
    """Penalty for passes behind the carrier's run; returns (penalty, cos alignment)."""
    if motion_dir is None or weights.backward_penalty <= 0:
        return 0.0, 0.0
    length = float(np.linalg.norm(pass_delta))
    if length < 1e-6:
        return 0.0, 0.0
    align = float(unit(pass_delta) @ motion_dir)
    return _backward_motion_penalty_from_align(align, weights), align


def _backward_attack_penalty(
    forward_gain: float,
    attack: np.ndarray,
    motion_dir: np.ndarray | None,
    weights: PassWeights,
) -> float:
    """Penalty for passes toward own goal (negative forward gain vs attack direction)."""
    if weights.backward_attack_penalty <= 0 or forward_gain >= 0:
        return 0.0
    if weights.backward_attack_only_when_running_forward:
        if motion_dir is None:
            return 0.0
        if float(motion_dir @ attack) < weights.backward_attack_motion_cos_threshold:
            return 0.0
    severity = min(1.0, -forward_gain / weights.forward_ref)
    return weights.backward_attack_penalty * severity


def _teammate_lane_penalty_amount(
    teammate_openness: float,
    weights: PassWeights,
    *,
    open_ref: float,
) -> float:
    """Soft deduction when a teammate is in the narrow corridor (0 if none)."""
    if weights.teammate_lane_penalty <= 0 or not np.isfinite(teammate_openness):
        return 0.0
    ref = weights.teammate_open_ref or open_ref
    if ref <= 0:
        return 0.0
    tightness = max(0.0, 1.0 - teammate_openness / ref)
    return weights.teammate_lane_penalty * tightness


def score_pass_options(
    detections: sv.Detections,
    carrier: Carrier,
    *,
    weights: PassWeights = PassWeights(),
    attack_dir: np.ndarray | None = None,
    positions: np.ndarray | None = None,
    carrier_motion_dir: np.ndarray | None = None,
    body_pitch_m: np.ndarray | None = None,
) -> list[PassOption]:
    """Rank every teammate as a passing option (best first).

    Pass *positions* (e.g. pitch coordinates in meters) to score in metric space
    instead of image pixels. *carrier_motion_dir* is the unit run vector from recent
    Kalman filter velocity (same coordinate system as *positions*).
    """
    pmask = player_mask(detections)
    feet_img = feet_xy(detections)
    feet = positions if positions is not None else feet_img
    body = bbox_center_xy(detections)
    body_lane = body_pitch_m if body_pitch_m is not None else body
    teams = detections.data["team"]
    use_image_lane = weights.lane_in_image_space and positions is not None
    lane_feet = feet_img if use_image_lane else feet

    teammates = pmask & (teams == carrier.team)
    teammates[carrier.index] = False
    opponents = pmask & (teams == (1 - carrier.team))

    carrier_xy = feet[carrier.index]
    opp_xy = feet[opponents]
    attack = (
        attack_direction(detections, carrier.team)
        if attack_dir is None
        else attack_dir
    )

    options: list[PassOption] = []
    for idx in np.flatnonzero(teammates):
        receiver_xy = feet[idx]
        delta = receiver_xy - carrier_xy
        length = float(np.linalg.norm(delta))
        if length < 1e-6:
            continue

        pass_len_px = float(
            np.linalg.norm(feet_img[idx] - feet_img[carrier.index])
        )
        lane_w = _lane_width_for_pass(
            weights, pass_length=length, pass_length_px=pass_len_px
        )
        lane_kw = dict(
            t_min=weights.lane_t_min,
            t_max=weights.lane_t_max,
            lane_width=lane_w,
        )
        team_lane_kw = {
            **lane_kw,
            "lane_width": _teammate_lane_width(
                weights, pass_length=length, pass_length_px=pass_len_px
            ),
        }
        lane_carrier = lane_feet[carrier.index]
        lane_receiver = lane_feet[idx]
        lane_opp_feet = lane_feet[opponents]
        lane_opp_body = body_lane[opponents]
        open_ref = _open_ref_for_lane(weights, lane_w)
        opp_radius = None
        if weights.lane_use_body_center and len(lane_opp_feet):
            boxes = detections.xyxy[opponents]
            if use_image_lane:
                opp_radius = 0.22 * (boxes[:, 2] - boxes[:, 0])
            elif pass_len_px > 1e-3 and length > 1e-3:
                m_per_px = length / pass_len_px
                opp_radius = 0.22 * (boxes[:, 2] - boxes[:, 0]) * m_per_px

        if len(opp_xy):
            if weights.use_lane_openness:
                if weights.lane_use_body_center:
                    opponent_openness, segment_opponent_openness = (
                        lane_segment_clearance_body(
                            lane_opp_feet,
                            lane_opp_body,
                            lane_carrier,
                            lane_receiver,
                            player_radius=opp_radius,
                            **lane_kw,
                        )
                    )
                    rivals_in_lane = count_lane_blockers_body(
                        lane_opp_feet,
                        lane_opp_body,
                        lane_carrier,
                        lane_receiver,
                        player_radius=opp_radius,
                        **lane_kw,
                    )
                else:
                    opponent_openness, segment_opponent_openness = lane_segment_clearance(
                        lane_opp_feet, lane_carrier, lane_receiver, **lane_kw
                    )
                    rivals_in_lane = count_lane_blockers(
                        lane_opp_feet, lane_carrier, lane_receiver, **lane_kw
                    )
            else:
                segment_opponent_openness = opponent_openness = float(
                    point_to_segment_distance(
                        lane_opp_feet, lane_carrier, lane_receiver
                    ).min()
                )
                rivals_in_lane = len(opp_xy)
            receiver_space = float(np.linalg.norm(opp_xy - receiver_xy, axis=1).min())
        else:
            opponent_openness = segment_opponent_openness = receiver_space = open_ref
            rivals_in_lane = 0

        blockers = teammates.copy()
        blockers[carrier.index] = False
        blockers[idx] = False
        team_xy = lane_feet[blockers]
        teammates_in_lane = 0
        if len(team_xy):
            if weights.use_lane_openness:
                teammate_openness, segment_teammate_openness = lane_segment_clearance(
                    team_xy, lane_carrier, lane_receiver, **team_lane_kw
                )
                teammates_in_lane = count_lane_blockers(
                    team_xy, lane_carrier, lane_receiver, **team_lane_kw
                )
            else:
                segment_teammate_openness = teammate_openness = float(
                    point_to_segment_distance(
                        team_xy, lane_carrier, lane_receiver
                    ).min()
                )
                teammates_in_lane = len(team_xy)
        else:
            teammate_openness = segment_teammate_openness = open_ref

        if weights.openness_rivals_only:
            openness = opponent_openness
        else:
            openness = min(opponent_openness, teammate_openness)

        forward_gain = float(delta @ attack)

        n_open = min(openness / open_ref, 1.0)
        n_space = min(receiver_space / weights.space_ref, 1.0)
        n_forward = float(np.clip(forward_gain / weights.forward_ref, -1.0, 1.0))

        score = (
            weights.openness * n_open
            + weights.forward * max(n_forward, 0.0)
            + weights.space * n_space
        )
        tm_ref = weights.teammate_open_ref
        if tm_ref is None:
            tm_ref = (team_lane_kw["lane_width"] or open_ref) * 0.5
        elif use_image_lane and length > 1e-3:
            tm_ref = tm_ref * (pass_len_px / length)
        score -= _teammate_lane_penalty_amount(
            teammate_openness, weights, open_ref=tm_ref
        )
        back_pen, motion_align = _backward_motion_penalty(
            delta, carrier_motion_dir, weights
        )
        score -= back_pen
        score -= _backward_attack_penalty(
            forward_gain, attack, carrier_motion_dir, weights
        )
        if length < weights.min_length or length > weights.max_length:
            continue

        options.append(
            PassOption(
                receiver_index=int(idx),
                receiver_xy=receiver_xy,
                score=float(score),
                openness=openness,
                opponent_openness=opponent_openness,
                teammate_openness=teammate_openness,
                segment_opponent_openness=segment_opponent_openness,
                segment_teammate_openness=segment_teammate_openness,
                rivals_in_lane=rivals_in_lane,
                teammates_in_lane=teammates_in_lane,
                forward_gain=forward_gain,
                motion_alignment=motion_align,
                receiver_space=receiver_space,
                length=length,
            )
        )

    options.sort(key=lambda o: o.score, reverse=True)
    return options
