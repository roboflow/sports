"""Regression: opponent pass flying past a teammate must not become a false pass."""

from __future__ import annotations

import numpy as np
import supervision as sv

from sports.common.passes import PassDetectionConfig, scan_possession_events
from sports.configs.soccer import BALL_CLASS_ID, PLAYER_CLASS_ID


def _box(feet_x: float, feet_y: float) -> np.ndarray:
    return np.array([[feet_x - 12, feet_y - 50, feet_x + 12, feet_y]], dtype=np.float32)


def _frame(
    *,
    players: list[tuple[int, int, tuple[float, float]]],
    ball: tuple[float, float],
) -> sv.Detections:
    """``players`` is ``(tracker_id, team, feet_xy)``."""
    boxes = [_box(*feet) for _, _, feet in players]
    ball_box = np.array(
        [[ball[0] - 6, ball[1] - 6, ball[0] + 6, ball[1]]],
        dtype=np.float32,
    )
    boxes.append(ball_box)
    return sv.Detections(
        xyxy=np.vstack(boxes),
        class_id=np.array(
            [PLAYER_CLASS_ID] * len(players) + [BALL_CLASS_ID],
            dtype=int,
        ),
        tracker_id=np.array([tid for tid, _, _ in players] + [-1], dtype=int),
        data={"team": np.array([team for _, team, _ in players] + [-1], dtype=int)},
    )


def _scan(ball_path, *, transit_min_speed_px_per_frame: float = 10.0):
    players = [
        (15, 0, (100.0, 220.0)),
        (6, 1, (280.0, 220.0)),
        (3, 1, (460.0, 220.0)),
    ]
    config = PassDetectionConfig(
        min_control_frames=2,
        min_arrival_frames=2,
        min_reception_arrival_frames=2,
        min_arrival_control_frames=1,
        adjacent_pass_max_gap_frames=15,
        min_ball_travel_px=5.0,
        min_ball_travel_m=0.05,
        max_pass_gap_frames=100,
        transit_min_speed_px_per_frame=transit_min_speed_px_per_frame,
    )
    frames = [(i, _frame(players=players, ball=ball)) for i, ball in enumerate(ball_path, 1)]
    return scan_possession_events(
        iter(frames),
        config=config,
        metric=False,
        fps=25.0,
    )


def test_flyby_teammate_does_not_steal_opponent_turnover():
    """#15 passes; ball skims #6; #3 intercepts → turnover #15→#3, not pass #6→#3."""
    ball_path = (
        [(110.0, 215.0)] * 5
        + [(130 + 20 * i, 215.0) for i in range(8)]
        + [(275.0, 215.0), (278.0, 215.0), (282.0, 215.0)]
        + [(300 + 20 * i, 215.0) for i in range(8)]
        + [(455.0, 215.0)] * 5
    )
    # Raise transit speed gate so proximity-only control near #6 is not vetoed by
    # speed alone — forces the credit_possession / promote guards to do the work.
    result = _scan(ball_path, transit_min_speed_px_per_frame=40.0)
    pass_pairs = [(p.passer_tid, p.receiver_tid) for p in result.passes]
    turn_pairs = [(t.passer_tid, t.interceptor_tid) for t in result.turnovers]

    assert (6, 3) not in pass_pairs, f"false fly-by pass detected: {pass_pairs}"
    assert (15, 3) in turn_pairs, (
        f"expected turnover #15→#3, got turns={turn_pairs} passes={pass_pairs}"
    )


def test_fast_flyby_still_turnovers_to_interceptor():
    """Default transit speed: ball flying past #6 still yields #15→#3 turnover."""
    ball_path = (
        [(110.0, 215.0)] * 5
        + [(130 + 25 * i, 215.0) for i in range(6)]
        + [(278.0 + 0.5 * i, 215.0) for i in range(8)]
        + [(300 + 25 * i, 215.0) for i in range(6)]
        + [(455.0, 215.0)] * 5
    )
    result = _scan(ball_path)
    pass_pairs = [(p.passer_tid, p.receiver_tid) for p in result.passes]
    turn_pairs = [(t.passer_tid, t.interceptor_tid) for t in result.turnovers]
    assert (6, 3) not in pass_pairs
    assert (15, 3) in turn_pairs
