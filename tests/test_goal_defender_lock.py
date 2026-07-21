"""Defensive-block goal sides + nearer-goal GK assignment (PR3)."""

from __future__ import annotations

import numpy as np

from sports.common.goalkeeper import (
    PITCH_LENGTH_CM,
    PITCH_WIDTH_CM,
    infer_goal_defenders,
    resolve_goalkeepers_team_by_goal,
)


def test_infer_goal_defenders_uses_defensive_blocks():
    """Team with the stronger defensive-block margin owns that goal."""
    mid_y = PITCH_WIDTH_CM / 2.0
    # Team 0 deepest on the left; team 1 deepest on the right.
    pitch = np.array(
        [
            [200.0, mid_y],
            [300.0, mid_y],
            [400.0, mid_y],
            [5000.0, mid_y],
            [PITCH_LENGTH_CM - 200.0, mid_y],
            [PITCH_LENGTH_CM - 300.0, mid_y],
            [PITCH_LENGTH_CM - 400.0, mid_y],
            [7000.0, mid_y],
        ],
        dtype=np.float32,
    )
    teams = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
    assert infer_goal_defenders(pitch, teams, n_defenders=3) == (0, 1)


def test_infer_goal_defenders_handshake_prefers_stronger_margin():
    mid_y = PITCH_WIDTH_CM / 2.0
    # Team 1 is clearly deeper left; right side is nearly tied → (1, 0).
    pitch = np.array(
        [
            [800.0, mid_y],
            [900.0, mid_y],
            [1000.0, mid_y],
            [200.0, mid_y],
            [250.0, mid_y],
            [300.0, mid_y],
            [PITCH_LENGTH_CM - 500.0, mid_y],
            [PITCH_LENGTH_CM - 480.0, mid_y],
            [PITCH_LENGTH_CM - 520.0, mid_y],
            [PITCH_LENGTH_CM - 450.0, mid_y],
            [PITCH_LENGTH_CM - 430.0, mid_y],
            [PITCH_LENGTH_CM - 470.0, mid_y],
        ],
        dtype=np.float32,
    )
    teams = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=int)
    assert infer_goal_defenders(pitch, teams, n_defenders=3) == (1, 0)


def test_resolve_goalkeeper_uses_nearer_goal_defender():
    mid_y = PITCH_WIDTH_CM / 2.0
    outfield = np.array(
        [
            [200.0, mid_y],
            [300.0, mid_y],
            [400.0, mid_y],
            [PITCH_LENGTH_CM - 200.0, mid_y],
            [PITCH_LENGTH_CM - 300.0, mid_y],
            [PITCH_LENGTH_CM - 400.0, mid_y],
        ],
        dtype=np.float32,
    )
    out_teams = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    gks = np.array(
        [
            [150.0, mid_y],
            [PITCH_LENGTH_CM - 150.0, mid_y],
        ],
        dtype=np.float32,
    )
    got = resolve_goalkeepers_team_by_goal(gks, outfield, out_teams)
    assert got.tolist() == [0, 1]
