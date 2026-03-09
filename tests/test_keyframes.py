"""Unit tests for sports.pipelines.keyframes.

All tests run without the SAM 3 model or any video file — only pure Python
and numpy are required.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# Permanently stub heavy deps before importing the module under test.
# Using setdefault so we don't overwrite a real install if it exists.
for _mod in ["ultralytics", "ultralytics.models", "ultralytics.models.sam", "supervision"]:
    sys.modules.setdefault(_mod, MagicMock())

from sports.pipelines.keyframes import (  # noqa: E402
    FOOTBALL,
    TENNIS,
    Keyframe,
    KeyframeGenerator,
    SportConfig,
    _OffsetSmoother,
    _rdp,
)


def _make_gen(sport: SportConfig = FOOTBALL) -> KeyframeGenerator:
    """Return a KeyframeGenerator without invoking SAM 3 predictor init."""
    gen = object.__new__(KeyframeGenerator)
    gen.sport = sport
    gen.crop_width_px = 1080
    gen.margin_px = 32
    gen.smoothing_alpha = 0.25
    gen.max_speed_px_per_s = 480.0
    gen.epsilon_frac = 0.008
    gen._predictor = MagicMock()
    return gen


# ---------------------------------------------------------------------------
# SportConfig
# ---------------------------------------------------------------------------


class TestSportConfig:
    def test_all_prompts_order(self):
        cfg = SportConfig(player_prompts=["player a", "player b"], ball_prompts=["ball x"])
        assert cfg.all_prompts == ["player a", "player b", "ball x"]

    def test_n_player_classes(self):
        cfg = SportConfig(player_prompts=["p1", "p2", "p3"], ball_prompts=["b"])
        assert cfg.n_player_classes == 3

    def test_football_config(self):
        assert FOOTBALL.ball_weight == 3.0
        assert "football player" in FOOTBALL.player_prompts
        assert len(FOOTBALL.ball_prompts) >= 1

    def test_tennis_config(self):
        assert TENNIS.ball_weight == 8.0
        assert TENNIS.conf < FOOTBALL.conf  # lower threshold for motion-blurred ball


# ---------------------------------------------------------------------------
# Keyframe
# ---------------------------------------------------------------------------


class TestKeyframe:
    def test_as_pair(self):
        kf = Keyframe(timestamp_s=1.5, offset_px=200)
        assert kf.as_pair() == (1.5, 200)

    def test_as_dict(self):
        kf = Keyframe(timestamp_s=2.0, offset_px=300)
        assert kf.as_dict() == {"t": 2.0, "o": 300}

    def test_frozen(self):
        kf = Keyframe(timestamp_s=0.0, offset_px=0)
        with pytest.raises(Exception):
            kf.timestamp_s = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _OffsetSmoother
# ---------------------------------------------------------------------------


class TestOffsetSmoother:
    def test_first_update_returns_target(self):
        s = _OffsetSmoother(fps=30.0, alpha=0.5, max_speed_px_per_s=1000.0)
        assert s.update(500.0) == 500.0

    def test_convergence(self):
        s = _OffsetSmoother(fps=30.0, alpha=0.5, max_speed_px_per_s=10_000.0)
        val = 0.0
        for _ in range(30):
            val = s.update(100.0)
        assert abs(val - 100.0) < 0.01

    def test_slew_rate_limit(self):
        # max 60 px/s at 30 fps → max 2 px/frame
        s = _OffsetSmoother(fps=30.0, alpha=1.0, max_speed_px_per_s=60.0)
        s.update(0.0)
        v = s.update(1000.0)
        assert abs(v - 2.0) < 1e-9

    def test_reset(self):
        s = _OffsetSmoother(fps=30.0, alpha=0.5, max_speed_px_per_s=1000.0)
        s.update(500.0)
        s.reset()
        assert s.value is None
        assert s.update(200.0) == 200.0

    def test_alpha_clipped_to_01(self):
        s = _OffsetSmoother(fps=30.0, alpha=5.0, max_speed_px_per_s=1000.0)
        assert s.alpha == 1.0
        s2 = _OffsetSmoother(fps=30.0, alpha=-1.0, max_speed_px_per_s=1000.0)
        assert s2.alpha == 0.0


# ---------------------------------------------------------------------------
# _rdp
# ---------------------------------------------------------------------------


class TestRdp:
    def test_trivial_two_points(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        result = _rdp(pts, epsilon=10.0)
        np.testing.assert_array_equal(result, pts)

    def test_collinear_points_reduced(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        result = _rdp(pts, epsilon=0.1)
        assert result.shape[0] == 2
        np.testing.assert_array_equal(result[0], [0.0, 0.0])
        np.testing.assert_array_equal(result[-1], [3.0, 0.0])

    def test_outlier_preserved(self):
        pts = np.array([[0.0, 0.0], [1.0, 100.0], [2.0, 0.0]])
        result = _rdp(pts, epsilon=1.0)
        assert result.shape[0] == 3

    def test_tight_epsilon_keeps_all(self):
        pts = np.array([[0.0, 0.0], [1.0, 5.0], [2.0, 0.0]])
        result = _rdp(pts, epsilon=0.0)
        assert result.shape[0] == 3


# ---------------------------------------------------------------------------
# KeyframeGenerator._compute_target_offset
# ---------------------------------------------------------------------------


class TestComputeTargetOffset:
    def test_no_detections_returns_centre(self):
        gen = _make_gen()
        empty = np.empty((0, 4), dtype=np.float32)
        result = gen._compute_target_offset(
            player_boxes=empty, ball_boxes=empty,
            frame_width=1920, crop_width=1080, max_offset=840,
        )
        assert result == pytest.approx(420.0)

    def test_centred_single_player(self):
        gen = _make_gen()
        # Player box centred at x=960 in a 1920-wide frame, crop=1080
        player = np.array([[420.0, 100.0, 1500.0, 800.0]], dtype=np.float32)
        empty = np.empty((0, 4), dtype=np.float32)
        result = gen._compute_target_offset(
            player_boxes=player, ball_boxes=empty,
            frame_width=1920, crop_width=1080, max_offset=840,
        )
        # desired_offset = 960 - 540 = 420; within bounds
        assert result == pytest.approx(420.0)

    def test_ball_pulls_crop_toward_it(self):
        gen = _make_gen()
        player = np.array([[0.0, 100.0, 200.0, 800.0]], dtype=np.float32)
        ball = np.array([[1700.0, 500.0, 1800.0, 600.0]], dtype=np.float32)
        empty = np.empty((0, 4), dtype=np.float32)
        result_no_ball = gen._compute_target_offset(
            player_boxes=player, ball_boxes=empty,
            frame_width=1920, crop_width=1080, max_offset=840,
        )
        result_with_ball = gen._compute_target_offset(
            player_boxes=player, ball_boxes=ball,
            frame_width=1920, crop_width=1080, max_offset=840,
        )
        assert result_with_ball > result_no_ball

    def test_offset_clamped_to_max(self):
        gen = _make_gen()
        player = np.array([[1800.0, 100.0, 1920.0, 800.0]], dtype=np.float32)
        empty = np.empty((0, 4), dtype=np.float32)
        result = gen._compute_target_offset(
            player_boxes=player, ball_boxes=empty,
            frame_width=1920, crop_width=1080, max_offset=840,
        )
        assert result <= 840.0

    def test_offset_not_negative(self):
        gen = _make_gen()
        player = np.array([[0.0, 100.0, 100.0, 800.0]], dtype=np.float32)
        empty = np.empty((0, 4), dtype=np.float32)
        result = gen._compute_target_offset(
            player_boxes=player, ball_boxes=empty,
            frame_width=1920, crop_width=1080, max_offset=840,
        )
        assert result >= 0.0

    def test_tennis_ball_dominates(self):
        """Tennis ball weight (8×) should dominate the crop centre vs player."""
        tennis_gen = _make_gen(TENNIS)
        football_gen = _make_gen(FOOTBALL)
        player = np.array([[0.0, 100.0, 200.0, 800.0]], dtype=np.float32)
        ball = np.array([[1700.0, 500.0, 1800.0, 600.0]], dtype=np.float32)
        tennis_offset = tennis_gen._compute_target_offset(
            player_boxes=player, ball_boxes=ball,
            frame_width=1920, crop_width=1080, max_offset=840,
        )
        football_offset = football_gen._compute_target_offset(
            player_boxes=player, ball_boxes=ball,
            frame_width=1920, crop_width=1080, max_offset=840,
        )
        # Tennis pulls harder toward the right-side ball
        assert tennis_offset >= football_offset


# ---------------------------------------------------------------------------
# KeyframeGenerator._split_boxes
# ---------------------------------------------------------------------------


class TestSplitBoxes:
    def _mock_result(self, xyxy: np.ndarray, cls: np.ndarray):
        boxes = MagicMock()
        boxes.__len__ = MagicMock(return_value=len(xyxy))
        boxes.xyxy.cpu().numpy.return_value = xyxy
        boxes.cls.cpu().numpy.return_value = cls
        result = MagicMock()
        result.boxes = boxes
        return result

    def test_none_boxes(self):
        gen = _make_gen()
        result = MagicMock()
        result.boxes = None
        players, balls = gen._split_boxes(result)
        assert players.shape == (0, 4)
        assert balls.shape == (0, 4)

    def test_empty_boxes(self):
        gen = _make_gen()
        xyxy = np.empty((0, 4), dtype=np.float32)
        cls = np.empty((0,), dtype=np.float32)
        result = self._mock_result(xyxy, cls)
        players, balls = gen._split_boxes(result)
        assert players.shape == (0, 4)
        assert balls.shape == (0, 4)

    def test_splits_correctly(self):
        gen = _make_gen()
        # FOOTBALL has 3 player classes (indices 0,1,2); ball = index 3+
        xyxy = np.array([
            [0, 0, 10, 10],    # class 0 → player
            [10, 0, 20, 10],   # class 1 → player
            [20, 0, 30, 10],   # class 3 → ball
        ], dtype=np.float32)
        cls = np.array([0, 1, 3], dtype=np.float32)
        result = self._mock_result(xyxy, cls)
        players, balls = gen._split_boxes(result)
        assert players.shape == (2, 4)
        assert balls.shape == (1, 4)
        np.testing.assert_array_equal(balls[0], [20, 0, 30, 10])

    def test_all_players(self):
        gen = _make_gen()
        xyxy = np.array([[0, 0, 10, 10], [10, 0, 20, 10]], dtype=np.float32)
        cls = np.array([0, 2], dtype=np.float32)
        result = self._mock_result(xyxy, cls)
        players, balls = gen._split_boxes(result)
        assert players.shape == (2, 4)
        assert balls.shape == (0, 4)

    def test_all_balls(self):
        gen = _make_gen()
        xyxy = np.array([[100, 100, 120, 120]], dtype=np.float32)
        cls = np.array([3], dtype=np.float32)
        result = self._mock_result(xyxy, cls)
        players, balls = gen._split_boxes(result)
        assert players.shape == (0, 4)
        assert balls.shape == (1, 4)
