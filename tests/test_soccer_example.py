import importlib.util
from pathlib import Path

import numpy as np
import pytest
import supervision as sv

from sports.annotators.soccer import draw_pitch

MAIN_PATH = Path(__file__).parents[1] / "examples" / "soccer" / "main.py"


@pytest.fixture(scope="module")
def main():
    """Load examples/soccer/main.py, which is a script rather than a package module."""
    spec = importlib.util.spec_from_file_location("soccer_main", MAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_keypoints(xy, confidence):
    """One pitch skeleton with the given 32 keypoint positions and confidences."""
    return sv.KeyPoints(
        xy=np.asarray(xy, dtype=np.float32)[np.newaxis],
        keypoint_confidence=np.asarray(confidence, dtype=np.float32)[np.newaxis],
    )


def pitch_keypoints_in_view(main, visible_ids):
    """Keypoints as a camera would see them: an exact 0.1 scale of the pitch,
    confident only for `visible_ids` and placed at junk positions elsewhere."""
    vertices = np.array(main.CONFIG.vertices, dtype=np.float32)
    xy = np.full_like(vertices, 1900.0)  # off-screen guesses pile up at the frame edge
    confidence = np.full(len(vertices), 0.1)
    xy[visible_ids] = vertices[visible_ids] * 0.1
    confidence[visible_ids] = 0.9
    return make_keypoints(xy, confidence)


PLAYER_AT_CENTRE_SPOT = sv.Detections(xyxy=np.array([[595.0, 330.0, 605.0, 350.0]]))


class TestHideLowConfidenceKeypoints:
    def test_zeroes_only_keypoints_below_threshold(self, main):
        keypoints = make_keypoints(
            [[10, 20], [30, 40], [50, 60]], [0.9, 0.2, main.KEYPOINT_CONFIDENCE_THRESHOLD])

        result = main.hide_low_confidence_keypoints(keypoints)

        np.testing.assert_array_equal(result.xy[0], [[10, 20], [0, 0], [50, 60]])

    def test_accepts_frame_without_pitch(self, main):
        # This is what `sv.KeyPoints.from_ultralytics` returns when no pitch is found.
        keypoints = sv.KeyPoints.empty()

        result = main.hide_low_confidence_keypoints(keypoints)

        assert len(result) == 0


class TestRenderRadar:
    def test_projects_players_with_confident_keypoints_only(self, main):
        # Halfway line and centre circle, which is all a centre camera might show.
        keypoints = main.hide_low_confidence_keypoints(
            pitch_keypoints_in_view(main, [13, 14, 15, 16, 30, 31]))

        radar = main.render_radar(PLAYER_AT_CENTRE_SPOT, keypoints, np.array([0]))

        # Bottom centre of the box is (600, 350), i.e. the centre spot (6000, 3500).
        centre_spot = (3500 // 10 + 50, 6000 // 10 + 50)
        assert tuple(radar[centre_spot]) == sv.Color.from_hex(main.COLORS[0]).as_bgr()

    @pytest.mark.parametrize(
        "visible_ids",
        [
            pytest.param([], id="no-keypoints-visible"),
            pytest.param([13, 14, 15], id="three-keypoints-visible"),
            pytest.param([13, 14, 15, 16], id="four-keypoints-on-halfway-line"),
        ],
    )
    def test_returns_empty_pitch_when_view_cannot_be_mapped(self, main, visible_ids):
        keypoints = main.hide_low_confidence_keypoints(pitch_keypoints_in_view(main, visible_ids))

        radar = main.render_radar(PLAYER_AT_CENTRE_SPOT, keypoints, np.array([0]))

        np.testing.assert_array_equal(radar, draw_pitch(config=main.CONFIG))

    def test_returns_empty_pitch_when_no_pitch_detected(self, main):
        # This is what `sv.KeyPoints.from_ultralytics` returns when no pitch is found.
        keypoints = sv.KeyPoints.empty()

        radar = main.render_radar(PLAYER_AT_CENTRE_SPOT, keypoints, np.array([0]))

        np.testing.assert_array_equal(radar, draw_pitch(config=main.CONFIG))
