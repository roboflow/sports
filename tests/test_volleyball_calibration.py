"""Library calibration and rendering tests without model or example imports."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pytest

from sports.annotators.volleyball import annotate_court, render_radar
from sports.common.calibration import CourtCalibration
from sports.configs.volleyball import VolleyballCourtConfiguration


@dataclass
class _RectangularCourt:
    """Provide another sport's rectangular geometry in meters."""

    length: float = 28
    width: float = 15

    @property
    def vertices(self) -> List[Tuple[float, float]]:
        """Return four corresponding perimeter corners."""
        return [(0, 0), (self.length, 0), (self.length, self.width), (0, self.width)]

    @property
    def edges(self) -> List[Tuple[int, int]]:
        """Return one-based perimeter edges."""
        return [(1, 2), (2, 3), (3, 4), (4, 1)]


def _calibration() -> CourtCalibration:
    """Calibrate a rectangular source view to the default volleyball court."""
    return CourtCalibration(
        [[10, 10], [310, 10], [310, 170], [10, 170]],
        (320, 180),
        VolleyballCourtConfiguration(),
    )


class TestCourtCalibration:
    """Four-corner calibration works independently of the video example."""

    def test_maps_corners_and_center(self) -> None:
        """Corresponding corners and the source center map to court geometry."""
        calibration = _calibration()
        points = np.vstack([calibration.corners, [160, 90]])

        transformed = calibration.transform_points(points)

        expected = list(calibration.config.vertices[:4]) + [(900, 450)]
        np.testing.assert_allclose(transformed, expected, atol=1e-3)
        assert transformed.dtype == np.float32

    def test_accepts_other_rectangular_geometry_and_units(self) -> None:
        """A structural court configuration supports another sport's meter units."""
        calibration = CourtCalibration(
            [[10, 10], [310, 10], [310, 170], [10, 170]],
            (320, 180),
            _RectangularCourt(),
        )

        transformed = calibration.transform_points([[160, 90]])

        np.testing.assert_allclose(transformed, [[14, 7.5]], atol=1e-5)

    def test_round_trip_preserves_source_json_schema(self, tmp_path: Path) -> None:
        """Saved source corners reload without needing a model or example module."""
        calibration = _calibration()
        path = tmp_path / "nested" / "court.json"
        calibration.save(path)

        loaded = CourtCalibration.load(path, (320, 180), calibration.config)

        np.testing.assert_array_equal(loaded.corners, calibration.corners)
        assert loaded.frame_size == calibration.frame_size
        assert set(json.loads(path.read_text())) == {"frame_size", "corners"}

    def test_rejects_different_video_resolution(self, tmp_path: Path) -> None:
        """A calibration cannot be silently reused at another source resolution."""
        path = tmp_path / "court.json"
        calibration = _calibration()
        calibration.save(path)

        with pytest.raises(ValueError, match="resolution"):
            CourtCalibration.load(path, (640, 360), calibration.config)

    @pytest.mark.parametrize(
        "corners",
        [
            [[10, 10], [310, 170], [310, 10], [10, 170]],
            [[10, 10], [100, 10], [200, 10], [310, 10]],
            [[10, 10], [310, 10], [310, 170]],
            [[10, 10], [320, 10], [310, 170], [10, 170]],
            [[10, 10], [310, 10], [310, float("nan")], [10, 170]],
        ],
    )
    def test_rejects_invalid_source_quadrilateral(
        self, corners: List[List[float]]
    ) -> None:
        """Invalid quadrilaterals and nonfinite or out-of-frame corners fail."""
        with pytest.raises(ValueError):
            CourtCalibration(corners, (320, 180), VolleyballCourtConfiguration())

    def test_empty_points_keep_coordinate_dimensions(self) -> None:
        """Empty source positions preserve their coordinate dimensions."""
        calibration = _calibration()

        positions = calibration.transform_points([])

        assert positions.shape == (0, 2)

    def test_empty_mask_keeps_alignment(self) -> None:
        """An empty set of source positions produces an empty selection mask."""
        calibration = _calibration()

        mask = calibration.player_mask([])

        assert mask.shape == (0,)

    def test_margin_uses_configuration_units(self) -> None:
        """The free zone retains nearby anchors and excludes distant background."""
        calibration = _calibration()

        mask = calibration.player_mask([[160, 90], [0, 90], [-100, 90]], margin=100)

        assert mask.tolist() == [True, True, False]

    def test_horizon_point_cannot_be_projected_to_court_origin(self) -> None:
        """A finite pixel on the vanishing line has no finite court position."""
        calibration = CourtCalibration(
            [[130, 80], [190, 80], [310, 160], [10, 160]],
            (320, 180),
            VolleyballCourtConfiguration(),
        )

        with pytest.raises(ValueError, match="cannot be projected"):
            calibration.transform_points([[100, 60]])

    def test_player_mask_excludes_horizon_and_keeps_point_alignment(self) -> None:
        """Horizon anchors are excluded without dropping valid anchors beside them."""
        calibration = CourtCalibration(
            [[130, 80], [190, 80], [310, 160], [10, 160]],
            (320, 180),
            VolleyballCourtConfiguration(),
        )

        mask = calibration.player_mask([[160, 100], [100, 60], [160, 120]])

        assert mask.tolist() == [True, False, True]

    def test_player_mask_still_rejects_nonfinite_source_positions(self) -> None:
        """Invalid source coordinates raise instead of becoming excluded observations."""
        calibration = _calibration()

        with pytest.raises(ValueError, match="finite array"):
            calibration.player_mask([[float("nan"), 60]])


class TestCourtAnnotations:
    """Court overlays expose reusable rendering behavior."""

    def test_court_overlay_projects_into_source_frame(self) -> None:
        """Calibration overlay draws a boundary and net at their source pixels."""
        frame = np.zeros((180, 320, 3), dtype=np.uint8)

        result = annotate_court(frame, _calibration())

        assert result is frame
        np.testing.assert_array_equal(frame[90, 10], [0, 255, 255])
        np.testing.assert_array_equal(frame[90, 160], [0, 255, 255])
        np.testing.assert_array_equal(frame[90, 80], [0, 0, 0])

    def test_radar_defaults_match_explicit_configuration(self) -> None:
        """Default geometry and team colors match an explicit caller configuration."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        points = np.array([[450, 450], [1350, 450]], dtype=np.float32)
        teams = np.array([0, 1])
        expected = render_radar(
            frame.copy(),
            points,
            teams,
            VolleyballCourtConfiguration(),
            ("#FF1493", "#00BFFF"),
        )

        result = render_radar(frame, points, teams)

        assert result is frame
        np.testing.assert_array_equal(result, expected)
        assert np.any(result[-252:, :] != 0)
        assert not result[:468].any()

    @pytest.mark.parametrize("coordinate", [np.nan, np.inf])
    @pytest.mark.parametrize("colors", [None, ()])
    def test_radar_validates_positions_even_when_team_is_not_drawn(
        self, coordinate: float, colors: Optional[Tuple[str, ...]]
    ) -> None:
        """Unknown team IDs cannot hide invalid positions from API validation."""
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="finite"):
            render_radar(frame, [[coordinate, 0]], [2], team_colors=colors)

    def test_radar_uses_supplied_team_colors(self) -> None:
        """A caller's palette changes player markers without changing court geometry."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        points = np.array([[450, 450]], dtype=np.float32)
        teams = np.array([0])

        result = render_radar(frame, points, teams, team_colors=("#00FF00",))

        channels = result.astype(np.int16)
        green = channels[:, :, 1] > channels[:, :, 0] + 50
        green &= channels[:, :, 1] > channels[:, :, 2] + 50
        assert green.any()

    @pytest.mark.parametrize("teams", [np.array([]), np.array([[0]]), np.array([0, 1])])
    def test_radar_rejects_unaligned_team_ids(
        self, teams: npt.NDArray[np.int64]
    ) -> None:
        """Team IDs must be one-dimensional and align with court positions."""
        frame = np.zeros((180, 320, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="teams"):
            render_radar(frame, np.array([[450, 450]]), teams)
