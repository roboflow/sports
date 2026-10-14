from typing import Dict, Union

import numpy as np
import numpy.typing as npt
import pytest
import supervision as sv

from sports.annotators.volleyball import draw_court, draw_points_on_court
from sports.configs.volleyball import VolleyballCourtConfiguration


def test_court_landmarks_preserve_calibration_order_and_attack_distances() -> None:
    """Boundary ordering and attack-line distances match the documented court."""
    config = VolleyballCourtConfiguration()
    assert config.vertices[:4] == [(0, 0), (1800, 0), (1800, 900), (0, 900)]
    assert config.vertices[4:6] == [(900, 0), (900, 900)]
    assert config.vertices[6:] == [(600, 0), (600, 900), (1200, 0), (1200, 900)]
    assert config.edges == [(1, 2), (2, 3), (3, 4), (4, 1), (5, 6), (7, 8), (9, 10)]
    assert len(config.labels) == len(config.vertices)
    assert len(set(config.labels)) == len(config.vertices)


def test_custom_court_landmarks_follow_dimensions() -> None:
    """Changing dimensions relocates the boundary, net, and attack lines."""
    config = VolleyballCourtConfiguration(
        length=2000, width=1000, attack_line_distance=250
    )
    assert config.vertices[2] == (2000, 1000)
    assert config.vertices[4][0] == 1000
    assert config.vertices[6][0] == 750
    assert config.vertices[8][0] == 1250


@pytest.mark.parametrize(
    "kwargs",
    [
        {"length": 0},
        {"width": -1},
        {"width": float("nan")},
        {"length": float("inf")},
        {"length": True},
        {"width": "900"},
        {"attack_line_distance": 0},
        {"attack_line_distance": 900},
        {"attack_line_distance": 1000},
    ],
)
def test_invalid_court_geometry_is_rejected(
    kwargs: Dict[str, Union[float, str]],
) -> None:
    """Invalid dimensions fail while constructing the configuration."""
    with pytest.raises(ValueError):
        VolleyballCourtConfiguration(**kwargs)


def test_renderer_draws_only_boundaries_net_and_attack_lines() -> None:
    """The rendered court contains volleyball lines and no soccer markings."""
    config = VolleyballCourtConfiguration()
    court = draw_court(config)
    assert court.shape == (331, 601, 3)
    assert court.dtype == np.uint8
    white = np.array(sv.Color.WHITE.as_bgr())
    for world_x in (0, 600, 900, 1200, 1800):
        assert np.array_equal(court[165, int(world_x * 0.3) + 30], white)
    for world_y in (0, 900):
        assert np.array_equal(court[int(world_y * 0.3) + 30, 120], white)
    # The center of the near half should contain no soccer circle or penalty mark.
    assert np.array_equal(court[165, 165], court[10, 10])
    assert np.any(court != court[10, 10])


def test_boundaries_remain_visible_without_padding() -> None:
    """Zero padding retains the rightmost and bottom boundary endpoints."""
    court = draw_court(VolleyballCourtConfiguration(), padding=0, scale=0.1)
    assert court.shape == (91, 181, 3)
    assert np.array_equal(court[-1, -1], np.array(sv.Color.WHITE.as_bgr()))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scale": 0},
        {"scale": -1},
        {"scale": float("nan")},
        {"scale": float("inf")},
        {"scale": True},
        {"scale": 0.0001},
        {"scale": 1000},
        {"padding": -1},
        {"padding": 1.5},
        {"padding": False},
    ],
)
def test_invalid_render_parameters_are_rejected(kwargs: Dict[str, float]) -> None:
    """Invalid scale and padding cannot allocate a court canvas."""
    with pytest.raises(ValueError):
        draw_court(VolleyballCourtConfiguration(), **kwargs)


def test_modified_invalid_config_is_revalidated_when_rendering() -> None:
    """Rendering rejects a configuration edited to invalid dimensions."""
    config = VolleyballCourtConfiguration()
    config.width = 0
    with pytest.raises(ValueError):
        draw_court(config)


def test_points_use_world_coordinates_and_modify_provided_court() -> None:
    """Court coordinates set the marker center while preserving image identity."""
    config = VolleyballCourtConfiguration()
    court = draw_court(config)
    result = draw_points_on_court(config, np.array([[450, 450]]), court=court)
    assert result is court
    assert np.array_equal(result[165, 165], np.array(sv.Color.RED.as_bgr()))
    assert np.array_equal(result[165, 171], np.array(sv.Color.BLACK.as_bgr()))


def test_empty_points_preserve_court_and_far_outside_points_are_skipped() -> None:
    """Empty and invisible positions do not paint misleading edge markers."""
    config = VolleyballCourtConfiguration()
    court = draw_court(config)
    original = court.copy()
    assert draw_points_on_court(config, np.empty((0, 2)), court=court) is court
    assert np.array_equal(original, court)
    result = draw_points_on_court(
        config, np.array([[-10000, 0], [1e100, 0]]), court=court
    )
    assert np.array_equal(original, result)


def test_points_in_visible_free_zone_are_retained() -> None:
    """A player within visible free-zone padding remains visible on the map."""
    result = draw_points_on_court(
        VolleyballCourtConfiguration(), np.array([[-50, 450]])
    )
    assert np.array_equal(result[165, 15], np.array(sv.Color.RED.as_bgr()))


@pytest.mark.parametrize(
    "points",
    [
        np.array([]),
        np.array([1, 2]),
        np.empty((0, 3)),
        np.ones((2, 3)),
        np.array([[float("nan"), 0]]),
        np.array([[0, float("inf")]]),
    ],
)
def test_invalid_point_shapes_and_nonfinite_positions_are_rejected(
    points: npt.NDArray[np.float64],
) -> None:
    """Drawing rejects malformed arrays and nonfinite court positions."""
    with pytest.raises(ValueError):
        draw_points_on_court(VolleyballCourtConfiguration(), points)


@pytest.mark.parametrize("kwargs", [{"radius": 0}, {"radius": 1.5}, {"thickness": -1}])
def test_invalid_marker_parameters_are_rejected(kwargs: Dict[str, float]) -> None:
    """Marker radius and border thickness must be positive integers."""
    with pytest.raises(ValueError):
        draw_points_on_court(VolleyballCourtConfiguration(), np.empty((0, 2)), **kwargs)


def test_incompatible_existing_court_is_rejected() -> None:
    """A supplied court must match the renderer's shape and uint8 dtype."""
    with pytest.raises(ValueError):
        draw_points_on_court(
            VolleyballCourtConfiguration(),
            np.empty((0, 2)),
            court=np.zeros((331, 601, 3), dtype=np.float32),
        )
    with pytest.raises(ValueError):
        draw_points_on_court(
            VolleyballCourtConfiguration(),
            np.empty((0, 2)),
            court=np.zeros((10, 10, 3), dtype=np.uint8),
        )
