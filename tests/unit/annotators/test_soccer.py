"""Black-box tests for the soccer pitch drawing helpers.

Pixel positions are derived from physical pitch size (cm), ``scale`` (px per cm) and
``padding`` (px), so a test only trusts the documented geometry, never golden images.
Colours are ``sv.Color(r, g, b)``; OpenCV images are BGR.
"""
import numpy as np
import pytest
import supervision as sv

from sports.annotators.soccer import (
    draw_paths_on_pitch,
    draw_pitch,
    draw_pitch_voronoi_diagram,
    draw_points_on_pitch,
)
from sports.configs.soccer import SoccerPitchConfiguration

BGR_RED = (0, 0, 255)
BGR_BLUE = (255, 0, 0)
BGR_DEFAULT_GRASS = (34, 139, 34)
BGR_WHITE = (255, 255, 255)
PAD = 50  # default padding in px
S = 0.1  # default scale in px per cm


@pytest.fixture
def config() -> SoccerPitchConfiguration:
    """Default 120 m x 70 m pitch; at scale 0.1 the field is 1200 x 700 px."""
    return SoccerPitchConfiguration()


def px(x_cm: float, y_cm: float, padding: int = PAD, scale: float = S) -> tuple:
    """Pixel (x, y) of a pitch coordinate given in cm."""
    return int(x_cm * scale) + padding, int(y_cm * scale) + padding


def colour_at(image: np.ndarray, x: int, y: int) -> tuple:
    return tuple(int(c) for c in image[y, x])


class TestDrawPitch:
    """``draw_pitch`` renders the pitch markings on a coloured background."""

    @pytest.mark.parametrize("scale, padding", [
        pytest.param(0.1, 50, id="defaults"),
        pytest.param(0.05, 20, id="half-scale-small-padding"),
        pytest.param(0.2, 0, id="double-scale-no-padding"),
    ])
    def test_image_size_follows_pitch_scale_and_padding(self, config, scale, padding):
        """Height = width*scale + 2*padding, width = length*scale + 2*padding, 3 channels."""
        image = draw_pitch(config, scale=scale, padding=padding)
        expected = (
            int(config.width * scale) + 2 * padding,
            int(config.length * scale) + 2 * padding,
            3,
        )
        assert (image.shape, image.dtype) == (expected, np.uint8)

    def test_custom_pitch_dimensions_change_image_size(self):
        """A smaller pitch configuration yields a proportionally smaller image."""
        small = SoccerPitchConfiguration(width=6000, length=10000)
        assert draw_pitch(small).shape == (600 + 2 * PAD, 1000 + 2 * PAD, 3)

    def test_padding_area_has_the_background_colour(self, config):
        """The margin around the pitch is filled with ``background_color``."""
        image = draw_pitch(config, background_color=sv.Color(r=10, g=20, b=30))
        corners = [(0, 0), (image.shape[1] - 1, 0), (0, image.shape[0] - 1)]
        assert {colour_at(image, x, y) for x, y in corners} == {(30, 20, 10)}

    def test_default_background_is_grass_green(self, config):
        """Without a colour argument the margin is grass green (34, 139, 34 RGB)."""
        assert colour_at(draw_pitch(config), 0, 0) == BGR_DEFAULT_GRASS

    @pytest.mark.parametrize("x_cm, y_cm", [
        pytest.param(0, 0, id="top-left-corner"),
        pytest.param(12000, 7000, id="bottom-right-corner"),
        pytest.param(6000, 0, id="middle-of-top-touch-line"),
        pytest.param(6000, 3500, id="middle-of-halfway-line"),
        pytest.param(0, 3500, id="middle-of-left-goal-line"),
        pytest.param(1100, 3500, id="left-penalty-spot"),
        pytest.param(12000 - 1100, 3500, id="right-penalty-spot"),
        pytest.param(6000 + 915, 3500, id="centre-circle-right-point"),
    ])
    def test_pitch_markings_are_drawn_in_line_colour(self, config, x_cm, y_cm):
        """Corners, touch/goal/halfway lines, spots and the centre circle are drawn.

        Each case is a point that must lie on a marking, given in pitch cm.
        """
        x, y = px(x_cm, y_cm)
        assert colour_at(draw_pitch(config), x, y) == BGR_WHITE

    @pytest.mark.parametrize("x_cm, y_cm", [
        pytest.param(3000, 1000, id="open-grass-left-of-centre"),
        pytest.param(9000, 6000, id="open-grass-right-of-centre"),
        pytest.param(6000 + 400, 3500 - 400, id="inside-centre-circle"),
    ])
    def test_open_grass_is_left_as_background(self, config, x_cm, y_cm):
        """Areas that hold no marking keep the background colour."""
        x, y = px(x_cm, y_cm)
        assert colour_at(draw_pitch(config), x, y) == BGR_DEFAULT_GRASS

    def test_line_and_background_colours_are_configurable(self, config):
        """Custom colours replace both the lines and the grass."""
        image = draw_pitch(
            config, background_color=sv.Color.BLUE, line_color=sv.Color.RED
        )
        x, y = px(0, 0)
        assert (colour_at(image, 0, 0), colour_at(image, x, y)) == (BGR_RED[::-1], BGR_RED)

    def test_thicker_lines_cover_more_pixels(self, config):
        """``line_thickness`` widens the markings."""
        thin = draw_pitch(config, line_thickness=2)
        thick = draw_pitch(config, line_thickness=8)
        white = np.array(BGR_WHITE)
        assert (thick == white).all(axis=2).sum() > (thin == white).all(axis=2).sum()

    def test_repeated_calls_return_independent_equal_images(self, config):
        """Drawing is deterministic and one call's image is not shared with the next."""
        first, second = draw_pitch(config), draw_pitch(config)
        first[:] = 0
        assert np.array_equal(second, draw_pitch(config))


class TestDrawPointsOnPitch:
    """``draw_points_on_pitch`` marks pitch positions (cm) on the image."""

    @pytest.mark.parametrize("x_cm, y_cm, scale", [
        pytest.param(6000, 3500, 0.1, id="centre-default-scale"),
        pytest.param(0, 0, 0.1, id="origin"),
        pytest.param(12000, 7000, 0.1, id="far-corner"),
        pytest.param(6000, 3500, 0.05, id="centre-half-scale"),
    ])
    def test_point_is_filled_with_face_colour_at_its_scaled_position(
        self, config, x_cm, y_cm, scale
    ):
        """Pitch position (cm) maps to pixel ``cm*scale + padding``."""
        result = draw_points_on_pitch(
            config, np.array([[x_cm, y_cm]]), face_color=sv.Color.RED, scale=scale
        )
        x, y = px(x_cm, y_cm, scale=scale)
        assert colour_at(result, x, y) == BGR_RED

    def test_point_has_an_edge_ring_in_edge_colour(self, config):
        """The marker outline (radius 10) is drawn in ``edge_color``."""
        result = draw_points_on_pitch(
            config, np.array([[6000, 3500]]), edge_color=sv.Color.BLUE, radius=10
        )
        x, y = px(6000, 3500)
        ring = result[y, x + 8:x + 13]
        assert (ring == BGR_BLUE).all(axis=1).any()

    def test_every_point_is_drawn(self, config):
        """All given positions are marked, not only the first."""
        xy = np.array([[2000, 1000], [6000, 3500], [10000, 6000]])
        result = draw_points_on_pitch(config, xy, face_color=sv.Color.RED)
        xs, ys = zip(*[px(*p) for p in xy])
        assert (result[list(ys), list(xs)] == BGR_RED).all()

    def test_no_points_returns_the_plain_pitch(self, config):
        """An empty position list draws nothing and returns the bare pitch."""
        result = draw_points_on_pitch(config, np.empty((0, 2)))
        assert np.array_equal(result, draw_pitch(config))

    def test_draws_on_the_supplied_pitch_image(self, config):
        """With ``pitch=...`` the points go onto that image and its content is kept."""
        canvas = np.full((100, 100, 3), 7, dtype=np.uint8)
        result = draw_points_on_pitch(
            config, np.array([[0, 0]]), face_color=sv.Color.RED, pitch=canvas
        )
        assert (result.shape, colour_at(result, 0, 0), colour_at(result, 50, 50)) == (
            (100, 100, 3), (7, 7, 7), BGR_RED
        )


class TestDrawPathsOnPitch:
    """``draw_paths_on_pitch`` connects consecutive positions with lines."""

    @pytest.fixture
    def corner_path(self) -> np.ndarray:
        """An L-shaped path in open grass: (1000,1000) -> (3000,1000) -> (3000,3000) cm."""
        return np.array([[1000, 1000], [3000, 1000], [3000, 3000]])

    def test_segments_between_consecutive_points_are_drawn(self, config, corner_path):
        """The midpoint of each segment carries the path colour."""
        result = draw_paths_on_pitch(config, [corner_path], color=sv.Color.RED)
        first_mid, second_mid = px(2000, 1000), px(3000, 2000)
        assert (colour_at(result, *first_mid), colour_at(result, *second_mid)) == (
            BGR_RED, BGR_RED
        )

    def test_path_colour_is_not_applied_off_the_path(self, config, corner_path):
        """Grass away from the path stays untouched."""
        result = draw_paths_on_pitch(config, [corner_path], color=sv.Color.RED)
        assert colour_at(result, *px(9000, 6000)) == BGR_DEFAULT_GRASS

    def test_every_path_is_drawn(self, config, corner_path):
        """Two paths in one call are both rendered.

        The docstring says all given paths are drawn on the pitch; a second
        team member's trajectory vanishing is the failure this guards.
        """
        second = np.array([[8000, 5000], [10000, 5000]])
        result = draw_paths_on_pitch(
            config, [corner_path, second], color=sv.Color.RED
        )
        assert (
            colour_at(result, *px(2000, 1000)),
            colour_at(result, *px(9000, 5000)),
        ) == (BGR_RED, BGR_RED)

    @pytest.mark.parametrize("paths", [
        pytest.param([], id="no-paths"),
        pytest.param([np.array([[1000, 1000]])], id="single-point-path"),
    ])
    def test_returns_the_plain_pitch_when_nothing_is_drawable(self, config, paths):
        """A pitch image is always returned, even if there is nothing to draw."""
        result = draw_paths_on_pitch(config, paths)
        assert isinstance(result, np.ndarray) and np.array_equal(
            result, draw_pitch(config)
        )


class TestDrawPitchVoronoiDiagram:
    """Each pixel is coloured by the team whose nearest player is closest."""

    @pytest.fixture
    def left_player(self) -> np.ndarray:
        """Team 1 player at (3000, 3500) cm."""
        return np.array([[3000, 3500]])

    @pytest.fixture
    def right_player(self) -> np.ndarray:
        """Team 2 player at (9000, 3500) cm."""
        return np.array([[9000, 3500]])

    @pytest.mark.parametrize("x_px, y_px, expected", [
        pytest.param(150, 100, BGR_RED, id="near-team-1"),
        pytest.param(1100, 700, BGR_BLUE, id="near-team-2"),
        pytest.param(0, 0, BGR_RED, id="padding-corner-nearest-to-team-1"),
        pytest.param(640, 400, BGR_RED, id="just-left-of-bisector"),
        pytest.param(660, 400, BGR_BLUE, id="just-right-of-bisector"),
    ])
    def test_pixel_takes_the_colour_of_the_nearest_team(
        self, config, left_player, right_player, x_px, y_px, expected
    ):
        """At full opacity each pixel is purely the nearest team's colour.

        The perpendicular bisector of the two players is at x = 6000 cm, i.e.
        pixel 650 with default padding and scale.
        """
        result = draw_pitch_voronoi_diagram(
            config, left_player, right_player,
            team_1_color=sv.Color.RED, team_2_color=sv.Color.BLUE, opacity=1.0,
        )
        assert colour_at(result, x_px, y_px) == expected

    def test_closest_player_of_a_team_decides_with_several_players(self, config):
        """A team's area is the union of its players' areas.

        Team 1 stands at both ends, team 2 in the middle: the far right is
        team 1's, the centre is team 2's.
        """
        team_1 = np.array([[1000, 3500], [11000, 3500]])
        team_2 = np.array([[6000, 3500]])
        result = draw_pitch_voronoi_diagram(
            config, team_1, team_2,
            team_1_color=sv.Color.RED, team_2_color=sv.Color.BLUE, opacity=1.0,
        )
        assert (colour_at(result, *px(11500, 3500)), colour_at(result, *px(6000, 3500))) == (
            BGR_RED, BGR_BLUE
        )

    def test_zero_opacity_returns_the_plain_pitch(self, config, left_player, right_player):
        """With opacity 0 the overlay is invisible."""
        result = draw_pitch_voronoi_diagram(
            config, left_player, right_player, opacity=0.0
        )
        assert np.array_equal(result, draw_pitch(config))

    def test_half_opacity_blends_team_colour_with_the_pitch(
        self, config, left_player, right_player
    ):
        """On open grass, opacity 0.5 gives the mean of team colour and grass."""
        result = draw_pitch_voronoi_diagram(
            config, left_player, right_player, team_1_color=sv.Color.RED, opacity=0.5
        )
        expected = 0.5 * np.array(BGR_RED) + 0.5 * np.array(BGR_DEFAULT_GRASS)
        np.testing.assert_allclose(result[100, 150], expected, atol=1.0)

    def test_output_has_the_same_size_as_the_pitch(self, config, left_player, right_player):
        """The overlay result is a full pitch-sized uint8 image."""
        result = draw_pitch_voronoi_diagram(config, left_player, right_player)
        assert (result.shape, result.dtype) == (draw_pitch(config).shape, np.uint8)
