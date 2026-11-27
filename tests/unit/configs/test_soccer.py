"""Black-box tests for ``SoccerPitchConfiguration``.

Expectations come from the geometry of a soccer pitch and the documented meaning
of each field, never from the literal numbers inside the implementation.
"""
import re

import numpy as np
import pytest

from sports.configs.soccer import SoccerPitchConfiguration


@pytest.fixture(params=[
    pytest.param({}, id="default"),
    pytest.param({"width": 6400, "length": 10500}, id="small-pitch"),
    pytest.param(
        {"width": 7500, "length": 12000, "centre_circle_radius": 1000},
        id="wide-pitch-big-circle",
    ),
])
def config(request) -> SoccerPitchConfiguration:
    """Default configuration plus two custom pitch sizes."""
    return SoccerPitchConfiguration(**request.param)


def _vertex_set(config: SoccerPitchConfiguration) -> set:
    return {(round(float(x), 6), round(float(y), 6)) for x, y in config.vertices}


class TestKeypointMetadata:
    """Every pitch keypoint carries a label and a colour."""

    def test_has_32_keypoints_with_matching_labels_and_colors(self, config):
        """The keypoint model predicts 32 points; each needs a label and colour.

        A length mismatch would make annotators drop or misattribute keypoints.
        """
        assert (len(config.vertices), len(config.labels), len(config.colors)) == (
            32, 32, 32
        )

    def test_labels_are_unique(self, config):
        """Labels identify keypoints, so no two keypoints may share one."""
        assert len(set(config.labels)) == len(config.labels)

    def test_colors_are_hex_rgb_strings(self, config):
        """Colours are written as ``#RRGGBB`` so they can feed a colour parser."""
        assert all(re.fullmatch(r"#[0-9A-Fa-f]{6}", c) for c in config.colors)


class TestVertexGeometry:
    """Keypoint coordinates (in cm) describe a real, symmetric pitch."""

    def test_all_vertices_lie_within_the_pitch(self, config):
        """A keypoint outside the touch/goal lines cannot be a pitch marking."""
        v = np.array(config.vertices, dtype=float)
        assert v.shape == (32, 2)
        assert (v >= 0).all()
        assert (v[:, 0] <= config.length).all() and (v[:, 1] <= config.width).all()

    def test_vertices_are_distinct(self, config):
        """Two keypoints on the same spot would make the homography degenerate."""
        assert len(_vertex_set(config)) == len(config.vertices)

    def test_four_pitch_corners_are_keypoints(self, config):
        """Corners are the most reliable landmarks and must be present."""
        corners = {
            (0.0, 0.0), (0.0, float(config.width)),
            (float(config.length), 0.0), (float(config.length), float(config.width)),
        }
        assert corners <= _vertex_set(config)

    @pytest.mark.parametrize("axis", [
        pytest.param(0, id="halfway-line-mirror"),
        pytest.param(1, id="long-axis-mirror"),
    ])
    def test_pitch_is_mirror_symmetric(self, config, axis):
        """Reflecting the keypoints across the halfway line or long axis maps onto itself.

        Soccer pitches are symmetric about both axes; an asymmetric set points to
        a mistyped coordinate.
        """
        v = np.array(config.vertices, dtype=float)
        mirrored = v.copy()
        mirrored[:, axis] = [config.length, config.width][axis] - v[:, axis]
        mirrored_set = {(round(x, 6), round(y, 6)) for x, y in mirrored.tolist()}
        assert mirrored_set == _vertex_set(config)

    def test_penalty_spots_sit_on_long_axis_at_penalty_distance(self, config):
        """Each penalty spot is centred across the width, 11 m-style from its goal line."""
        centre_y = config.width / 2
        expected = {
            (float(config.penalty_spot_distance), centre_y),
            (float(config.length - config.penalty_spot_distance), centre_y),
        }
        assert expected <= _vertex_set(config)

    def test_centre_circle_meets_halfway_line_and_long_axis(self, config):
        """The centre circle touches the halfway line and long axis at +-radius."""
        mid, centre_y, r = config.length / 2, config.width / 2, config.centre_circle_radius
        expected = {
            (mid, centre_y - r), (mid, centre_y + r),
            (mid - r, centre_y), (mid + r, centre_y),
        }
        assert expected <= _vertex_set(config)

    def test_goal_line_carries_box_edges_centred_on_the_goal(self, config):
        """On the goal line: corner, penalty-box edges, goal-box edges, corner.

        Both boxes are centred on the goal, so their edges are equidistant from
        the pitch centre line.
        """
        v = np.array(config.vertices, dtype=float)
        ys = np.sort(v[v[:, 0] == 0][:, 1])
        half_w = config.width / 2
        expected = [
            0,
            half_w - config.penalty_box_width / 2,
            half_w - config.goal_box_width / 2,
            half_w + config.goal_box_width / 2,
            half_w + config.penalty_box_width / 2,
            config.width,
        ]
        np.testing.assert_allclose(ys, expected)

    def test_goal_box_nests_inside_penalty_box_inside_pitch(self, config):
        """Default dimensions respect the nesting goal box < penalty box < pitch."""
        assert 0 < config.goal_box_width < config.penalty_box_width < config.width
        assert (
            0 < config.goal_box_length < config.penalty_box_length < config.length / 2
        )

    def test_dimensions_propagate_to_vertices(self):
        """Changing pitch size must change the extreme keypoint coordinates."""
        small = SoccerPitchConfiguration(width=6000, length=10000)
        v = np.array(small.vertices, dtype=float)
        assert (v[:, 0].max(), v[:, 1].max()) == (10000, 6000)


class TestEdges:
    """Edges connect 1-based keypoint indices into the pitch line drawing."""

    def test_edges_reference_valid_one_based_indices(self, config):
        """An index of 0 or 33 would silently wrap or crash when drawing."""
        e = np.array(config.edges)
        assert e.min() >= 1 and e.max() <= len(config.vertices)

    def test_edges_have_no_self_loops_or_duplicates(self, config):
        """A pitch line joins two different points and is listed once."""
        undirected = {frozenset(edge) for edge in config.edges}
        assert (
            all(a != b for a, b in config.edges)
            and len(undirected) == len(config.edges)
        )

    def test_every_edge_is_horizontal_or_vertical(self, config):
        """Pitch lines run parallel to a pitch axis; a diagonal means a wrong pair."""
        v = np.array(config.vertices, dtype=float)
        e = np.array(config.edges) - 1
        start, end = v[e[:, 0]], v[e[:, 1]]
        aligned = (start[:, 0] == end[:, 0]) | (start[:, 1] == end[:, 1])
        assert aligned.all()

    def test_only_spots_and_circle_points_are_off_straight_lines(self, config):
        """Only penalty spots (9, 22) and centre-circle side points (31, 32) lack an edge.

        These four are a dot and the curved circle's extremes; every other
        keypoint is a corner of a straight pitch line and must be connected.
        """
        used = {i for edge in config.edges for i in edge}
        assert set(range(1, 33)) - used == {9, 22, 31, 32}

    def test_pitch_outline_is_closed(self, config):
        """Touch lines and goal lines form a closed loop around the pitch."""
        outline = {
            frozenset(e) for e in [
                (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),  # left goal line
                (6, 17), (17, 30),  # bottom touch line
                (30, 29), (29, 28), (28, 27), (27, 26), (26, 25),  # right goal line
                (25, 14), (14, 1),  # top touch line
            ]
        }
        assert outline <= {frozenset(edge) for edge in config.edges}


class TestInstanceIsolation:
    """Each configuration instance owns its lists."""

    def test_default_lists_are_not_shared_between_instances(self):
        """Mutating one config's lists must not leak into another instance.

        Shared mutable defaults are a classic dataclass bug that corrupts every
        later pitch drawing in the same process.
        """
        first, second = SoccerPitchConfiguration(), SoccerPitchConfiguration()
        first.edges.append((1, 30))
        first.labels.append("zz")
        first.colors.append("#000000")
        assert (
            (1, 30) not in second.edges
            and "zz" not in second.labels
            and len(second.colors) == 32
        )
