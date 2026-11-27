"""Black-box tests for ``ViewTransformer`` (homography between two planes).

Expected values come from an independent numpy projective-geometry oracle and from
algebraic properties (identity, translation, scale, round trip), not from OpenCV.
"""

import numpy as np
import pytest

from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

# float32 coordinates up to ~1e3 px carry ~1e-4 px of rounding error.
POINT_ATOL = 0.05

FRAME_CORNERS = np.array([[0, 0], [640, 0], [640, 360], [0, 360]], dtype=np.float32)


def project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to Nx2 points using plain numpy."""
    homogeneous = np.hstack([points, np.ones((len(points), 1))]) @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


@pytest.fixture
def homography() -> np.ndarray:
    """A hand-chosen perspective matrix (rotation/shear, shift and perspective)."""
    return np.array(
        [
            [1.2, 0.1, 30.0],
            [-0.05, 0.9, 12.0],
            [0.0004, 0.0002, 1.0],
        ]
    )


@pytest.fixture
def perspective_transformer(homography) -> ViewTransformer:
    """Transformer fitted on frame corners mapped through ``homography``."""
    target = project(homography, FRAME_CORNERS.astype(float)).astype(np.float32)
    return ViewTransformer(source=FRAME_CORNERS, target=target)


class TestConstruction:
    """Constructor validates the point sets it is given."""

    def test_rejects_source_and_target_of_different_shape(self):
        """Point sets with different counts cannot define a correspondence."""
        with pytest.raises(ValueError, match="same shape"):
            ViewTransformer(FRAME_CORNERS, FRAME_CORNERS[:3])

    def test_rejects_points_that_are_not_2d(self):
        """A homography between planes needs (x, y) pairs, not 3D points."""
        points_3d = np.zeros((4, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="2D"):
            ViewTransformer(points_3d, points_3d)

    def test_rejects_coincident_points(self):
        """Four identical points give no homography and raise the documented error."""
        same = np.zeros((4, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="could not be calculated"):
            ViewTransformer(same, same)

    def test_rejects_degenerate_collinear_points(self):
        """Four points on one line define no plane mapping, so construction fails.

        Silently building a singular matrix would map every later point to
        (0, 0), corrupting all downstream projections without any error.
        """
        line = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.float32)
        with pytest.raises(ValueError):
            ViewTransformer(line, line)

    def test_accepts_integer_point_arrays(self):
        """Callers often have integer pixel coordinates; they must be usable."""
        corners = FRAME_CORNERS.astype(np.int32)
        transformer = ViewTransformer(corners, corners)
        np.testing.assert_allclose(
            transformer.transform_points(np.array([[10, 20]])),
            [[10, 20]],
            atol=POINT_ATOL,
        )


class TestTransformPoints:
    """Points are mapped as the homography defined by the correspondences."""

    def test_maps_source_points_onto_their_targets(
        self, perspective_transformer, homography
    ):
        """The four fitted correspondences must be reproduced.

        This is the defining property of the transform.
        """
        expected = project(homography, FRAME_CORNERS.astype(float))
        result = perspective_transformer.transform_points(FRAME_CORNERS)
        np.testing.assert_allclose(result, expected, atol=POINT_ATOL)

    def test_matches_independent_projective_oracle_at_interior_points(
        self, perspective_transformer, homography
    ):
        """Points never used for fitting follow the same perspective mapping.

        Guards against a transform that only memorises the fitted corners.
        """
        inner = np.array(
            [[100, 50], [320, 180], [600, 300], [10, 350]], dtype=np.float32
        )
        result = perspective_transformer.transform_points(inner)
        np.testing.assert_allclose(
            result, project(homography, inner.astype(float)), atol=POINT_ATOL
        )

    def test_identity_correspondences_leave_points_unchanged(self):
        """Identical source and target describe the identity mapping."""
        transformer = ViewTransformer(FRAME_CORNERS, FRAME_CORNERS)
        points = np.array([[5, 6], [300, 200]], dtype=np.float32)
        np.testing.assert_allclose(
            transformer.transform_points(points), points, atol=POINT_ATOL
        )

    def test_translation_correspondences_shift_points(self):
        """Target = source + (15, -7) shifts every point by the same offset."""
        transformer = ViewTransformer(FRAME_CORNERS, [*FRAME_CORNERS, 15, -7])
        points = np.array([[5, 6], [300, 200]], dtype=np.float32)
        np.testing.assert_allclose(
            transformer.transform_points(points), [*points, 15, -7], atol=POINT_ATOL
        )

    def test_large_offset_translation_preserves_invertible_mapping(self):
        """Changing the target origin does not degenerate a valid homography."""
        corners = np.array([[0, 0], [1024, 0], [1024, 512], [0, 512]], dtype=np.float32)
        points = np.array([[256, 256], [768, 128]], dtype=np.float32)
        offset = np.array([2**27, 0], dtype=np.float32)
        transformer = ViewTransformer(corners, corners + offset)

        # The inputs and expected outputs are exactly representable in float32.
        np.testing.assert_allclose(
            transformer.transform_points(points),
            points + offset,
            rtol=0,
            atol=POINT_ATOL,
        )

    def test_scale_correspondences_scale_points(self):
        """Target = 0.5 * source halves every coordinate (frame to minimap)."""
        transformer = ViewTransformer(FRAME_CORNERS, FRAME_CORNERS * 0.5)
        points = np.array([[100, 40], [640, 360]], dtype=np.float32)
        np.testing.assert_allclose(
            transformer.transform_points(points), points * 0.5, atol=POINT_ATOL
        )

    def test_forward_then_inverse_round_trips(self, homography):
        """Swapping source/target gives the inverse mapping.

        Projecting to the target plane and back must return the start points.
        """
        target = project(homography, FRAME_CORNERS.astype(float)).astype(np.float32)
        forward = ViewTransformer(FRAME_CORNERS, target)
        backward = ViewTransformer(target, FRAME_CORNERS)
        points = np.array([[100, 50], [320, 180], [600, 300]], dtype=np.float32)
        np.testing.assert_allclose(
            backward.transform_points(forward.transform_points(points)),
            points,
            atol=POINT_ATOL,
        )

    @pytest.mark.parametrize("count", [1, 2, 7])
    def test_returns_one_float32_xy_row_per_input_point(
        self, perspective_transformer, count
    ):
        """Output keeps point count and order as float32 (N, 2)."""
        points = np.linspace([10, 10], [600, 300], count, dtype=np.float32)
        result = perspective_transformer.transform_points(points)
        assert (result.shape, result.dtype) == ((count, 2), np.float32)

    def test_point_order_is_preserved(self, perspective_transformer):
        """Row i of the output is the image of row i of the input."""
        points = np.array([[100, 50], [320, 180], [600, 300]], dtype=np.float32)
        batch = perspective_transformer.transform_points(points)
        singles = np.vstack(
            [perspective_transformer.transform_points(p[None]) for p in points]
        )
        np.testing.assert_allclose(batch, singles, atol=POINT_ATOL)

    def test_empty_input_returns_empty_output(self, perspective_transformer):
        """No detections in a frame is normal and must not raise."""
        result = perspective_transformer.transform_points(
            np.empty((0, 2), dtype=np.float32)
        )
        assert result.size == 0

    def test_rejects_points_that_are_not_2d(self, perspective_transformer):
        """Three-column input is not a plane coordinate and must be refused."""
        with pytest.raises(ValueError, match="2D"):
            perspective_transformer.transform_points(np.ones((2, 3), dtype=np.float32))


class TestTransformImage:
    """Images are warped by the same homography."""

    @staticmethod
    def _translation_transformer(dx: float, dy: float) -> ViewTransformer:
        return ViewTransformer(FRAME_CORNERS, [*FRAME_CORNERS, dx, dy])

    @pytest.mark.parametrize(
        "channels",
        [
            pytest.param((), id="grayscale"),
            pytest.param((3,), id="color"),
        ],
    )
    def test_output_has_requested_resolution_and_keeps_dtype(self, channels):
        """``resolution_wh`` is (width, height); output must be h x w (x channels)."""
        transformer = ViewTransformer(FRAME_CORNERS, FRAME_CORNERS)
        image = np.zeros((360, 640, *channels), dtype=np.uint8)
        result = transformer.transform_image(image, resolution_wh=(200, 100))
        assert (result.shape, result.dtype) == ((100, 200, *channels), np.uint8)

    def test_translation_moves_a_pixel_by_the_offset(self):
        """A white pixel at (x=10, y=20) lands at (x=15, y=27) after shift (5, 7)."""
        image = np.zeros((100, 100), dtype=np.uint8)
        image[20, 10] = 255
        result = self._translation_transformer(5, 7).transform_image(
            image, resolution_wh=(100, 100)
        )
        assert result[27, 15] == 255 and result.sum() == 255

    def test_scaling_up_enlarges_content(self):
        """With a 2x scale a 10x10 block at (10, 10) fills (20..40) in the output."""
        transformer = ViewTransformer(FRAME_CORNERS, FRAME_CORNERS * 2)
        image = np.zeros((100, 100), dtype=np.uint8)
        image[10:20, 10:20] = 255
        result = transformer.transform_image(image, resolution_wh=(100, 100))
        assert (result[25:35, 25:35] == 255).all() and (result[0:15, 0:15] == 0).all()

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param((10,), id="1d"),
            pytest.param((2, 10, 10, 3), id="4d"),
        ],
    )
    def test_rejects_images_that_are_not_grayscale_or_color(self, shape):
        """Only HxW and HxWxC images are supported."""
        transformer = ViewTransformer(FRAME_CORNERS, FRAME_CORNERS)
        with pytest.raises(ValueError, match="grayscale or color"):
            transformer.transform_image(
                np.zeros(shape, dtype=np.uint8), resolution_wh=(10, 10)
            )


class TestPitchProjection:
    """End-to-end use: map camera-frame positions to pitch coordinates."""

    def test_frame_positions_map_back_to_known_pitch_positions(self):
        """Fit frame->pitch from all 32 keypoints, then project players.

        The synthetic camera maps pitch centimetres to pixels; projecting the
        pixel positions back must recover the original pitch coordinates.
        """
        pitch = np.array(SoccerPitchConfiguration().vertices, dtype=float)
        camera = np.array(
            [
                [0.08, 0.02, 100.0],
                [0.0, 0.05, 80.0],
                [0.0, 0.00005, 1.0],
            ]
        )
        frame_keypoints = project(camera, pitch).astype(np.float32)
        transformer = ViewTransformer(
            source=frame_keypoints, target=pitch.astype(np.float32)
        )
        players_on_pitch = np.array(
            [[6000, 3500], [2500, 1200], [10500, 6000]], dtype=float
        )
        players_in_frame = project(camera, players_on_pitch).astype(np.float32)
        np.testing.assert_allclose(
            transformer.transform_points(players_in_frame),
            players_on_pitch,
            atol=1.0,  # 1 cm on a 120 m pitch
        )
