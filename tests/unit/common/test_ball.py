"""Black-box tests for ``BallTracker`` and ``BallAnnotator``.

Expectations follow the documented intent: the tracker keeps the one detection
nearest to the recent trajectory; the annotator draws a fading trail of circles.
"""
import numpy as np
import pytest
import supervision as sv

from sports.common.ball import BallAnnotator, BallTracker


def ball_at(*centres: tuple, confidence: tuple = ()) -> sv.Detections:
    """Detections whose 4x4 boxes are centred on the given (x, y) points."""
    xyxy = np.array(
        [[x - 2, y - 2, x + 2, y + 2] for x, y in centres], dtype=float
    ).reshape(-1, 4)
    kwargs = {"confidence": np.array(confidence, dtype=float)} if confidence else {}
    return sv.Detections(xyxy=xyxy, **kwargs)


def centres_of(detections: sv.Detections) -> np.ndarray:
    return detections.get_anchors_coordinates(sv.Position.CENTER)


class TestBallTracker:
    """The tracker picks the detection that best fits the recent trajectory."""

    def test_returns_exactly_one_detection_out_of_many(self):
        """Several ball candidates in a frame collapse to a single ball."""
        tracker = BallTracker()
        result = tracker.update(ball_at((100, 100), (300, 300), (50, 400)))
        assert len(result) == 1

    def test_prefers_candidate_near_recent_positions_over_a_distant_one(self):
        """With history at (100, 100), a candidate at (103, 100) beats one at (500, 400).

        A far-away false positive must not hijack a ball that has been tracked
        steadily in one place.
        """
        tracker = BallTracker()
        for _ in range(3):
            tracker.update(ball_at((100, 100)))
        result = tracker.update(ball_at((500, 400), (103, 100)))
        np.testing.assert_allclose(centres_of(result), [[103, 100]])

    def test_returned_detection_keeps_its_own_attributes(self):
        """The selected detection carries its own confidence, not another's."""
        tracker = BallTracker()
        for _ in range(3):
            tracker.update(ball_at((100, 100)))
        result = tracker.update(
            ball_at((500, 400), (103, 100), confidence=(0.9, 0.3))
        )
        np.testing.assert_allclose(result.confidence, [0.3])

    def test_single_detection_is_returned_as_is(self):
        """With one candidate there is nothing to choose; it is returned."""
        tracker = BallTracker()
        result = tracker.update(ball_at((42, 17)))
        np.testing.assert_allclose(centres_of(result), [[42, 17]])

    def test_empty_detections_are_returned_empty(self):
        """A frame without a ball yields no detection and no error."""
        tracker = BallTracker()
        result = tracker.update(sv.Detections.empty())
        assert len(result) == 0

    def test_tracking_resumes_after_a_frame_without_ball(self):
        """A missed detection must not break tracking on later frames."""
        tracker = BallTracker()
        tracker.update(ball_at((100, 100)))
        tracker.update(sv.Detections.empty())
        result = tracker.update(ball_at((500, 400), (102, 101)))
        np.testing.assert_allclose(centres_of(result), [[102, 101]])

    def test_old_positions_leave_the_window(self):
        """Only the last ``buffer_size`` frames influence the choice.

        After the ball moved from (0, 0) to (1000, 1000) for longer than the
        window, the old location is forgotten, so (1000, 1000) wins even though
        (0, 0) is listed first. An unbounded history would tie and prefer (0, 0).
        """
        tracker = BallTracker(buffer_size=3)
        for _ in range(3):
            tracker.update(ball_at((0, 0)))
        for _ in range(3):
            tracker.update(ball_at((1000, 1000)))
        result = tracker.update(ball_at((0, 0), (1000, 1000)))
        np.testing.assert_allclose(centres_of(result), [[1000, 1000]])


class TestBallAnnotatorRadius:
    """The trail radius grows from 1 (oldest) to ``radius`` (newest)."""

    @pytest.mark.parametrize("i, max_i, expected", [
        pytest.param(0, 5, 1, id="oldest-is-1"),
        pytest.param(4, 5, 10, id="newest-is-max-radius"),
        pytest.param(0, 1, 10, id="single-entry-is-max-radius"),
    ])
    def test_endpoints_of_the_interpolation(self, i, max_i, expected):
        """Oldest trail entry has radius 1, newest has the configured radius."""
        assert BallAnnotator(radius=10).interpolate_radius(i, max_i) == expected

    def test_radius_never_shrinks_toward_newer_entries(self):
        """Newer entries are drawn at least as large as older ones."""
        annotator = BallAnnotator(radius=10)
        radii = [annotator.interpolate_radius(i, 6) for i in range(6)]
        assert radii == sorted(radii)

    def test_radius_stays_between_one_and_max(self):
        """Interpolated radii are always within [1, radius]."""
        annotator = BallAnnotator(radius=12)
        radii = [annotator.interpolate_radius(i, 9) for i in range(9)]
        assert min(radii) >= 1 and max(radii) <= 12


class TestBallAnnotatorAnnotate:
    """Annotating draws circles around the ball's bottom-centre."""

    @pytest.fixture
    def blank(self) -> np.ndarray:
        """A black 200x200 BGR frame."""
        return np.zeros((200, 200, 3), dtype=np.uint8)

    def test_draws_a_ring_around_the_bottom_centre_of_the_box(self, blank):
        """A box ending at y=100 gets a ring centred on (100, 100), hollow inside.

        Bottom-centre (where the ball touches the ground) is the anchor; the
        circle is an outline, so the centre pixel stays untouched.
        """
        detections = ball_at((100, 98))  # box x 98..102, bottom y = 100
        result = BallAnnotator(radius=10).annotate(blank, detections)
        assert result[100, 110].any() and not result[100, 100].any()

    def test_returns_a_frame_of_the_same_shape_and_dtype(self, blank):
        """Annotation never changes frame geometry or pixel type."""
        result = BallAnnotator(radius=10).annotate(blank, ball_at((100, 98)))
        assert (result.shape, result.dtype) == ((200, 200, 3), np.uint8)

    def test_no_detections_leave_the_frame_untouched(self, blank):
        """A frame without a ball gets no marks."""
        result = BallAnnotator(radius=10).annotate(blank.copy(), sv.Detections.empty())
        assert not result.any()

    def test_previous_positions_are_drawn_as_a_trail(self, blank):
        """After the ball moves, its earlier location is still marked on a new frame."""
        annotator = BallAnnotator(radius=10, buffer_size=5)
        annotator.annotate(blank.copy(), ball_at((50, 98)))
        result = annotator.annotate(blank.copy(), ball_at((150, 98)))
        old_spot = result[99:102, 49:52]
        assert old_spot.any()

    def test_positions_older_than_the_buffer_are_forgotten(self, blank):
        """With ``buffer_size=2`` the third frame no longer shows the first position."""
        annotator = BallAnnotator(radius=10, buffer_size=2)
        annotator.annotate(blank.copy(), ball_at((20, 98)))
        annotator.annotate(blank.copy(), ball_at((100, 98)))
        result = annotator.annotate(blank.copy(), ball_at((180, 98)))
        assert not result[95:106, 15:26].any()
