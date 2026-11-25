"""Render an indoor volleyball court and tracked player positions."""

from math import isfinite
from numbers import Integral, Real
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import supervision as sv

from sports.configs.volleyball import VolleyballCourtConfiguration

if TYPE_CHECKING:
    from sports.common.calibration import CourtCalibration


def _canvas_shape(
    config: VolleyballCourtConfiguration, padding: int, scale: float
) -> Tuple[int, int, int]:
    """Validate rendering dimensions and include both court boundary endpoints."""
    config.validate()
    if isinstance(padding, bool) or not isinstance(padding, Integral) or padding < 0:
        raise ValueError("padding must be a nonnegative integer.")
    if (
        isinstance(scale, bool)
        or not isinstance(scale, Real)
        or not isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("scale must be a finite positive number.")
    scaled_length = config.length * scale
    scaled_width = config.width * scale
    if not isfinite(scaled_length) or not isfinite(scaled_width):
        raise ValueError("Scaled court dimensions must be finite.")
    if scaled_length < 1 or scaled_width < 1:
        raise ValueError(
            "scale must render each court dimension at least one pixel wide."
        )
    # Include both end points, even when padding is zero.
    width = int(scaled_length) + 2 * int(padding) + 1
    height = int(scaled_width) + 2 * int(padding) + 1
    if height * width > 25_000_000:
        raise ValueError("Court preview must contain at most 25 million pixels.")
    return height, width, 3


def draw_court(
    config: VolleyballCourtConfiguration, padding: int = 30, scale: float = 0.3
) -> npt.NDArray[np.uint8]:
    """Draw boundaries, the center/net line, and both attack lines.

    Args:
        config: Court dimensions and one-based landmark edges, in centimeters.
        padding: Nonnegative border around the court in pixels. Defaults to 30.
        scale: Finite positive pixels per centimeter. Defaults to 0.3.

    Returns:
        New uint8 BGR image. Its shape is
        (int(width * scale) + 2 * padding + 1,
        int(length * scale) + 2 * padding + 1, 3). The extra pixel keeps both
        boundary endpoints visible, including with zero padding.

    Raises:
        ValueError: If court dimensions, padding, or scale are invalid, scaled
            dimensions are less than one pixel, or the image exceeds 25 million
            pixels.

    Example:
        >>> court = draw_court(VolleyballCourtConfiguration())
        >>> court.shape
        (331, 601, 3)
    """
    shape = _canvas_shape(config, padding, scale)
    offset = int(padding)
    court = np.empty(shape, dtype=np.uint8)
    court[:] = sv.Color.from_hex("#263747").as_bgr()
    vertices = config.vertices
    for start, end in config.edges:
        source = vertices[start - 1]
        target = vertices[end - 1]
        cv2.line(
            court,
            (int(source[0] * scale) + offset, int(source[1] * scale) + offset),
            (int(target[0] * scale) + offset, int(target[1] * scale) + offset),
            sv.Color.WHITE.as_bgr(),
            thickness=2,
        )
    return court


def draw_points_on_court(
    config: VolleyballCourtConfiguration,
    xy: npt.ArrayLike,
    face_color: sv.Color = sv.Color.RED,
    edge_color: sv.Color = sv.Color.BLACK,
    radius: int = 6,
    thickness: int = 2,
    padding: int = 30,
    scale: float = 0.3,
    court: Optional[npt.NDArray[np.uint8]] = None,
) -> npt.NDArray[np.uint8]:
    """Draw finite ``(N, 2)`` court positions, modifying a supplied image in place.

    Points in the visible free zone are retained. Points entirely outside the
    image are skipped, so detections beyond the court do not pile up at an edge.

    Args:
        config: Court geometry in centimeters, used for canvas dimensions.
        xy: Finite court (x, y) coordinates with shape (N, 2). Empty arrays must
            have shape (0, 2).
        face_color: Marker fill color. Defaults to red.
        edge_color: Marker border color. Defaults to black.
        radius: Positive integer marker radius in pixels. Defaults to 6.
        thickness: Positive integer border thickness in pixels. Defaults to 2.
        padding: Nonnegative court border in pixels. Defaults to 30.
        scale: Finite positive pixels per centimeter. Defaults to 0.3.
        court: Optional uint8 BGR court image with the configured canvas shape.
            A new court is drawn when omitted; a supplied image is modified.

    Returns:
        Court image containing the markers. When court is supplied, the returned
        array is the same object.

    Raises:
        ValueError: If geometry, rendering parameters, point shape or values,
            marker dimensions, or the supplied image shape/dtype are invalid.

    Example:
        >>> config = VolleyballCourtConfiguration()
        >>> court = draw_court(config)
        >>> result = draw_points_on_court(config, [[450, 450]], court=court)
        >>> result is court
        True
    """
    shape = _canvas_shape(config, padding, scale)
    for name, value in (("radius", radius), ("thickness", thickness)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    try:
        points = np.asarray(xy, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("xy must be a numeric array of shape (N, 2).") from error
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("xy must have shape (N, 2), including when empty.")
    if not np.isfinite(points).all():
        raise ValueError("xy coordinates must be finite.")
    if court is None:
        court = draw_court(config, padding=padding, scale=scale)
    elif (
        not isinstance(court, np.ndarray)
        or court.dtype != np.uint8
        or court.shape != shape
    ):
        raise ValueError(
            "court must be a uint8 BGR image matching the configured canvas."
        )

    height, width = court.shape[:2]
    margin = int(radius) + int(thickness)
    for x, y in points:
        center_x = float(x) * scale + padding
        center_y = float(y) * scale + padding
        if (
            center_x < -margin
            or center_x >= width + margin
            or center_y < -margin
            or center_y >= height + margin
        ):
            continue
        center = (int(center_x), int(center_y))
        cv2.circle(court, center, int(radius), face_color.as_bgr(), thickness=-1)
        cv2.circle(
            court, center, int(radius), edge_color.as_bgr(), thickness=int(thickness)
        )
    return court


def annotate_court(
    frame: npt.NDArray[np.uint8], calibration: "CourtCalibration"
) -> npt.NDArray[np.uint8]:
    """Project court lines and corner labels onto a calibrated source frame.

    Args:
        frame: Source uint8 BGR image, modified in place. Its resolution and
            camera view must match the supplied calibration.
        calibration: Four-corner source-to-court calibration. Configuration
            vertices and one-based edges specify the lines to project.

    Returns:
        The supplied frame with yellow court lines and A, B, C, D labels.

    Raises:
        numpy.linalg.LinAlgError: If the calibration homography is singular.
        cv2.error: If the image or projected geometry cannot be drawn.
    """
    pixels = calibration.corners.astype(int)
    vertices = np.asarray(calibration.config.vertices, dtype=np.float32).reshape(
        -1, 1, 2
    )
    projected = cv2.perspectiveTransform(
        vertices, np.linalg.inv(calibration.transformer.m)
    ).reshape(-1, 2)
    for start, end in calibration.config.edges:
        first, second = np.round(projected[[start - 1, end - 1]]).astype(int)
        cv2.line(frame, tuple(first), tuple(second), (0, 255, 255), 2)
    for label, point in zip("ABCD", pixels):
        cv2.putText(
            frame, label, tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )
    return frame


def render_radar(
    frame: npt.NDArray[np.uint8],
    court_xy: npt.ArrayLike,
    teams: npt.ArrayLike,
    config: Optional[VolleyballCourtConfiguration] = None,
    team_colors: Optional[Sequence[str]] = None,
) -> npt.NDArray[np.uint8]:
    """Blend a court map and team-colored positions into the bottom of a frame.

    The map is centered horizontally and fitted within 45% of the source width
    and 35% of its height. Positions beyond the visible map are skipped.

    Args:
        frame: Source uint8 BGR image, modified in place.
        court_xy: Finite court (x, y) positions with shape (N, 2), in centimeters.
        teams: Team IDs with shape (N,), aligned with court_xy. Color indices
            start at zero; IDs outside the color sequence are not displayed.
        config: Court geometry. Defaults to the standard indoor volleyball court.
        team_colors: Hexadecimal fill colors indexed by team ID. Defaults to
            pink (#FF1493) for team zero and blue (#00BFFF) for team one.

    Returns:
        The supplied frame with a court map blended at 85% map opacity.

    Raises:
        ValueError: If positions are nonnumeric or nonfinite, point/team shapes
            do not align, or court rendering fails.
        cv2.error: If the source frame or map cannot be resized or blended.
    """
    if config is None:
        config = VolleyballCourtConfiguration()
    if team_colors is None:
        team_colors = ("#FF1493", "#00BFFF")
    try:
        points = np.asarray(court_xy, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "court_xy must contain numeric positions of shape (N, 2)."
        ) from error
    team_ids = np.asarray(teams)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("court_xy must have shape (N, 2), including when empty.")
    if not np.isfinite(points).all():
        raise ValueError("court_xy coordinates must be finite.")
    if team_ids.ndim != 1 or len(team_ids) != len(points):
        raise ValueError("teams must have shape (N,) aligned with court_xy.")
    court = draw_court(config)
    for team, color in enumerate(team_colors):
        court = draw_points_on_court(
            config,
            points[team_ids == team],
            face_color=sv.Color.from_hex(color),
            court=court,
        )
    height, width = frame.shape[:2]
    scale = min(width * 0.45 / court.shape[1], height * 0.35 / court.shape[0])
    court = cv2.resize(
        court,
        (max(1, round(court.shape[1] * scale)), max(1, round(court.shape[0] * scale))),
    )
    court_height, court_width = court.shape[:2]
    x, y = (width - court_width) // 2, height - court_height
    region = frame[y : y + court_height, x : x + court_width]
    frame[y : y + court_height, x : x + court_width] = cv2.addWeighted(
        region, 0.15, court, 0.85, 0
    )
    return frame
