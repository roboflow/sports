"""Persist and apply a four-corner calibration for a stationary camera."""

import json
from pathlib import Path
from typing import Protocol, Sequence, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt

from sports.common.view import ViewTransformer


class CourtGeometry(Protocol):
    """Describe the rectangular target geometry needed by a court calibration.

    Attributes:
        length: Court extent along the x-axis, measured from zero.
        width: Court extent along the y-axis, measured from zero.
        vertices: Landmark pairs with the four corresponding perimeter corners
            first. Coordinates use the same units as length and width.
        edges: Pairs of one-based landmark indices used to draw court lines.
    """

    length: float
    width: float

    @property
    def vertices(self) -> Sequence[Tuple[float, float]]:
        """Return court landmarks with the four boundary corners first.

        Returns:
            Perimeter corners followed by optional line landmarks.
        """
        ...

    @property
    def edges(self) -> Sequence[Tuple[int, int]]:
        """Return one-based endpoint indices for court lines.

        Returns:
            Pairs indexing the geometry's landmark sequence.
        """
        ...


class CourtCalibration:
    """Map source image pixels onto a rectangular court with a homography.

    The configuration supplies target geometry, so calibration is independent
    of a particular sport. Its first four vertices must be the corresponding
    perimeter corners, ordered consistently with the source points. The camera
    view and source resolution must remain stable while using this calibration.

    Args:
        corners: Four finite source (x, y) points with shape (4, 2), in perimeter
            order. Every point must be inside the source frame.
        frame_size: Source frame (width, height) in positive integer pixels.
        config: Rectangular court geometry. Length and width define the
            projection's coordinate units and the bounds used by player_mask.

    Attributes:
        corners: Source perimeter points as a float32 array of shape (4, 2).
        frame_size: Source frame (width, height) in pixels.
        config: Target geometry supplied by the caller.
        transformer: Homography from source pixels to court coordinates.

    Raises:
        ValueError: If source points or dimensions are invalid, the points do
            not form a convex quadrilateral, or a homography cannot be computed.

    Example:
        >>> from sports.configs.volleyball import VolleyballCourtConfiguration
        >>> calibration = CourtCalibration(
        ...     [[10, 10], [310, 10], [310, 170], [10, 170]],
        ...     (320, 180), VolleyballCourtConfiguration())
        >>> calibration.transform_points([[160, 90]]).round().tolist()
        [[900.0, 450.0]]
    """

    def __init__(
        self,
        corners: npt.ArrayLike,
        frame_size: Tuple[int, int],
        config: CourtGeometry,
    ) -> None:
        """Validate source geometry and construct its court homography."""
        self.corners: npt.NDArray[np.float32] = np.asarray(corners, dtype=np.float32)
        size = np.asarray(frame_size, dtype=float)
        if size.shape != (2,) or not np.isfinite(size).all():
            raise ValueError("frame_size must contain a finite width and height.")
        if (size <= 0).any() or (size != np.floor(size)).any():
            raise ValueError("frame_size must contain positive integer dimensions.")
        self.frame_size: Tuple[int, int] = (int(size[0]), int(size[1]))
        if self.corners.shape != (4, 2) or not np.isfinite(self.corners).all():
            raise ValueError("corners must contain four finite [x, y] points.")
        width, height = self.frame_size
        if (
            (self.corners < 0).any()
            or (self.corners[:, 0] >= width).any()
            or (self.corners[:, 1] >= height).any()
        ):
            raise ValueError("Court corners must be inside the source frame.")
        contour = self.corners.reshape(-1, 1, 2)
        if not cv2.isContourConvex(contour) or abs(cv2.contourArea(contour)) < 1:
            raise ValueError(
                "Court corners must form a convex, nondegenerate quadrilateral."
            )
        self.config = config
        self.transformer = ViewTransformer(
            source=self.corners,
            target=np.asarray(config.vertices[:4], dtype=np.float32),
        )

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        frame_size: Tuple[int, int],
        config: CourtGeometry,
    ) -> "CourtCalibration":
        """Read calibration JSON and check that its resolution matches the video.

        Args:
            path: JSON file containing frame_size and four source corners.
            frame_size: Expected source video (width, height) in pixels.
            config: Corresponding target court geometry.

        Returns:
            Validated calibration using the supplied target geometry.

        Raises:
            ValueError: If required fields are missing, the JSON is malformed,
                source geometry is invalid, or the stored resolution differs.
            OSError: If the calibration file cannot be read.
        """
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        if (
            not isinstance(data, dict)
            or "corners" not in data
            or "frame_size" not in data
        ):
            raise ValueError("Court JSON must contain frame_size and corners.")
        calibration = cls(data["corners"], data["frame_size"], config)
        if calibration.frame_size != tuple(frame_size):
            raise ValueError(
                f"Calibration is for {calibration.frame_size}, but video is {frame_size}. "
                "Recalibrate at the source video resolution."
            )
        return calibration

    def save(self, path: Union[str, Path]) -> None:
        """Write source frame dimensions and corners as UTF-8 JSON.

        Target geometry is supplied again when loading; it is not embedded in
        the file. Parent directories are created when needed.

        Args:
            path: Destination JSON path. An existing file is overwritten.

        Raises:
            OSError: If directories or the destination file cannot be written.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {"frame_size": list(self.frame_size), "corners": self.corners.tolist()},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def transform_points(self, points: npt.ArrayLike) -> npt.NDArray[np.float32]:
        """Project source points into the configuration's court coordinates.

        Args:
            points: Finite source (x, y) pixels with shape (N, 2). An empty list
                or array is accepted and returns an array of shape (0, 2).

        Returns:
            Float32 court positions of shape (N, 2), in the configuration's
            units. Points outside the frame may project into the free zone.

        Raises:
            ValueError: If point shape or values are invalid, or projected
                positions are not finite. Points whose homogeneous denominator
                is within float32 epsilon of zero have no finite projection.
        """
        pixels = self._source_points(points)
        if not self._projectable_mask(pixels).all():
            raise ValueError("These points cannot be projected with this calibration.")
        transformed = self.transformer.transform_points(pixels)
        if not np.isfinite(transformed).all():
            raise ValueError("These points cannot be projected with this calibration.")
        return transformed

    @staticmethod
    def _source_points(points: npt.ArrayLike) -> npt.NDArray[np.float32]:
        """Normalize finite source points while preserving the empty (0, 2) shape."""
        pixels = np.asarray(points, dtype=np.float32)
        if pixels.shape == (0,):
            pixels = pixels.reshape(0, 2)
        if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.isfinite(pixels).all():
            raise ValueError("Points must be a finite array with shape (N, 2).")
        return pixels

    def _projectable_mask(
        self, pixels: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.bool_]:
        """Reject horizon points that OpenCV would otherwise map to the origin."""
        matrix = self.transformer.m
        denominators = pixels @ matrix[2, :2] + matrix[2, 2]
        return np.isfinite(denominators) & (
            np.abs(denominators) > np.finfo(np.float32).eps
        )

    def player_mask(
        self, points: npt.ArrayLike, margin: float = 300
    ) -> npt.NDArray[np.bool_]:
        """Select source points that project inside the court and its free zone.

        Args:
            points: Finite source (x, y) points with shape (N, 2), typically
                bottom-center player box anchors.
            margin: Nonnegative extension of each court boundary in the
                configuration's coordinate units. Defaults to 300.

        Returns:
            Boolean mask of shape (N,) aligned with the source points. Court
            coordinates from -margin through length/width + margin are retained.
            Points without a finite court projection are False, including points
            on the homography horizon.

        Raises:
            ValueError: If the margin is nonfinite or negative, or point shape
                or source values are invalid.
        """
        if not np.isfinite(margin) or margin < 0:
            raise ValueError("Court margin must be finite and nonnegative.")
        pixels = self._source_points(points)
        projectable = self._projectable_mask(pixels)
        mask = np.zeros(len(pixels), dtype=bool)
        xy = self.transformer.transform_points(pixels[projectable])
        mask[projectable] = (
            np.isfinite(xy).all(axis=1)
            & (xy[:, 0] >= -margin)
            & (xy[:, 0] <= self.config.length + margin)
            & (xy[:, 1] >= -margin)
            & (xy[:, 1] <= self.config.width + margin)
        )
        return mask
