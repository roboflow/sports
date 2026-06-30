from typing import Any, Dict, List, Mapping, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import supervision as sv

from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

HOMOGRAPHY_RANSAC_REPROJ_THRESH = 10.0
DISPLAY_MIN_KEYPOINTS = 4
PITCH_CONFIG = SoccerPitchConfiguration()


class RansacViewTransformer(ViewTransformer):
    """ViewTransformer that optionally fits homography with RANSAC."""

    def __init__(
        self,
        source: npt.NDArray,
        target: npt.NDArray,
        use_ransac: bool = True,
        ransac_thresh: float = HOMOGRAPHY_RANSAC_REPROJ_THRESH,
    ) -> None:
        """
        Initialize from matching source and target point pairs.

        Args:
            source (npt.NDArray): Image-space points with shape (N, 2).
            target (npt.NDArray): Pitch-space points with shape (N, 2).
            use_ransac (bool): Fit with RANSAC when at least four points exist.
            ransac_thresh (float): RANSAC reprojection threshold in pixels.
        """
        src = np.asarray(source, dtype=np.float32)
        dst = np.asarray(target, dtype=np.float32)
        if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
            raise ValueError("source/target must be matching (N, 2) arrays")
        if use_ransac and len(src) >= 4:
            m_inv, _ = cv2.findHomography(
                dst, src, cv2.RANSAC, ransacReprojThreshold=ransac_thresh
            )
            if m_inv is not None:
                try:
                    self.m = np.linalg.inv(m_inv)
                    return
                except np.linalg.LinAlgError:
                    pass
        m, _ = cv2.findHomography(src, dst)
        if m is None:
            raise ValueError("Homography matrix could not be calculated.")
        self.m = m


def pitch_vertex_count(config: SoccerPitchConfiguration = PITCH_CONFIG) -> int:
    """Return the number of pitch template vertices."""
    return len(config.vertices)


def align_pitch_keypoints(
    keypoints: sv.KeyPoints,
    n_vertices: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize keypoint arrays to the pitch template vertex count.

    Args:
        keypoints (sv.KeyPoints): Input pitch keypoints.
        n_vertices (Optional[int]): Target vertex count.

    Returns:
        Tuple[np.ndarray, np.ndarray]: xy and confidence arrays.
    """
    n = n_vertices or pitch_vertex_count()
    if keypoints.xy.shape[0] == 0:
        return np.zeros((n, 2), dtype=np.float32), np.zeros(n, dtype=np.float32)
    xy = keypoints.xy[0].astype(np.float32)
    if keypoints.confidence is None:
        conf = np.ones(len(xy), dtype=np.float32)
    else:
        conf = keypoints.confidence[0].astype(np.float32)
    if len(conf) < n:
        conf = np.pad(conf, (0, n - len(conf)))
    else:
        conf = conf[:n]
    if xy.shape[0] < n:
        xy = np.pad(xy, ((0, n - xy.shape[0]), (0, 0)), constant_values=0)
    elif xy.shape[0] > n:
        xy = xy[:n]
    return xy, conf


def pitch_keypoint_accept_mask(
    xy: np.ndarray,
    conf: np.ndarray,
    confidence: float = 0.5,
) -> np.ndarray:
    """
    Return a mask of keypoints accepted for homography fitting.

    Args:
        xy (np.ndarray): Keypoint coordinates.
        conf (np.ndarray): Keypoint confidence scores.
        confidence (float): Minimum confidence threshold.

    Returns:
        np.ndarray: Boolean mask aligned with keypoint rows.
    """
    n = len(conf)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if len(xy) < n:
        xy = np.pad(xy.astype(np.float32), ((0, n - len(xy)), (0, 0)), constant_values=0)
    elif len(xy) > n:
        xy = xy[:n]
    return (conf > confidence) & (xy[:, 0] > 1) & (xy[:, 1] > 1)


def keypoints_from_inference_field(
    inference_result: Mapping[str, Any],
    n_vertices: Optional[int] = None,
) -> sv.KeyPoints:
    """
    Map Roboflow Inference keypoints into fixed pitch vertex slots.

    Args:
        inference_result (Mapping[str, Any]): Inference API response dict.
        n_vertices (Optional[int]): Target vertex count.

    Returns:
        sv.KeyPoints: Keypoints indexed by pitch vertex id.
    """
    n = n_vertices or pitch_vertex_count()
    if hasattr(inference_result, "model_dump"):
        payload = inference_result.model_dump(by_alias=True, exclude_none=True)
    elif hasattr(inference_result, "dict"):
        payload = inference_result.dict(exclude_none=True, by_alias=True)
    elif isinstance(inference_result, Mapping):
        payload = inference_result
    else:
        return sv.KeyPoints.empty()

    predictions = payload.get("predictions") or []
    if not predictions:
        return sv.KeyPoints.empty()

    prediction = max(predictions, key=lambda p: float(p.get("confidence", 0.0)))
    xy = np.zeros((1, n, 2), dtype=np.float32)
    conf = np.zeros((1, n), dtype=np.float32)

    for kp in prediction.get("keypoints") or []:
        idx = int(kp.get("class_id", -1))
        if idx < 0 or idx >= n:
            continue
        xy[0, idx, 0] = float(kp["x"])
        xy[0, idx, 1] = float(kp["y"])
        conf[0, idx] = float(kp.get("confidence", 0.0))

    return sv.KeyPoints(xy=xy, confidence=conf)


def valid_pitch_cm(
    xy: np.ndarray,
    config: SoccerPitchConfiguration = PITCH_CONFIG,
    margin_cm: float = 200.0,
) -> np.ndarray:
    """
    Return a mask for warped points inside the pitch rectangle.

    Args:
        xy (np.ndarray): Pitch-space points in centimeters.
        config (SoccerPitchConfiguration): Pitch configuration.
        margin_cm (float): Inset margin from pitch edges.

    Returns:
        np.ndarray: Boolean mask aligned with xy rows.
    """
    if xy is None or len(xy) == 0:
        return np.zeros(0, dtype=bool)
    finite = np.isfinite(xy).all(axis=1)
    return (
        finite
        & (xy[:, 0] >= margin_cm)
        & (xy[:, 0] <= config.length - margin_cm)
        & (xy[:, 1] >= margin_cm)
        & (xy[:, 1] <= config.width - margin_cm)
    )


def fit_pitch_homography(
    keypoints: Optional[sv.KeyPoints],
    config: SoccerPitchConfiguration = PITCH_CONFIG,
    confidence: float = 0.9,
    min_keypoints: int = DISPLAY_MIN_KEYPOINTS,
    use_ransac: bool = False,
    ransac_thresh: float = HOMOGRAPHY_RANSAC_REPROJ_THRESH,
) -> Optional[ViewTransformer]:
    """
    Fit image-to-pitch homography from pitch keypoints for one frame.

    Args:
        keypoints (Optional[sv.KeyPoints]): Pitch keypoints for one frame.
        config (SoccerPitchConfiguration): Pitch configuration.
        confidence (float): Minimum keypoint confidence.
        min_keypoints (int): Minimum accepted keypoints required to fit.
        use_ransac (bool): Whether to use RANSAC during fitting.
        ransac_thresh (float): RANSAC reprojection threshold in pixels.

    Returns:
        Optional[ViewTransformer]: Fitted homography, or None when fitting fails.
    """
    if keypoints is None or keypoints.xy.shape[0] == 0:
        return None
    n = pitch_vertex_count(config)
    xy, conf = align_pitch_keypoints(keypoints, n_vertices=n)
    mask = pitch_keypoint_accept_mask(xy, conf, confidence=confidence)
    if mask.sum() < min_keypoints:
        return None
    src = xy[mask].astype(np.float32)
    dst = np.array(config.vertices, dtype=np.float32)[mask]
    try:
        return RansacViewTransformer(
            source=src,
            target=dst,
            use_ransac=use_ransac,
            ransac_thresh=ransac_thresh,
        )
    except ValueError:
        return None
