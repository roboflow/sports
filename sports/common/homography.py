from typing import Any, Mapping, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import supervision as sv

from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

HOMOGRAPHY_RANSAC_REPROJ_THRESH = 10.0
# Post-fit mean reprojection gate for m/s homography (slightly above RANSAC thresh).
SPEED_GATE_MAX_REPROJ_PX = 11.0
# Reject a gated fit when pitch-space mapping jumps more than this between frames.
SPEED_GATE_MAX_JUMP_CM = 600.0
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
            # Prefer forward RANSAC (src→dst). Fall back to inverse-then-invert,
            # then to a plain least-squares fit.
            m, _ = cv2.findHomography(
                src, dst, cv2.RANSAC, ransacReprojThreshold=ransac_thresh
            )
            if m is not None:
                self.m = m
                return
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


def image_displacement_to_pitch_m(
    feet_px: np.ndarray,
    displacement_px: np.ndarray,
    transformer: ViewTransformer | None,
) -> np.ndarray | None:
    """Map an image-space displacement at ``feet_px`` into pitch meters."""
    if transformer is None:
        return None
    feet = np.asarray(feet_px, dtype=np.float64).reshape(2)
    disp = np.asarray(displacement_px, dtype=np.float64).reshape(2)
    pts = np.stack([feet, feet + disp]).astype(np.float32)
    pitch_cm = transformer.transform_points(pts)
    return (pitch_cm[1] - pitch_cm[0]) / 100.0


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


def _pitch_correspondences(
    keypoints: Optional[sv.KeyPoints],
    *,
    config: SoccerPitchConfiguration = PITCH_CONFIG,
    confidence: float = 0.9,
    min_keypoints: int = DISPLAY_MIN_KEYPOINTS,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return accepted image and pitch point pairs for one frame."""
    if keypoints is None or keypoints.xy.shape[0] == 0:
        return None
    n = pitch_vertex_count(config)
    xy, conf = align_pitch_keypoints(keypoints, n_vertices=n)
    mask = pitch_keypoint_accept_mask(xy, conf, confidence=confidence)
    if mask.sum() < min_keypoints:
        return None
    src = xy[mask].astype(np.float32)
    dst = np.array(config.vertices, dtype=np.float32)[mask]
    return src, dst


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
    pts = _pitch_correspondences(
        keypoints,
        config=config,
        confidence=confidence,
        min_keypoints=min_keypoints,
    )
    if pts is None:
        return None
    src, dst = pts
    try:
        return RansacViewTransformer(
            source=src,
            target=dst,
            use_ransac=use_ransac,
            ransac_thresh=ransac_thresh,
        )
    except ValueError:
        return None


def _mean_reproj_px(
    transformer: ViewTransformer, src: np.ndarray, dst: np.ndarray
) -> float:
    """Average reprojection error in image pixels."""
    try:
        m_inv = np.linalg.inv(transformer.m)
    except np.linalg.LinAlgError:
        return float("inf")
    reproj = cv2.perspectiveTransform(
        dst.reshape(-1, 1, 2).astype(np.float32), m_inv
    ).reshape(-1, 2)
    return float(np.linalg.norm(reproj - src, axis=1).mean())


def _homography_jump_cm(
    prev: ViewTransformer, candidate: ViewTransformer, src: np.ndarray
) -> float:
    """Largest pitch-space displacement (cm) between two homographies."""
    if src is None or len(src) == 0:
        return 0.0
    pts = np.asarray(src, dtype=np.float32)
    prev_cm = prev.transform_points(pts)
    cand_cm = candidate.transform_points(pts)
    deltas = np.linalg.norm(cand_cm - prev_cm, axis=1)
    finite = deltas[np.isfinite(deltas)]
    if finite.size == 0:
        return float("inf")
    return float(finite.max())


def replay_gated_transforms(
    keypoints_by_frame: dict[int, sv.KeyPoints | None],
    *,
    confidence: float = 0.9,
    max_reproj_px: float = SPEED_GATE_MAX_REPROJ_PX,
    max_jump_cm: float = SPEED_GATE_MAX_JUMP_CM,
    ransac_thresh: float = HOMOGRAPHY_RANSAC_REPROJ_THRESH,
    config: SoccerPitchConfiguration = PITCH_CONFIG,
) -> dict[int, ViewTransformer | None]:
    """Re-derive gated speed homographies from cached keypoints."""
    locked: ViewTransformer | None = None
    last_was_accept = False
    transforms: dict[int, ViewTransformer | None] = {}
    for frame_idx in sorted(int(fi) for fi in keypoints_by_frame):
        kps = keypoints_by_frame.get(frame_idx)
        pts = _pitch_correspondences(kps, config=config, confidence=confidence)
        if pts is None:
            last_was_accept = False
            transforms[frame_idx] = None
            continue
        src, dst = pts
        try:
            candidate = RansacViewTransformer(
                source=src,
                target=dst,
                use_ransac=True,
                ransac_thresh=ransac_thresh,
            )
        except ValueError:
            last_was_accept = False
            transforms[frame_idx] = None
            continue

        err = _mean_reproj_px(candidate, src, dst)
        if err <= max_reproj_px:
            if (
                locked is not None
                and last_was_accept
                and _homography_jump_cm(locked, candidate, src) > max_jump_cm
            ):
                last_was_accept = False
                transforms[frame_idx] = None
                continue
            locked = candidate
            last_was_accept = True
            transforms[frame_idx] = candidate
            continue

        last_was_accept = False
        if locked is None:
            try:
                locked = RansacViewTransformer(
                    source=src, target=dst, use_ransac=False
                )
            except ValueError:
                pass
        transforms[frame_idx] = None
    return transforms


def build_minimap_transform_map(
    keypoints_by_frame: dict[int, sv.KeyPoints | None],
    *,
    confidence: float = 0.9,
    config: SoccerPitchConfiguration = PITCH_CONFIG,
) -> dict[int, ViewTransformer | None]:
    """Per-frame ungated homography for the visible minimap."""
    out: dict[int, ViewTransformer | None] = {}
    for fi, kps in keypoints_by_frame.items():
        out[int(fi)] = fit_pitch_homography(
            kps, config=config, confidence=confidence, use_ransac=False
        )
    return out


def gap_fill_speed_transforms(
    gated_by_frame: dict[int, ViewTransformer | None],
    ungated_by_frame: dict[int, ViewTransformer | None],
) -> dict[int, ViewTransformer]:
    """Speed H per frame: gated where available, else ungated minimap fit."""
    filled: dict[int, ViewTransformer] = {}
    for fi in gated_by_frame:
        gated = gated_by_frame[fi]
        t = gated if gated is not None else ungated_by_frame.get(int(fi))
        if t is not None:
            filled[int(fi)] = t
    return filled


def valid_pitch_cm(
    xy: np.ndarray,
    config: SoccerPitchConfiguration = PITCH_CONFIG,
    *,
    margin_cm: float = 200.0,
) -> np.ndarray:
    """Mask for warped points that fall inside the pitch rectangle (drops outliers)."""
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
