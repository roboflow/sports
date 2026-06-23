"""analytics/homography.py — demo-quality pitch homography for player-motion analytics.

Imports primitives from the sports library where equivalent; does NOT vendor
ViewTransformer, draw_pitch, or draw_points_on_pitch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import numpy.typing as npt
import supervision as sv

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

# ── constants ──────────────────────────────────────────────────────────────────
HOMOGRAPHY_RANSAC_REPROJ_THRESH = 10.0
SPEED_GATE_MAX_REPROJ_PX = 11.0
# Maximum per-frame homography jump tolerated between two CONSECUTIVE accepted fits,
# measured as the largest pitch-space displacement (cm) of the frame's own keypoint
# correspondences when warped through the previous accepted H vs the new candidate H.
# A candidate that clears the reprojection gate can still teleport the whole mapping
# for a single frame (corrupting the radar trace); such a frame is rejected and the
# previous locked H is held. 600 cm (6 m) sits well above normal frame-to-frame camera
# motion of in-view points yet far below a genuine misfit jump, so it only rejects
# spikes without over-rejecting ordinary panning. The gate is skipped right after any
# gap (no consecutive prior accept) so a stale lock cannot block recovery.
SPEED_GATE_MAX_JUMP_CM = 600.0
DISPLAY_MIN_KEYPOINTS = 4
PITCH_CONFIG = SoccerPitchConfiguration()


# ---------------------------------------------------------------------------
# RansacViewTransformer — sports ViewTransformer + RANSAC inverse trick
# ---------------------------------------------------------------------------

class RansacViewTransformer(ViewTransformer):
    """ViewTransformer with RANSAC reprojection gating.

    Fits H via the *inverse* path (pitch→image) so the RANSAC threshold is in
    image pixels, then inverts.
    Subclasses sports.common.view.ViewTransformer so it is a drop-in replacement.
    """

    def __init__(
        self,
        source: npt.NDArray,
        target: npt.NDArray,
        *,
        use_ransac: bool = True,
        ransac_thresh: float = HOMOGRAPHY_RANSAC_REPROJ_THRESH,
    ) -> None:
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


# ---------------------------------------------------------------------------
# Keypoint alignment helpers
# ---------------------------------------------------------------------------

def pitch_vertex_count(config: SoccerPitchConfiguration = PITCH_CONFIG) -> int:
    return len(config.vertices)


def align_pitch_keypoints(
    keypoints: sv.KeyPoints,
    *,
    n_vertices: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalise keypoint arrays to the pitch template vertex count (pad/truncate)."""
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
    *,
    confidence: float = 0.5,
) -> np.ndarray:
    """True where a keypoint is accepted for homography fitting."""
    n = len(conf)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if len(xy) < n:
        xy = np.pad(xy.astype(np.float32), ((0, n - len(xy)), (0, 0)), constant_values=0)
    elif len(xy) > n:
        xy = xy[:n]
    return (conf > confidence) & (xy[:, 0] > 1) & (xy[:, 1] > 1)


def keypoints_from_inference_field(
    inference_result,
    *,
    n_vertices: int | None = None,
) -> sv.KeyPoints:
    """Map Roboflow Inference keypoints into fixed pitch vertex slots by class_id.

    Inference omits low-confidence vertices, so sequential packing mis-aligns
    landmarks after the first missing point. This maps by class_id instead.
    """
    n = n_vertices or pitch_vertex_count()
    if hasattr(inference_result, "model_dump"):
        inference_result = inference_result.model_dump(by_alias=True, exclude_none=True)
    elif hasattr(inference_result, "dict"):
        inference_result = inference_result.dict(exclude_none=True, by_alias=True)

    predictions = inference_result.get("predictions") or []
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


# ---------------------------------------------------------------------------
# Internal homography helpers
# ---------------------------------------------------------------------------

def _mean_reproj_px(
    transformer: ViewTransformer, src: np.ndarray, dst: np.ndarray
) -> float:
    """Average reprojection error in image pixels (pitch→image→compare with src)."""
    try:
        m_inv = np.linalg.inv(transformer.m)
    except np.linalg.LinAlgError:
        return float("inf")
    reproj = cv2.perspectiveTransform(
        dst.reshape(-1, 1, 2).astype(np.float32), m_inv
    ).reshape(-1, 2)
    return float(np.linalg.norm(reproj - src, axis=1).mean())


def _flip_pitch_x_targets(dst: np.ndarray, length: float) -> np.ndarray:
    out = dst.copy()
    out[:, 0] = length - out[:, 0]
    return out


def _homography_jump_cm(
    prev: ViewTransformer, candidate: ViewTransformer, src: np.ndarray
) -> float:
    """Largest pitch-space displacement (cm) between two image→pitch homographies.

    Warps the same in-view image points (the frame's accepted keypoints) through the
    previous accepted H and the new candidate H, and returns the max distance between
    the two pitch mappings — how far a fixed image point would teleport on the pitch if
    the candidate replaced the previous H.
    """
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


def _players_on_pitch_score(
    transformer: ViewTransformer,
    detections,
    config: SoccerPitchConfiguration = PITCH_CONFIG,
) -> tuple[int, float]:
    """Count in-bounds players and team separation (cm) under a candidate homography.

    A correctly oriented homography places most players inside the pitch rectangle and
    keeps the two teams horizontally separated; a mirrored one tends to throw players
    off-pitch. Team separation is only used when team ids are present on ``detections``.
    """
    from analytics.support import feet_xy, player_mask

    pmask = player_mask(detections)
    if not pmask.any():
        return 0, 0.0
    feet = feet_xy(detections)[pmask].astype(np.float32)
    cm = transformer.transform_points(feet)
    in_bounds = int(valid_pitch_cm(cm, config, margin_cm=80.0).sum())
    separation = 0.0
    team = detections.data.get("team") if detections.data else None
    if team is not None:
        teams = np.asarray(team)[pmask]
        if np.any(teams == 0) and np.any(teams == 1):
            separation = abs(
                float(cm[teams == 0, 0].mean()) - float(cm[teams == 1, 0].mean())
            )
    return in_bounds, separation


def _score_homography_candidate(
    transformer: ViewTransformer,
    src: np.ndarray,
    target: np.ndarray,
    detections,
    *,
    max_reproj_px: float,
) -> float:
    """Layout score for a candidate H (higher is better).

    Rewards players landing on the pitch and the two teams being spread apart, and
    penalizes keypoint reprojection error. Candidates above the reprojection gate, or
    with too few players on the pitch, are pushed to the bottom of the ranking.
    """
    err = _mean_reproj_px(transformer, src, target)
    if err > max_reproj_px:
        return -1e9
    in_bounds, separation = _players_on_pitch_score(transformer, detections)
    if in_bounds < 4:
        return -1e6 + in_bounds
    return in_bounds * 15.0 + separation / 40.0 - err * 2.0


def _orientation_matches_anchor(
    candidate: ViewTransformer,
    anchor: ViewTransformer,
    src: np.ndarray,
    *,
    min_corr: float = 0.85,
) -> bool:
    """Reject H that mirrors the pitch left/right vs the orientation anchor."""
    if len(src) < 4:
        return True
    pa = anchor.transform_points(src)
    pc = candidate.transform_points(src)
    corr = np.corrcoef(pa[:, 0], pc[:, 0])[0, 1]
    if not np.isfinite(corr):
        return True
    return float(corr) >= min_corr


# ---------------------------------------------------------------------------
# homography_from_keypoints_radar — ungated minimap H (for gaps / traces)
# ---------------------------------------------------------------------------

def homography_from_keypoints_radar(
    keypoints: sv.KeyPoints | None,
    *,
    config: SoccerPitchConfiguration = PITCH_CONFIG,
    confidence: float = 0.9,
    min_keypoints: int = DISPLAY_MIN_KEYPOINTS,
    use_ransac: bool = False,
) -> ViewTransformer | None:
    """Per-frame minimap H from confidence-gated keypoints (no reprojection gate)."""
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
        return RansacViewTransformer(source=src, target=dst, use_ransac=use_ransac)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Minimap homography selection (no-mirror)
# ---------------------------------------------------------------------------
# The visible radar (minimap, per-track traces and live dots) reads a single shared
# homography per frame fitted to the PLAIN pitch vertices only — no plain-vs-mirror
# candidates — so it physically cannot flip. It is taken verbatim from the per-frame
# keypoint fit (homography_from_keypoints_radar) with no jump gate or smoothing layered
# on top, so slight frame-to-frame jitter is expected and accepted in exchange for the
# no-flip guarantee. The gated, mirror-capable, jump-gated tracker H (radar_transforms /
# speed_transforms) stays reserved for the metrics (speed, distance, goalkeeper
# goal-distance), which need the stable oriented fit.


def build_radar_homography_map(
    metric: "MetricContext",
    *,
    confidence: float = 0.9,
) -> dict[int, ViewTransformer | None]:
    """No-mirror keypoint-radar H per frame for the visible minimap, traces and live dots.

    Returns the ungated per-frame keypoint homography map (fitted to the plain pitch
    vertices only), shared by the minimap, the per-track traces and the live dots. With
    no mirror branch it can never flip, and no jump gate or EMA is applied here, so the
    visible radar follows the raw per-frame fit. Frames without enough accepted keypoints
    map to ``None`` and are skipped by callers. The metrics keep the gated, mirror-capable
    ``radar_transforms`` / ``speed_transforms``.
    """
    return metric.keypoint_radar_transforms(confidence)


# ---------------------------------------------------------------------------
# PitchHomographyTracker
# ---------------------------------------------------------------------------

class PitchHomographyTracker:
    """Sequence-stable homography: per-frame RANSAC fit + orientation lock + reproj gate.

    Each frame uses only its own confidence-filtered keypoints (no cross-frame stacking).
    Stability via orientation lock after the first good fit + reprojection gate.
    """

    def __init__(
        self,
        *,
        confidence: float = 0.9,
        ransac_thresh: float = HOMOGRAPHY_RANSAC_REPROJ_THRESH,
        max_reproj_px: float = SPEED_GATE_MAX_REPROJ_PX,
        max_jump_cm: float = SPEED_GATE_MAX_JUMP_CM,
        config: SoccerPitchConfiguration | None = None,
    ) -> None:
        self.confidence = confidence
        self.ransac_thresh = ransac_thresh
        self.max_reproj_px = max_reproj_px
        self.max_jump_cm = max_jump_cm
        self.config = config or PITCH_CONFIG
        self._targets = np.array(self.config.vertices, dtype=np.float32)
        self._locked: ViewTransformer | None = None
        self._orientation_anchor: ViewTransformer | None = None
        # Was the immediately preceding frame an accept? The jump gate only fires
        # between consecutive accepts, so a stale lock held across a gap cannot block
        # re-baselining once a good fit returns.
        self._last_was_accept: bool = False

    def _frame_correspondences(
        self, keypoints: sv.KeyPoints
    ) -> tuple[np.ndarray, np.ndarray] | None:
        n = len(self._targets)
        xy, conf = align_pitch_keypoints(keypoints, n_vertices=n)
        mask = pitch_keypoint_accept_mask(xy, conf, confidence=self.confidence)
        if mask.sum() < DISPLAY_MIN_KEYPOINTS:
            return None
        return xy[mask].astype(np.float32), self._targets[mask]

    def _fit_frame(
        self, src: np.ndarray, dst: np.ndarray, *, detections=None
    ) -> tuple[ViewTransformer, np.ndarray] | None:
        """Pick the plain vs mirrored target for this frame's keypoints.

        When per-frame player ``detections`` are supplied, candidate orientations are
        ranked by a layout score (in-bounds players + team separation, penalized by
        reprojection error) so the correct (non-mirrored) pitch is chosen even when the
        mirrored fit has a marginally lower keypoint error. Without detections the rank
        falls back to lowest reprojection error.
        """
        length = float(self.config.length)
        candidates: list[tuple[float, float, ViewTransformer, np.ndarray]] = []
        for target in (dst, _flip_pitch_x_targets(dst, length)):
            try:
                t = RansacViewTransformer(
                    source=src,
                    target=target,
                    use_ransac=True,
                    ransac_thresh=self.ransac_thresh,
                )
            except ValueError:
                continue
            err = _mean_reproj_px(t, src, target)
            if detections is not None:
                score = _score_homography_candidate(
                    t, src, target, detections, max_reproj_px=self.max_reproj_px
                )
            else:
                score = -err
            candidates.append((score, err, t, target))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        for score, err, t, target in candidates:
            if err > self.max_reproj_px:
                continue
            if self._orientation_anchor is None or _orientation_matches_anchor(
                t, self._orientation_anchor, src
            ):
                return t, target
        # return best-ranked candidate if nothing passes the anchor check
        _score, _err, t, target = candidates[0]
        return t, target

    def update(
        self, keypoints: sv.KeyPoints | None, *, detections=None
    ) -> tuple[ViewTransformer | None, ViewTransformer | None]:
        """Return (speed_transformer, radar_transformer).

        speed_transformer is None when the frame fails the reprojection gate.
        radar_transformer holds the last accepted H (orientation-locked). When per-frame
        player ``detections`` are passed they steer the plain-vs-mirror orientation pick.
        """
        if keypoints is None or keypoints.xy.shape[0] == 0:
            self._last_was_accept = False
            return None, self._locked

        pair = self._frame_correspondences(keypoints)
        if pair is None:
            self._last_was_accept = False
            return None, self._locked

        src_now, dst_now = pair
        fitted = self._fit_frame(src_now, dst_now, detections=detections)
        if fitted is None:
            self._last_was_accept = False
            return None, self._locked

        cand_t, target = fitted
        err = _mean_reproj_px(cand_t, src_now, target)
        if err <= self.max_reproj_px:
            # Jump gate: a candidate that clears the reprojection gate can still
            # teleport the whole mapping for one frame. Reject it (hold the previous H,
            # treated like a gate-fail so the gap-fill / locked path takes over) when it
            # would displace the in-view points beyond ``max_jump_cm`` relative to the
            # previous accepted H. Only applied between consecutive accepts so the first
            # frame and post-gap recovery frames are exempt.
            if (
                self._locked is not None
                and self._last_was_accept
                and _homography_jump_cm(self._locked, cand_t, src_now) > self.max_jump_cm
            ):
                self._last_was_accept = False
                return None, self._locked
            if self._orientation_anchor is None:
                self._orientation_anchor = cand_t
            self._locked = cand_t
            self._last_was_accept = True
            return cand_t, self._locked

        # RANSAC failed gate — try plain fallback just for radar lock
        self._last_was_accept = False
        if self._locked is None:
            try:
                fallback = RansacViewTransformer(
                    source=src_now, target=dst_now, use_ransac=False
                )
                self._locked = fallback
            except ValueError:
                pass
        return None, self._locked


# ---------------------------------------------------------------------------
# replay_tracker_transforms — re-gate from cached keypoints
# ---------------------------------------------------------------------------

def replay_tracker_transforms(
    keypoints_by_frame: dict[int, sv.KeyPoints | None],
    *,
    confidence: float = 0.9,
    max_reproj_px: float = SPEED_GATE_MAX_REPROJ_PX,
    max_jump_cm: float = SPEED_GATE_MAX_JUMP_CM,
    ransac_thresh: float = HOMOGRAPHY_RANSAC_REPROJ_THRESH,
    config: SoccerPitchConfiguration = PITCH_CONFIG,
    detections_by_frame: dict[int, sv.Detections] | None = None,
) -> tuple[dict[int, ViewTransformer | None], dict[int, ViewTransformer | None]]:
    """Re-derive gated speed + radar homographies from already-detected keypoints.

    When ``detections_by_frame`` is supplied, the per-frame player detections steer the
    plain-vs-mirror orientation pick so the gated homography is not mirrored.
    """
    tracker = PitchHomographyTracker(
        confidence=confidence,
        ransac_thresh=ransac_thresh,
        max_reproj_px=max_reproj_px,
        max_jump_cm=max_jump_cm,
        config=config,
    )
    transforms: dict[int, ViewTransformer | None] = {}
    radar_transforms: dict[int, ViewTransformer | None] = {}
    for frame_idx in sorted(int(fi) for fi in keypoints_by_frame):
        kps = keypoints_by_frame.get(frame_idx)
        dets = detections_by_frame.get(frame_idx) if detections_by_frame is not None else None
        speed_t, radar_t = tracker.update(kps, detections=dets)
        transforms[frame_idx] = speed_t
        radar_transforms[frame_idx] = radar_t
    return transforms, radar_transforms


def build_metric_from_maps(
    keypoints_by_frame: dict[int, sv.KeyPoints | None],
    *,
    detections_by_frame: dict[int, sv.Detections] | None = None,
    pitch_confidence: float = 0.9,
    max_reproj_px: float = SPEED_GATE_MAX_REPROJ_PX,
    max_jump_cm: float = SPEED_GATE_MAX_JUMP_CM,
    ransac_thresh: float = HOMOGRAPHY_RANSAC_REPROJ_THRESH,
    config: SoccerPitchConfiguration = PITCH_CONFIG,
) -> "MetricContext":
    """Build a :class:`MetricContext` from precomputed keypoints (and optional detections).

    Used when keypoints come from the on-disk cache: the gated/radar homographies are
    rebuilt from them (cheap), with per-frame detections steering the orientation pick.
    """
    transforms, radar_transforms = replay_tracker_transforms(
        keypoints_by_frame,
        confidence=pitch_confidence,
        max_reproj_px=max_reproj_px,
        max_jump_cm=max_jump_cm,
        ransac_thresh=ransac_thresh,
        config=config,
        detections_by_frame=detections_by_frame,
    )
    return MetricContext(
        transforms=transforms,
        radar_transforms=radar_transforms,
        keypoints=keypoints_by_frame,
    )


# ---------------------------------------------------------------------------
# MetricContext
# ---------------------------------------------------------------------------

@dataclass
class MetricContext:
    """Per-frame speed H, radar H, and pitch keypoints."""

    transforms: dict[int, Any]       # gated speed H per frame (may be None)
    radar_transforms: dict[int, Any]  # orientation-locked radar H per frame
    keypoints: dict[int, Any]         # raw sv.KeyPoints per frame

    @property
    def speed_transforms(self) -> dict[int, Any]:
        """Gated speed H (non-None only) — for compute_kinematics distance."""
        return {int(fi): t for fi, t in self.transforms.items() if t is not None}

    def speed_transforms_gap_filled(self, confidence: float = 0.9) -> dict[int, Any]:
        """Speed H per frame: gated where available, else ungated keypoint H on gap frames.

        Used for the displayed Kalman speed badge so it stays continuous through
        frames the reprojection gate rejects. Distance integration still uses the
        gated-only speed_transforms.
        """
        ungated = self.keypoint_radar_transforms(confidence)
        filled: dict[int, Any] = {}
        for fi in self.transforms:
            gated = self.transforms[fi]
            t = gated if gated is not None else ungated.get(int(fi))
            if t is not None:
                filled[int(fi)] = t
        return filled

    def keypoint_radar_transforms(self, confidence: float = 0.9) -> dict[int, Any]:
        """Per-frame ungated sports-radar H from keypoints (radar minimap / traces).

        Memoized per confidence level.
        """
        cache = self.__dict__.setdefault("_kp_radar_cache", {})
        key = round(float(confidence), 6)
        if key not in cache:
            kps = self.keypoints or {}
            cache[key] = {
                int(fi): homography_from_keypoints_radar(k, confidence=confidence)
                for fi, k in kps.items()
            }
        return cache[key]


# ---------------------------------------------------------------------------
# ensure_pitch_homography_maps — build MetricContext from video
# ---------------------------------------------------------------------------

def ensure_pitch_homography_maps(
    source_video_path: str,
    pitch_detector_fn: Callable[[np.ndarray], sv.KeyPoints],
    *,
    fps: float,
    max_frames: int | None = None,
    pitch_confidence: float = 0.9,
    detections_by_frame: dict[int, sv.Detections] | None = None,
    player_detector_fn: Callable[[np.ndarray], sv.Detections] | None = None,
) -> MetricContext:
    """Run pitch keypoint detection on every frame and build gated homographies.

    When per-frame player detections are available — either precomputed via
    ``detections_by_frame`` or detected inline with ``player_detector_fn`` — they steer
    the plain-vs-mirror orientation pick so the gated speed/radar homography (which also
    feeds goalkeeper goal-distance assignment and distance) is not mirrored.
    """
    tracker = PitchHomographyTracker(confidence=pitch_confidence)
    transforms: dict[int, ViewTransformer | None] = {}
    radar_transforms: dict[int, ViewTransformer | None] = {}
    keypoints_by_frame: dict[int, sv.KeyPoints | None] = {}

    cap = cv2.VideoCapture(source_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {source_video_path}")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if max_frames is not None and frame_idx > max_frames:
                break
            kps = pitch_detector_fn(frame)
            keypoints_by_frame[frame_idx] = kps
            dets = None
            if detections_by_frame is not None:
                dets = detections_by_frame.get(frame_idx)
            elif player_detector_fn is not None:
                dets = player_detector_fn(frame)
            speed_t, radar_t = tracker.update(kps, detections=dets)
            transforms[frame_idx] = speed_t
            radar_transforms[frame_idx] = radar_t
    finally:
        cap.release()

    return MetricContext(
        transforms=transforms,
        radar_transforms=radar_transforms,
        keypoints=keypoints_by_frame,
    )
