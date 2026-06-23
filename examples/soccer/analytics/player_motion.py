"""Shared player-motion utilities for analytics modes.

Owns BoTSORT tracker factories, Kalman speed/velocity smoothers, feet anchoring,
kinematics integration, team-classifier helpers, referee/goalkeeper hygiene,
detector factories, and annotation helpers (ellipses, joystick dots, radar minimap,
trace drawing). Does not own homography (see ``homography``), clip orchestration
(see ``clip_pipeline``), or mode-specific render loops.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import supervision as sv
from trackers import BoTSORTTracker, ByteTrackTracker
from trackers.utils.state_representations import XCYCWHStateEstimator, XYXYStateEstimator

from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer

# ── class / team id constants ──────────────────────────────────────────────────
BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3
TEAM_NONE = -1

# ── paths (data/ lives one dir above analytics/) ──────────────────────────────
_SOCCER_DIR = Path(__file__).resolve().parent.parent
PLAYER_DETECTION_MODEL_PATH = str(_SOCCER_DIR / "data" / "football-player-detection.pt")
PITCH_DETECTION_MODEL_PATH = str(_SOCCER_DIR / "data" / "football-pitch-detection.pt")

# ── tracker / Kalman defaults ──────────────────────────────────────────────────
DEFAULT_TRACK_ACTIVATION_THRESHOLD = 0.55
DEFAULT_HIGH_CONF_DET_THRESHOLD = 0.6
DEFAULT_MINIMUM_IOU_THRESHOLD_FIRST_ASSOC = 0.15
DEFAULT_MIN_SPEED_PX = 0.5

# ── referee / goalkeeper hygiene (box-overlap IoU thresholds) ──────────────────
# Referees must never reach the player set used for tracking, kinematics or
# annotation, or they would pick up a track id, an ellipse and a speed/distance
# readout and inflate the distance numbers. An outfield player box overlapping a
# referee box is treated as the same person detected under two classes and dropped;
# a tracklet that ever sits this close to a referee box across the clip is flagged
# and excluded entirely.
REFEREE_PLAYER_IOU_THRESHOLD = 0.25
REFEREE_TRACK_IOU_THRESHOLD = 0.4
GOALKEEPER_PLAYER_IOU_THRESHOLD = 0.25

# ── team classification ────────────────────────────────────────────────────────
STRIDE = 60  # frames between team-fit samples (matches existing main.py)


# ---------------------------------------------------------------------------
# Tracker factory
# ---------------------------------------------------------------------------

# Process-level count of player trackers built. Each full multi-object tracking pass
# over the clip constructs exactly one tracker here, so this doubles as a cheap proof
# of how many BoTSORT passes a run performed: the in-process run-all shares a single
# pass (count == 1), whereas running the modes separately rebuilds a tracker per pass.
# Inspect via ``get_tracker_build_count`` or set ``ANALYTICS_DEBUG_TRACKER=1`` to log.
TRACKER_BUILD_COUNT = 0


def get_tracker_build_count() -> int:
    """Number of player trackers built so far in this process (BoTSORT-pass counter)."""
    return TRACKER_BUILD_COUNT


def reset_tracker_build_count() -> None:
    """Reset the player-tracker build counter (used by tests / run-all verification)."""
    global TRACKER_BUILD_COUNT
    TRACKER_BUILD_COUNT = 0


def create_player_tracker(
    frame_rate: float,
    *,
    kind: str = "botsort",
    track_activation_threshold: float = DEFAULT_TRACK_ACTIVATION_THRESHOLD,
    high_conf_det_threshold: float = DEFAULT_HIGH_CONF_DET_THRESHOLD,
    minimum_iou_threshold_first_assoc: float = DEFAULT_MINIMUM_IOU_THRESHOLD_FIRST_ASSOC,
):
    """Return a multi-object tracker configured for football players."""
    global TRACKER_BUILD_COUNT
    TRACKER_BUILD_COUNT += 1
    if os.environ.get("ANALYTICS_DEBUG_TRACKER"):
        print(f"[analytics] BoTSORT pass #{TRACKER_BUILD_COUNT} (tracker kind={kind!r})")
    if kind == "bytetrack":
        tracker = ByteTrackTracker(
            frame_rate=frame_rate,
            track_activation_threshold=track_activation_threshold,
            high_conf_det_threshold=high_conf_det_threshold,
        )
    else:
        tracker = BoTSORTTracker(
            frame_rate=frame_rate,
            track_activation_threshold=track_activation_threshold,
            high_conf_det_threshold=high_conf_det_threshold,
            minimum_iou_threshold_first_assoc=minimum_iou_threshold_first_assoc,
            enable_cmc=(kind == "botsort"),
            cmc_method="sparseOptFlow",
        )
    # Tracker ids are minted from a class-level counter shared across instances. Reset
    # it here so each sequential pass (goalkeeper lock / first pass / render) starts ids
    # from 0; this keeps ids consistent across passes so per-tracklet locks and the
    # PLAYER_FOCUS --track-id selection refer to the same players in every pass.
    tracker.reset()
    return tracker


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def feet_xy(detections: sv.Detections) -> np.ndarray:
    """Bottom-center anchor (ground contact point)."""
    return detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)


def player_mask(detections: sv.Detections) -> np.ndarray:
    """Boolean mask of outfield players + goalkeepers."""
    return np.isin(detections.class_id, (PLAYER_CLASS_ID, GOALKEEPER_CLASS_ID))


def get_crops(frame: np.ndarray, detections: sv.Detections) -> list[np.ndarray]:
    """Crop bounding boxes from frame."""
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.ndarray,
    goalkeepers: sv.Detections,
) -> np.ndarray:
    """Assign each goalkeeper to the nearest team centroid."""
    gk_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pl_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    t0 = pl_xy[players_team_id == 0].mean(axis=0) if (players_team_id == 0).any() else pl_xy.mean(axis=0)
    t1 = pl_xy[players_team_id == 1].mean(axis=0) if (players_team_id == 1).any() else pl_xy.mean(axis=0)
    result = []
    for xy in gk_xy:
        d0 = float(np.linalg.norm(xy - t0))
        d1 = float(np.linalg.norm(xy - t1))
        result.append(0 if d0 <= d1 else 1)
    return np.array(result, dtype=int)


# ---------------------------------------------------------------------------
# Referee / goalkeeper hygiene
# ---------------------------------------------------------------------------

def split_detection_roles(
    dets: sv.Detections,
) -> tuple[sv.Detections, sv.Detections, sv.Detections]:
    """Split detections into (players, goalkeepers, referees) by class id."""
    return (
        dets[dets.class_id == PLAYER_CLASS_ID],
        dets[dets.class_id == GOALKEEPER_CLASS_ID],
        dets[dets.class_id == REFEREE_CLASS_ID],
    )


def _detection_overlap_with_referees(
    subject: sv.Detections,
    referees: sv.Detections,
    *,
    iou_threshold: float = REFEREE_PLAYER_IOU_THRESHOLD,
) -> np.ndarray:
    """Per-row mask: subject box overlaps a referee box, or holds a referee centre."""
    if len(subject) == 0 or len(referees) == 0:
        return np.zeros(len(subject), dtype=bool)
    ious = sv.box_iou_batch(subject.xyxy, referees.xyxy)
    overlap = ious.max(axis=1) >= iou_threshold
    if overlap.all():
        return overlap
    # A referee whose centre sits inside the player box but with low IoU (very different
    # box sizes) is still the same person, so fold those in too.
    rcx = (referees.xyxy[:, 0] + referees.xyxy[:, 2]) / 2.0
    rcy = (referees.xyxy[:, 1] + referees.xyxy[:, 3]) / 2.0
    x1, y1, x2, y2 = (
        subject.xyxy[:, 0], subject.xyxy[:, 1], subject.xyxy[:, 2], subject.xyxy[:, 3]
    )
    for i in np.flatnonzero(~overlap):
        inside = (rcx >= x1[i]) & (rcx <= x2[i]) & (rcy >= y1[i]) & (rcy <= y2[i])
        if inside.any():
            overlap[i] = True
    return overlap


def suppress_players_overlapping_referees(
    players: sv.Detections,
    referees: sv.Detections,
    *,
    iou_threshold: float = REFEREE_PLAYER_IOU_THRESHOLD,
) -> sv.Detections:
    """Drop outfield player rows that duplicate a referee detection (one person, two classes)."""
    if len(players) == 0 or len(referees) == 0:
        return players
    drop = _detection_overlap_with_referees(players, referees, iou_threshold=iou_threshold)
    return players[~drop]


def suppress_goalkeepers_overlapping_players(
    goalkeepers: sv.Detections,
    players: sv.Detections,
    *,
    iou_threshold: float = GOALKEEPER_PLAYER_IOU_THRESHOLD,
) -> sv.Detections:
    """Drop goalkeeper rows that duplicate a nearby outfield player (class flicker)."""
    if len(goalkeepers) == 0 or len(players) == 0:
        return goalkeepers
    ious = sv.box_iou_batch(goalkeepers.xyxy, players.xyxy)
    drop = ious.max(axis=1) >= iou_threshold
    return goalkeepers[~drop]


def filter_referees_from_detections(
    dets: sv.Detections,
    *,
    blocked_tracker_ids: set[int] | frozenset[int] | None = None,
    iou_threshold: float = REFEREE_PLAYER_IOU_THRESHOLD,
) -> sv.Detections:
    """Remove referee rows, outfield players overlapping a referee, and flagged tracklets."""
    if len(dets) == 0:
        return dets
    ref_mask = dets.class_id == REFEREE_CLASS_ID
    keep = ~ref_mask
    out_mask = dets.class_id == PLAYER_CLASS_ID
    refs = dets[ref_mask]
    if len(refs) and out_mask.any():
        overlap = _detection_overlap_with_referees(
            dets[out_mask], refs, iou_threshold=iou_threshold
        )
        out_idx = np.flatnonzero(out_mask)
        keep[out_idx[overlap]] = False
    if blocked_tracker_ids and dets.tracker_id is not None:
        for i, tid in enumerate(dets.tracker_id):
            if int(tid) in blocked_tracker_ids:
                keep[i] = False
    return dets[keep]


def collect_referee_tracker_ids(
    frames,
    *,
    iou_threshold: float = REFEREE_TRACK_IOU_THRESHOLD,
) -> frozenset[int]:
    """Clip-level set of tracker ids that ever strongly overlap a referee box.

    ``frames`` yields ``(frame_idx, detections)`` where each detections row carries the
    tracked outfield players (with ids) alongside the raw referee rows. A player tracklet
    that lands on a referee box on any frame is a misclassified referee and is flagged so
    every later pass can exclude it from tracking, kinematics and annotation.
    """
    flagged: set[int] = set()
    for _, dets in frames:
        if dets.tracker_id is None:
            continue
        out_mask = dets.class_id == PLAYER_CLASS_ID
        ref_mask = dets.class_id == REFEREE_CLASS_ID
        if not out_mask.any() or not ref_mask.any():
            continue
        ious = sv.box_iou_batch(dets.xyxy[out_mask], dets.xyxy[ref_mask])
        out_idx = np.flatnonzero(out_mask)
        tids = dets.tracker_id[out_idx]
        for j, mx in enumerate(ious.max(axis=1)):
            if float(mx) >= iou_threshold and int(tids[j]) >= 0:
                flagged.add(int(tids[j]))
    return frozenset(flagged)


def enforce_one_goalkeeper_per_team(
    dets: sv.Detections,
    *,
    frame_width: float | None = None,
) -> sv.Detections:
    """Keep at most one goalkeeper per team; drop lower-confidence duplicate GK rows.

    Goalkeepers are bucketed by team id when known, else by which broadcast half the
    box centre falls in, and only the highest-confidence (then tallest) keeper survives
    in each bucket. Applied per frame before tracking so a flickered second keeper
    cannot spawn a spurious track.
    """
    if len(dets) == 0:
        return dets
    gk_indices = [
        i for i in range(len(dets)) if int(dets.class_id[i]) == GOALKEEPER_CLASS_ID
    ]
    if len(gk_indices) <= 1:
        return dets
    teams = dets.data.get("team") if dets.data else None

    def _group_key(i: int) -> int:
        team = int(teams[i]) if teams is not None else TEAM_NONE
        if team in (0, 1):
            return team
        cx = float((dets.xyxy[i, 0] + dets.xyxy[i, 2]) * 0.5)
        mid = (frame_width * 0.5) if frame_width and frame_width > 0 else 960.0
        return 0 if cx < mid else 1

    def _rank(i: int) -> tuple[float, float]:
        conf = float(dets.confidence[i]) if dets.confidence is not None else 0.0
        height = float(dets.xyxy[i, 3] - dets.xyxy[i, 1])
        return (conf, height)

    by_group: dict[int, list[int]] = {}
    for i in gk_indices:
        by_group.setdefault(_group_key(i), []).append(i)
    drop = np.zeros(len(dets), dtype=bool)
    for group_indices in by_group.values():
        if len(group_indices) <= 1:
            continue
        best = max(group_indices, key=_rank)
        for i in group_indices:
            if i != best:
                drop[i] = True
    return dets[~drop]


def build_trackable_detections(
    raw: sv.Detections | None,
    *,
    frame_width: float | None = None,
) -> sv.Detections:
    """Players + goalkeepers to feed the tracker, with referees and duplicate keepers removed.

    Single source of truth for the trackable set so the clip-level lock pass and every
    render pass hand the tracker identical input — that keeps tracker ids aligned across
    passes (the existing two-pass distance / focus code depends on this). Enforces one
    goalkeeper per team, drops referee rows and outfield players that flicker onto a
    referee, and removes goalkeepers duplicating a nearby player.
    """
    if raw is None or len(raw) == 0:
        return sv.Detections.empty()
    cleaned = enforce_one_goalkeeper_per_team(raw, frame_width=frame_width)
    cleaned = filter_referees_from_detections(cleaned)
    players, gks, _ = split_detection_roles(cleaned)
    gks = suppress_goalkeepers_overlapping_players(gks, players)
    if len(players) or len(gks):
        return sv.Detections.merge([players, gks])
    return sv.Detections.empty()


def combine_for_referee_check(
    tracked: sv.Detections,
    referees: sv.Detections,
) -> sv.Detections:
    """Stack tracked player/gk rows (with ids) and referee rows (id -1) for ref detection.

    The result carries only ``xyxy`` / ``class_id`` / ``tracker_id`` — enough for
    :func:`collect_referee_tracker_ids` to flag player tracklets that land on a referee.
    """
    if len(tracked) and len(referees):
        tids = (
            tracked.tracker_id
            if tracked.tracker_id is not None
            else np.full(len(tracked), -1)
        )
        return sv.Detections(
            xyxy=np.concatenate([tracked.xyxy, referees.xyxy], axis=0).astype(np.float32),
            class_id=np.concatenate([tracked.class_id, referees.class_id]).astype(int),
            tracker_id=np.concatenate(
                [tids, np.full(len(referees), -1, dtype=int)]
            ).astype(int),
        )
    return tracked if len(tracked) else referees


def drop_blocked_tracker_ids(
    dets: sv.Detections,
    blocked_tracker_ids: set[int] | frozenset[int] | None,
) -> sv.Detections:
    """Remove tracked rows whose tracker id was flagged as a referee tracklet."""
    if (
        not blocked_tracker_ids
        or dets is None
        or len(dets) == 0
        or dets.tracker_id is None
    ):
        return dets
    keep = np.array(
        [int(t) not in blocked_tracker_ids for t in dets.tracker_id], dtype=bool
    )
    if keep.all():
        return dets
    return dets[keep]


# ---------------------------------------------------------------------------
# Kalman velocity helpers
# ---------------------------------------------------------------------------

def _kalman_feet_velocity_from_tracklet(tracklet) -> np.ndarray | None:
    """Extract feet-referenced Kalman velocity from a tracker tracklet.

    Reads the bottom-edge velocity straight from the Kalman state vector, which only
    carries the velocity components once the filter has built an 8-element state
    (position + velocity per coordinate); shorter states have no usable velocity yet.
    """
    est = tracklet.state_estimator
    x = est.kf.x.flatten()
    if len(x) < 7:
        return None
    if isinstance(est, XYXYStateEstimator):
        vx = (float(x[4]) + float(x[6])) / 2.0
        vy = float(x[7])
    elif isinstance(est, XCYCWHStateEstimator):
        vx = float(x[4])
        vy = float(x[5]) + float(x[7]) / 2.0
    else:
        vx, vy = float(x[4]), float(x[5])
    return np.array([vx, vy], dtype=np.float64)


def _kalman_velocity_by_tracker_id(
    tracker, *, min_speed: float = DEFAULT_MIN_SPEED_PX
) -> dict[int, np.ndarray]:
    """Map confirmed tracker_id → feet-referenced Kalman velocity."""
    out: dict[int, np.ndarray] = {}
    for tracklet in tracker.tracks:
        tid = int(tracklet.tracker_id)
        if tid < 0:
            continue
        vel = _kalman_feet_velocity_from_tracklet(tracklet)
        if vel is None:
            continue
        if float(np.linalg.norm(vel)) < min_speed:
            continue
        out[tid] = vel
    return out


def kalman_velocity_arrays(
    detections: sv.Detections,
    tracker,
    *,
    min_speed: float = DEFAULT_MIN_SPEED_PX,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row kf_vx / kf_vy arrays aligned with detections."""
    n = len(detections)
    kf_vx = np.full(n, np.nan, dtype=np.float32)
    kf_vy = np.full(n, np.nan, dtype=np.float32)
    if n == 0 or detections.tracker_id is None:
        return kf_vx, kf_vy
    id_to_vel = _kalman_velocity_by_tracker_id(tracker, min_speed=min_speed)
    for i, tid in enumerate(detections.tracker_id):
        vel = id_to_vel.get(int(tid))
        if vel is None:
            continue
        kf_vx[i] = float(vel[0])
        kf_vy[i] = float(vel[1])
    return kf_vx, kf_vy


def attach_kalman_velocity(
    dets: sv.Detections,
    player_tracker,
    *,
    needs_frame: bool,
    image: np.ndarray | None,
) -> sv.Detections:
    """Run one tracker step on player rows and merge kf_vx / kf_vy onto detections."""
    pmask = np.isin(dets.class_id, (PLAYER_CLASS_ID, GOALKEEPER_CLASS_ID))
    n = len(dets)
    kf_vx = np.full(n, np.nan, dtype=np.float32)
    kf_vy = np.full(n, np.nan, dtype=np.float32)
    if pmask.any():
        trackable = dets[pmask]
        player_tracker.update(
            trackable,
            frame=image if needs_frame else None,
        )
        vx_sub, vy_sub = kalman_velocity_arrays(trackable, player_tracker)
        kf_vx[pmask] = vx_sub
        kf_vy[pmask] = vy_sub
    data = dict(dets.data) if dets.data else {}
    data["kf_vx"] = kf_vx
    data["kf_vy"] = kf_vy
    return sv.Detections(
        xyxy=dets.xyxy,
        class_id=dets.class_id,
        tracker_id=dets.tracker_id,
        confidence=dets.confidence,
        data=data,
    )


def kalman_ground_speed_m_s(
    feet_px: np.ndarray,
    vel_px: np.ndarray,
    transformer: ViewTransformer | None,
    *,
    fps: float,
    min_speed_px: float = 0.0,
) -> float | None:
    """Ground speed (m/s) from Kalman image velocity via pitch homography."""
    if transformer is None or fps <= 0:
        return None
    vel = np.asarray(vel_px, dtype=np.float64).reshape(2)
    vx, vy = float(vel[0]), float(vel[1])
    if not np.isfinite(vx) or not np.isfinite(vy):
        return 0.0
    speed_px_val = float(np.hypot(vx, vy))
    if speed_px_val < min_speed_px:
        return 0.0
    feet = np.asarray(feet_px, dtype=np.float64).reshape(2)
    p0 = transformer.transform_points(feet.reshape(1, 2).astype(np.float32))
    p1 = transformer.transform_points((feet + vel).reshape(1, 2).astype(np.float32))
    delta_cm = p1[0] - p0[0]
    delta_m = delta_cm / 100.0
    return float(np.linalg.norm(delta_m)) * float(fps)


class KalmanSpeedDisplaySmoother:
    """EMA on displayed ground speed (m/s) per track."""

    def __init__(self, *, alpha: float = 0.3) -> None:
        self.alpha = float(np.clip(alpha, 0.05, 1.0))
        self._speed: dict[int, float] = {}

    def smooth(self, tracker_id: int, speed_m_s: float) -> float:
        if tracker_id < 0:
            return float(speed_m_s)
        a = self.alpha
        if tracker_id in self._speed:
            speed_m_s = a * float(speed_m_s) + (1.0 - a) * self._speed[tracker_id]
        self._speed[tracker_id] = float(speed_m_s)
        return float(speed_m_s)


class KalmanVelocitySmoother:
    """Per-track EMA on kf_vx / kf_vy; holds last value on NaN."""

    def __init__(self, *, alpha: float = 0.3) -> None:
        self.alpha = float(np.clip(alpha, 0.05, 1.0))
        self._state: dict[int, tuple[float, float]] = {}

    def smooth_detections(self, dets: sv.Detections) -> sv.Detections:
        if len(dets) == 0 or dets.data is None:
            return dets
        kf_vx = dets.data.get("kf_vx")
        kf_vy = dets.data.get("kf_vy")
        if kf_vx is None or kf_vy is None:
            return dets
        n = len(dets)
        out_vx = np.full(n, np.nan, dtype=np.float32)
        out_vy = np.full(n, np.nan, dtype=np.float32)
        tids = dets.tracker_id if dets.tracker_id is not None else np.full(n, -1, dtype=int)
        a = self.alpha
        for i in range(n):
            tid = int(tids[i])
            raw_x, raw_y = float(kf_vx[i]), float(kf_vy[i])
            has_raw = np.isfinite(raw_x) and np.isfinite(raw_y)
            if tid >= 0 and has_raw:
                if tid in self._state:
                    px, py = self._state[tid]
                    sx = a * raw_x + (1.0 - a) * px
                    sy = a * raw_y + (1.0 - a) * py
                else:
                    sx, sy = raw_x, raw_y
                self._state[tid] = (sx, sy)
                out_vx[i], out_vy[i] = sx, sy
            elif tid >= 0 and tid in self._state:
                out_vx[i], out_vy[i] = self._state[tid]
            elif has_raw:
                out_vx[i], out_vy[i] = raw_x, raw_y
        data = dict(dets.data)
        data["kf_vx"] = out_vx
        data["kf_vy"] = out_vy
        return sv.Detections(
            xyxy=dets.xyxy,
            class_id=dets.class_id,
            tracker_id=dets.tracker_id,
            confidence=dets.confidence,
            data=data,
        )


class JoystickDotSmoother:
    """EMA on joystick offset from ellipse center."""

    def __init__(self, *, alpha: float = 0.32) -> None:
        self.alpha = float(np.clip(alpha, 0.05, 1.0))
        self._offset: dict[int, tuple[float, float]] = {}

    def smooth(
        self, tracker_id: int, cx: float, cy: float, px: float, py: float
    ) -> tuple[int, int]:
        if tracker_id < 0:
            return int(round(px)), int(round(py))
        ox, oy = px - cx, py - cy
        a = self.alpha
        if tracker_id in self._offset:
            pox, poy = self._offset[tracker_id]
            ox = a * ox + (1.0 - a) * pox
            oy = a * oy + (1.0 - a) * poy
        self._offset[tracker_id] = (ox, oy)
        return int(round(cx + ox)), int(round(cy + oy))


# ---------------------------------------------------------------------------
# Detector factories
# ---------------------------------------------------------------------------

def create_player_detector(
    *,
    backend: str = "yolo",
    model_path: str | None = None,
    model_id: str = "football-players-detection-3zvbc/11",
    device: str = "cpu",
    threshold: float = 0.5,
    api_key: str | None = None,
):
    """Return a callable(frame_bgr) -> sv.Detections for player detection."""
    if backend == "yolo":
        from ultralytics import YOLO
        path = model_path or PLAYER_DETECTION_MODEL_PATH
        model = YOLO(str(path)).to(device=device)

        def _detect_yolo(frame: np.ndarray) -> sv.Detections:
            results = model.predict(frame, conf=threshold, verbose=False, device=device)[0]
            dets = sv.Detections.from_ultralytics(results)
            # Map YOLO classes to canonical role IDs (same mapping as sports main.py).
            return dets

        return _detect_yolo

    if backend == "inference":
        import os
        from inference import get_model
        key = api_key or os.environ.get("ROBOFLOW_API_KEY")
        if not key:
            raise RuntimeError("Set ROBOFLOW_API_KEY for inference player detection.")
        model = get_model(model_id=model_id, api_key=key)

        def _detect_inf(frame: np.ndarray) -> sv.Detections:
            result = model.infer(frame, confidence=threshold)[0]
            return sv.Detections.from_inference(result)

        return _detect_inf

    raise ValueError(f"Unknown player detector backend: {backend!r}")


def create_pitch_keypoint_detector(
    *,
    backend: str = "yolo",
    model_path: str | None = None,
    model_id: str = "football-field-detection-f07vi/15",
    device: str = "cpu",
    api_key: str | None = None,
) -> Callable[[np.ndarray], sv.KeyPoints]:
    """Return a callable(frame_bgr) -> sv.KeyPoints for pitch keypoint detection."""
    if backend == "yolo":
        from ultralytics import YOLO
        path = model_path or PITCH_DETECTION_MODEL_PATH
        model = YOLO(str(path)).to(device=device)

        def _kp_yolo(frame: np.ndarray) -> sv.KeyPoints:
            result = model.predict(frame, conf=0.3, verbose=False, device=device)[0]
            return sv.KeyPoints.from_ultralytics(result)

        return _kp_yolo

    if backend == "inference":
        import os
        from analytics.homography import keypoints_from_inference_field
        from inference import get_model
        key = api_key or os.environ.get("ROBOFLOW_API_KEY")
        if not key:
            raise RuntimeError("Set ROBOFLOW_API_KEY for inference pitch keypoints.")
        model = get_model(model_id=model_id, api_key=key)

        def _kp_inf(frame: np.ndarray) -> sv.KeyPoints:
            result = model.infer(frame, confidence=0.3)[0]
            return keypoints_from_inference_field(result)

        return _kp_inf

    raise ValueError(f"Unknown pitch detector backend: {backend!r}")


# ---------------------------------------------------------------------------
# Team classifier fitting (STRIDE=60, mirrors existing main.py)
# ---------------------------------------------------------------------------

def fit_team_classifier(
    cap: cv2.VideoCapture,
    player_detector_fn: Callable[[np.ndarray], sv.Detections] | None = None,
    *,
    device: str = "cpu",
    stride: int = STRIDE,
    max_frames: int | None = None,
    det_by_frame: dict[int, sv.Detections] | None = None,
) -> TeamClassifier:
    """Sample frames at STRIDE and fit TeamClassifier on player crops.

    When ``det_by_frame`` is supplied (e.g. from the on-disk detection cache) the boxes
    are taken from it instead of re-running the detector.
    """
    team_classifier = TeamClassifier(device=device)
    crops: list[np.ndarray] = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if max_frames is not None and frame_idx > max_frames:
            break
        if frame_idx % stride != 0:
            continue
        if det_by_frame is not None:
            dets = det_by_frame.get(frame_idx)
            if dets is None:
                continue
        else:
            dets = player_detector_fn(frame)
        players = dets[dets.class_id == PLAYER_CLASS_ID]
        crops.extend(get_crops(frame, players))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if crops:
        team_classifier.fit(crops)
    return team_classifier


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

TEAM_COLORS = [sv.Color.from_hex("#FF1493"), sv.Color.from_hex("#00BFFF")]
REFEREE_COLOR = sv.Color.from_hex("#FFD700")
NEUTRAL_COLOR = sv.Color.from_hex("#CCCCCC")

MS_TO_KMH = 3.6


def _team_color(team: int) -> sv.Color:
    if team in (0, 1):
        return TEAM_COLORS[team]
    return NEUTRAL_COLOR


def draw_team_ellipses(
    frame: np.ndarray,
    detections: sv.Detections,
    *,
    thickness: int = 2,
    show_ids: bool = True,
) -> None:
    """Draw ground-contact ellipses colored by team (optional track-id labels)."""
    if len(detections) == 0 or detections.data is None:
        return
    teams = detections.data.get("team", np.full(len(detections), TEAM_NONE))
    tids = detections.tracker_id if detections.tracker_id is not None else np.full(len(detections), -1)
    for i, xyxy in enumerate(detections.xyxy):
        class_id = int(detections.class_id[i])
        team = int(teams[i])
        if class_id == REFEREE_CLASS_ID:
            color = REFEREE_COLOR
        else:
            color = _team_color(team)
        x1, y1, x2, y2 = xyxy
        # Match sv.EllipseAnnotator geometry: feet-centered open arc whose semi-axes
        # are the full box width and 0.35x that width (not half), so the ground ellipse
        # sits under the player at the expected visible size.
        width = float(x2 - x1)
        cx = int((x1 + x2) / 2)
        cy = int(y2)
        rx = max(int(width), 1)
        ry = max(int(0.35 * width), 1)
        cv2.ellipse(
            frame,
            (cx, cy),
            (rx, ry),
            0.0, -45, 235,
            color.as_bgr(),
            thickness,
            cv2.LINE_AA,
        )
        tid = int(tids[i])
        if show_ids and tid >= 0:
            # Clean id chip (shared badge styling) instead of a raw putText number.
            _draw_chip(frame, f"#{tid}", (cx, cy - ry - 8), team_bgr=color.as_bgr())


# ── radial Kalman speed badge ──
_SPEED_BADGE_BG_BGR = (16, 18, 24)
SPEED_SPRINT_MS = 5.0

# Shared chip styling: speed, distance and id markers all use the same compact
# translucent box (dark bg, team-colored side rail, thin border, shadowed value)
# so they read as one family of chips.
_CHIP_FONT = cv2.FONT_HERSHEY_DUPLEX
_CHIP_VALUE_SCALE = 0.48
_CHIP_VALUE_THICK = 1
_CHIP_PAD_X = 3
_CHIP_PAD_Y = 2
_CHIP_RAIL_W = 2
_CHIP_TEXT_BGR = (240, 242, 248)


def _format_speed_value(speed_m_s: float) -> str:
    """Round to 1 decimal m/s — readable without false precision."""
    return f"{round(max(0.0, float(speed_m_s)), 1):.1f}"


def _format_distance_value(distance_m: float) -> str:
    """Whole metres with an explicit unit — e.g. "12 m"."""
    return f"{max(0, int(round(float(distance_m))))} m"


def _draw_text_shadow(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    font_scale: float,
    color_bgr: tuple[int, int, int],
    thickness: int = 1,
    shadow_offset: tuple[int, int] = (1, 1),
) -> None:
    x, y = org
    sx, sy = shadow_offset
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, text, (x + sx, y + sy), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, font_scale, color_bgr, thickness, cv2.LINE_AA)


def _chip_box_size(text: str) -> tuple[int, int, int, int]:
    """Pixel footprint of a value chip: (box_w, box_h, value_h, baseline)."""
    (vw, vh), baseline = cv2.getTextSize(text, _CHIP_FONT, _CHIP_VALUE_SCALE, _CHIP_VALUE_THICK)
    box_w = vw + _CHIP_PAD_X * 2 + _CHIP_RAIL_W
    box_h = vh + baseline + _CHIP_PAD_Y * 2
    return box_w, box_h, vh, baseline


def _draw_chip(
    frame: np.ndarray,
    text: str,
    center: tuple[float, float],
    *,
    team_bgr: tuple[int, int, int],
    border_bgr: tuple[int, int, int] | None = None,
) -> tuple[int, int]:
    """Draw a compact translucent value chip centered at ``center``.

    Shared by the speed badge, the distance marker and the track-id label so the
    three share one visual style. Returns the drawn (box_w, box_h).
    """
    box_w, box_h, vh, _baseline = _chip_box_size(text)
    bcx, bcy = center
    x0 = int(round(bcx - box_w * 0.5))
    y0 = int(round(bcy - box_h * 0.5))
    fh, fw = frame.shape[:2]
    x0 = int(np.clip(x0, 2, max(2, fw - box_w - 2)))
    y0 = int(np.clip(y0, 2, max(2, fh - box_h - 2)))
    x1, y1 = x0 + box_w, y0 + box_h
    if border_bgr is None:
        border_bgr = tuple(int(c * 0.7) for c in team_bgr)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), _SPEED_BADGE_BG_BGR, -1)
    cv2.rectangle(overlay, (x0, y0), (x0 + _CHIP_RAIL_W, y1), team_bgr, -1)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), border_bgr, 1, cv2.LINE_AA)
    frame[:] = cv2.addWeighted(overlay, 0.62, frame, 0.38, 0)
    _draw_text_shadow(
        frame, text, (x0 + _CHIP_RAIL_W + _CHIP_PAD_X, y0 + _CHIP_PAD_Y + vh),
        font_scale=_CHIP_VALUE_SCALE, color_bgr=_CHIP_TEXT_BGR, thickness=_CHIP_VALUE_THICK,
    )
    return box_w, box_h


def _speed_badge_radial(
    cx: float, cy: float, px: float, py: float, vx: float, vy: float,
    *, min_speed_px: float = DEFAULT_MIN_SPEED_PX,
) -> tuple[float, float]:
    """Outward ray for the badge: stick direction, else Kalman velocity, else up."""
    dx, dy = float(px - cx), float(py - cy)
    dist = float(np.hypot(dx, dy))
    if dist >= 1.0:
        return dx / dist, dy / dist
    speed = float(np.hypot(vx, vy))
    if np.isfinite(vx) and np.isfinite(vy) and speed >= min_speed_px:
        return vx / speed, vy / speed
    return 0.0, -1.0


def draw_speed_badge(
    frame: np.ndarray,
    speed_m_s: float,
    cx: float, cy: float, px: int, py: int, vx: float, vy: float,
    *,
    team_bgr: tuple[int, int, int],
    dot_radius: int,
    min_speed_px: float = DEFAULT_MIN_SPEED_PX,
) -> None:
    """Speed chip riding just outside the smoothed joystick dot along the stick ray."""
    value = _format_speed_value(speed_m_s)
    _box_w, box_h, _vh, _baseline = _chip_box_size(value)

    ux, uy = _speed_badge_radial(cx, cy, float(px), float(py), vx, vy, min_speed_px=min_speed_px)
    outward = float(dot_radius) + 5.0 + box_h * 0.5
    bcx, bcy = float(px) + ux * outward, float(py) + uy * outward

    # Sprint speeds get the full team color border; otherwise a dimmed variant.
    border = team_bgr if speed_m_s >= SPEED_SPRINT_MS else tuple(int(c * 0.7) for c in team_bgr)
    _draw_chip(frame, value, (bcx, bcy), team_bgr=team_bgr, border_bgr=border)


def draw_speed_legend(frame: np.ndarray) -> None:
    """Global unit key — numbers on players are m/s."""
    text = "speed  m/s"
    font = cv2.FONT_HERSHEY_DUPLEX
    scale, thick = 0.42, 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    pad_x, pad_y = 8, 5
    x0, y1 = 12, frame.shape[0] - 12
    y0 = y1 - th - baseline - pad_y * 2
    x1 = x0 + tw + pad_x * 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (16, 18, 24), -1)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (70, 72, 82), 1, cv2.LINE_AA)
    frame[:] = cv2.addWeighted(overlay, 0.62, frame, 0.38, 0)
    _draw_text_shadow(
        frame, text, (x0 + pad_x, y1 - pad_y - baseline),
        font_scale=scale, color_bgr=(220, 222, 230), thickness=thick,
    )


# ── joystick-dot geometry ──
# Dot radius scales with bbox width; a speed "stick" drives how far the dot reaches
# toward the drawn ground-ellipse edge, keeping dot size and deflection proportional
# to the player bbox at every scale (instead of a fixed radius / fixed arm).
JOYSTICK_MIN_SPEED_PX = 0.5
JOYSTICK_MAX_SPEED_PX = 4.0
JOYSTICK_ELLIPSE_THICKNESS = 2.0  # matches draw_team_ellipses stroke


def _dot_radius_for_ellipse(semi_axis_a: float) -> int:
    """Scale the dot with bbox width (the ellipse horizontal semi-axis ``a = x2-x1``)."""
    return int(np.clip(round(semi_axis_a * 0.13), 3, 8))


def _ellipse_extent_in_direction(a: float, b: float, ux: float, uy: float) -> float:
    """Distance from ellipse center to its edge along a unit direction."""
    denom = (b * ux) ** 2 + (a * uy) ** 2
    if denom < 1e-12:
        return float(min(a, b))
    return float((a * b) / np.sqrt(denom))


def _joystick_dot_reach(
    stick: float,
    a: float,
    b: float,
    ux: float,
    uy: float,
    *,
    dot_radius: float,
    ellipse_thickness: float = JOYSTICK_ELLIPSE_THICKNESS,
) -> float:
    """Center distance for a joystick dot tied to the drawn ellipse.

    At full deflection (``stick == 1``) the dot center sits on the ellipse edge plus
    half the stroke and its own radius, so the filled circle rides the outer edge.
    """
    edge = _ellipse_extent_in_direction(a, b, ux, uy)
    outer = edge + 0.5 * ellipse_thickness + dot_radius
    return float(stick) * outer


def kalman_speed_stick(
    speed_px: float,
    *,
    min_speed_px: float = JOYSTICK_MIN_SPEED_PX,
    max_speed_px: float = JOYSTICK_MAX_SPEED_PX,
) -> float | None:
    """Map Kalman speed (px/frame) to a joystick deflection in [0, 1] (sqrt curve)."""
    if not np.isfinite(speed_px) or speed_px < min_speed_px:
        return None
    if max_speed_px <= min_speed_px:
        return 1.0
    linear = float(np.clip((speed_px - min_speed_px) / (max_speed_px - min_speed_px), 0.0, 1.0))
    # Slight curve so typical jogging reads closer to the ellipse edge.
    return float(np.sqrt(linear))


def draw_joystick_dots(
    frame: np.ndarray,
    detections: sv.Detections,
    joystick_smoother: JoystickDotSmoother | None = None,
    *,
    dot_radius: int = 5,
    arm_scale: float = 1.4,
    speed_by_tid: dict[int, float] | None = None,
    show_speed: bool = False,
    min_speed_ms: float = 0.0,
) -> None:
    """Draw a team-colored directional velocity dot on each player, optional speed badge.

    The dot color matches the player's team; referees / unassigned rows get no dot.
    When ``show_speed`` and ``speed_by_tid`` are given, a radial m/s badge rides the dot.

    Dot radius scales with bbox width; a speed "stick" drives how far the dot reaches
    toward the ground-ellipse edge at full deflection. ``dot_radius`` / ``arm_scale`` are
    retained for call-site compatibility but are no longer used (the dot is sized per
    player).
    """
    del dot_radius, arm_scale  # superseded by per-player ellipse-relative sizing
    if len(detections) == 0 or detections.data is None:
        return
    kf_vx = detections.data.get("kf_vx")
    kf_vy = detections.data.get("kf_vy")
    if kf_vx is None or kf_vy is None:
        return
    teams = detections.data.get("team", np.full(len(detections), TEAM_NONE))
    tids = detections.tracker_id if detections.tracker_id is not None else np.full(len(detections), -1)
    for i, xyxy in enumerate(detections.xyxy):
        cls = int(detections.class_id[i])
        if cls in (BALL_CLASS_ID, REFEREE_CLASS_ID):
            continue
        team = int(teams[i])
        if team not in (0, 1):
            continue
        color = _team_color(team).as_bgr()
        vx, vy = float(kf_vx[i]), float(kf_vy[i])
        if not (np.isfinite(vx) and np.isfinite(vy)):
            continue
        speed = float(np.hypot(vx, vy))
        if speed < DEFAULT_MIN_SPEED_PX:
            continue
        x1, y1, x2, y2 = xyxy
        cx = (float(x1) + float(x2)) / 2.0
        cy = float(y2)
        # Ellipse semi-axes match draw_team_ellipses: a = full bbox width, b = 0.35a.
        a = float(x2 - x1)
        b = 0.35 * a
        radius = _dot_radius_for_ellipse(a)
        px, py = cx, cy
        stick = kalman_speed_stick(speed)
        if stick is not None:
            ux, uy = vx / speed, vy / speed
            reach = _joystick_dot_reach(stick, a, b, ux, uy, dot_radius=float(radius))
            px, py = cx + ux * reach, cy + uy * reach
        tid = int(tids[i])
        if joystick_smoother is not None:
            px, py = joystick_smoother.smooth(tid, cx, cy, px, py)
        ipx, ipy = int(round(px)), int(round(py))
        cv2.circle(frame, (ipx, ipy), radius, color, -1, cv2.LINE_AA)
        if show_speed and speed_by_tid is not None and tid >= 0:
            spd = speed_by_tid.get(tid)
            if spd is not None and spd >= min_speed_ms:
                draw_speed_badge(
                    frame, float(spd), cx, cy, ipx, ipy, vx, vy,
                    team_bgr=color, dot_radius=radius,
                )


def draw_distance_labels(
    frame: np.ndarray,
    detections: sv.Detections,
    distance_by_tid: dict[int, float],
) -> None:
    """Draw cumulative distance above each tracked player as a styled chip.

    Uses the shared chip styling (see ``_draw_chip``) so the distance marker matches
    the speed badge, and carries an explicit metre unit (e.g. "12 m"). Used by the
    DISTANCE and PLAYER_FOCUS features.
    """
    if len(detections) == 0 or detections.tracker_id is None:
        return
    teams = (
        detections.data.get("team", np.full(len(detections), TEAM_NONE))
        if detections.data else np.full(len(detections), TEAM_NONE)
    )
    for i, tid in enumerate(detections.tracker_id):
        tid = int(tid)
        if tid < 0:
            continue
        dist_m = distance_by_tid.get(tid)
        if dist_m is None:
            continue
        xyxy = detections.xyxy[i]
        x1, y1, x2 = xyxy[0], xyxy[1], xyxy[2]
        cx = (float(x1) + float(x2)) / 2.0
        label = _format_distance_value(dist_m)
        _, box_h, _vh, _baseline = _chip_box_size(label)
        cy = float(y1) - 8 - box_h * 0.5
        team = int(teams[i])
        team_bgr = _team_color(team).as_bgr()
        _draw_chip(frame, label, (cx, cy), team_bgr=team_bgr)


def track_id_color(tid: int) -> tuple[int, int, int]:
    """Deterministic BGR color from tracker id (hashed palette)."""
    import colorsys
    hue = (tid * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_goals_on_pitch(
    config,
    *,
    left_defender_team: int,
    right_defender_team: int,
    team_colors: list = TEAM_COLORS,
    padding: int = 50,
    scale: float = 0.1,
    pitch: np.ndarray,
    fill_alpha: float = 0.38,
) -> np.ndarray:
    """Highlight each goal mouth in the defending team's color."""
    w = config.width
    length = config.length
    gbw = getattr(config, "goal_box_width", 1832)
    gbl = getattr(config, "goal_box_length", 550)
    y0, y1 = (w - gbw) / 2, (w + gbw) / 2

    def _goal_patch(goal_x_cm: float, defender: int, depth_cm: float) -> None:
        color = team_colors[defender % len(team_colors)].as_bgr()
        mouth_x = int(goal_x_cm * scale) + padding
        py0 = int(y0 * scale) + padding
        py1 = int(y1 * scale) + padding
        inner_x = int((goal_x_cm + depth_cm) * scale) + padding
        x_lo, x_hi = sorted((mouth_x, inner_x))
        overlay = pitch.copy()
        cv2.rectangle(overlay, (x_lo, py0), (x_hi, py1), color, -1)
        cv2.addWeighted(overlay, fill_alpha, pitch, 1.0 - fill_alpha, 0, pitch)
        cv2.line(pitch, (mouth_x, py0), (mouth_x, py1), color, 5, cv2.LINE_AA)
        cv2.line(pitch, (mouth_x, py0), (mouth_x, py1), (255, 255, 255), 1, cv2.LINE_AA)

    _goal_patch(0.0, left_defender_team, gbl)
    _goal_patch(float(length), right_defender_team, -gbl)
    return pitch


# ── radar / minimap geometry + opacity ──
# The minimap panel is alpha-blended over the footage (not pasted opaque) so both the
# radar and the underlying game frame stay visible behind it.
RADAR_MINIMAP_SCALE = 0.065
RADAR_MINIMAP_PAD = 30
RADAR_MINIMAP_ALPHA = 0.6  # blend weight of the minimap panel over the footage


def overlay_minimap(
    frame: np.ndarray,
    radar: np.ndarray,
    *,
    margin_x: int = 12,
    margin_y: int = 12,
    alpha: float = RADAR_MINIMAP_ALPHA,
) -> None:
    """Alpha-blend a radar panel into the bottom-right corner of ``frame`` in place.

    Blending (instead of an opaque paste) keeps both the radar and the footage it sits
    on top of legible; shared by every demo that draws a minimap.
    """
    rh, rw = radar.shape[:2]
    fh, fw = frame.shape[:2]
    x0 = fw - rw - margin_x
    y0 = fh - rh - margin_y
    if x0 < 0 or y0 < 0:
        return
    roi = frame[y0:y0 + rh, x0:x0 + rw]
    cv2.addWeighted(radar, alpha, roi, 1.0 - alpha, 0, roi)


def draw_radar_minimap(
    frame: np.ndarray,
    detections: sv.Detections,
    transformer: ViewTransformer | None,
    *,
    minimap_scale: float = RADAR_MINIMAP_SCALE,
    padding: int = RADAR_MINIMAP_PAD,
    margin_x: int = 12,
    margin_y: int = 12,
    locked_goal_defenders: tuple[int, int] | None = None,
    alpha: float = RADAR_MINIMAP_ALPHA,
) -> np.ndarray:
    """Overlay a translucent radar minimap in the bottom-right corner of frame."""
    from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
    from sports.configs.soccer import SoccerPitchConfiguration

    if transformer is None:
        return frame
    config = SoccerPitchConfiguration()
    radar = draw_pitch(config=config, padding=padding, scale=minimap_scale)

    teams = None
    if detections.data is not None:
        teams = detections.data.get("team")

    # ── goal shading by defending team ──────────────────────────────────────
    pmask_goal = player_mask(detections)
    left_def, right_def = (locked_goal_defenders or (TEAM_NONE, TEAM_NONE))
    if locked_goal_defenders is None and pmask_goal.any() and teams is not None:
        from analytics.goalkeepers import infer_goal_defenders
        xy_goal = feet_xy(detections[pmask_goal]).astype(np.float32)
        xy_goal_cm = transformer.transform_points(xy_goal)
        left_def, right_def = infer_goal_defenders(xy_goal_cm, np.asarray(teams)[pmask_goal])
    if left_def in (0, 1) and right_def in (0, 1):
        radar = draw_goals_on_pitch(
            config, left_defender_team=left_def, right_defender_team=right_def,
            team_colors=TEAM_COLORS, padding=padding, scale=minimap_scale, pitch=radar,
        )

    pmask = player_mask(detections)
    if pmask.any():
        pdet = detections[pmask]
        xy = feet_xy(pdet).astype(np.float32)
        xy_cm = transformer.transform_points(xy)
        p_teams = teams[pmask] if teams is not None else np.full(pmask.sum(), TEAM_NONE)
        for tid_val in (0, 1):
            mask = p_teams == tid_val
            if mask.any():
                radar = draw_points_on_pitch(
                    config=config,
                    xy=xy_cm[mask],
                    face_color=TEAM_COLORS[tid_val],
                    edge_color=sv.Color.BLACK,
                    radius=int(18 * minimap_scale / 0.1),
                    padding=padding,
                    scale=minimap_scale,
                    pitch=radar,
                )

    overlay_minimap(frame, radar, margin_x=margin_x, margin_y=margin_y, alpha=alpha)
    return frame


def draw_trace_on_minimap(
    radar: np.ndarray,
    trace_cm: np.ndarray,
    color_bgr: tuple[int, int, int],
    *,
    padding: int = 30,
    scale: float = 0.065,
    thickness: int = 2,
    smooth_window: int | None = None,
    margin_cm: float = 80.0,
) -> np.ndarray:
    """Draw a pitch-cm polyline trace on a minimap image.

    Cleans the *visible* trace only — this does not touch the minimap homography
    (the no-mirror keypoint H stays as-is): off-pitch homography spikes are dropped
    via :func:`valid_pitch_cm` (``margin_cm`` ~80) and the remaining points are
    median-smoothed over a small odd window (``HOMOGRAPHY_PITCH_SMOOTH``) so the
    radar polyline reads cleanly instead of jittering on raw per-frame warps. NaN
    points are skipped.
    """
    from analytics.homography import valid_pitch_cm

    window = HOMOGRAPHY_PITCH_SMOOTH if smooth_window is None else smooth_window
    trace_cm = np.asarray(trace_cm, dtype=np.float64)
    if trace_cm.ndim == 2 and len(trace_cm):
        # Drop off-pitch warps so spurious spikes never reach the polyline.
        on_pitch = valid_pitch_cm(trace_cm, margin_cm=margin_cm)
        trace_cm = trace_cm[on_pitch]
        # Median-smooth the kept points (purely cosmetic; homography is unchanged).
        trace_cm = _smooth_trajectory(trace_cm, window)
    pts: list[tuple[int, int]] = []
    for pt in trace_cm:
        if np.any(np.isnan(pt)):
            continue
        px = int(pt[0] * scale) + padding
        py = int(pt[1] * scale) + padding
        pts.append((px, py))
    if len(pts) >= 2:
        cv2.polylines(radar, [np.array(pts, dtype=np.int32)], False, color_bgr, thickness, cv2.LINE_AA)
    if pts:
        cv2.circle(radar, pts[-1], 7, color_bgr, -1, cv2.LINE_AA)
        cv2.circle(radar, pts[-1], 7, (255, 255, 255), 1, cv2.LINE_AA)
    return radar


def build_trace_minimap(
    detections: sv.Detections,
    transformer: ViewTransformer | None,
    trace_by_tid: dict[int, list[np.ndarray]],
    focus_tid: int | None = None,
    *,
    config=None,
    scale: float = RADAR_MINIMAP_SCALE,
    padding: int = RADAR_MINIMAP_PAD,
    locked_goal_defenders: tuple[int, int] | None = None,
) -> np.ndarray:
    """Build a radar minimap with per-track colored traces + current player dots.

    Shared by DISTANCE and PLAYER_FOCUS so both render the same radar. With
    ``focus_tid=None`` (follow-all / DISTANCE) every track keeps its own colored
    trace + dot; a focus id (PLAYER_FOCUS single-focus) draws only that player's
    trace + dot and omits everyone else.
    """
    from sports.annotators.soccer import draw_pitch
    from sports.configs.soccer import SoccerPitchConfiguration
    from analytics.homography import valid_pitch_cm

    if config is None:
        config = SoccerPitchConfiguration()
    radar = draw_pitch(config=config, padding=padding, scale=scale)
    if locked_goal_defenders is not None:
        left_def, right_def = locked_goal_defenders
        if left_def in (0, 1) and right_def in (0, 1):
            radar = draw_goals_on_pitch(
                config, left_defender_team=left_def, right_defender_team=right_def,
                team_colors=TEAM_COLORS, padding=padding, scale=scale, pitch=radar,
            )
    # draw traces (smoothing + outlier filtering handled in draw_trace_on_minimap).
    # Single-focus draws only the focused track's trace; follow-all keeps every
    # track's own colored trace.
    for tid, pts in trace_by_tid.items():
        if len(pts) < 2:
            continue
        if focus_tid is not None and tid != focus_tid:
            continue
        trace = np.stack(pts, axis=0)
        color = track_id_color(tid)
        # smooth_window defaults to HOMOGRAPHY_PITCH_SMOOTH — the shared window also
        # used for distance integration, so the drawn and integrated trajectory match.
        radar = draw_trace_on_minimap(radar, trace, color, padding=padding, scale=scale)

    # draw current player positions
    if transformer is not None and len(detections):
        pmask = player_mask(detections)
        if pmask.any():
            pdet = detections[pmask]
            xy = feet_xy(pdet).astype(np.float32)
            xy_cm = transformer.transform_points(xy)
            # Drop off-pitch warps so outlier dots stop rendering on the radar.
            on_pitch = valid_pitch_cm(xy_cm, config, margin_cm=80.0)
            tids_p = pdet.tracker_id if pdet.tracker_id is not None else np.full(len(pdet), -1)
            teams = pdet.data.get("team", np.full(len(pdet), TEAM_NONE)) if pdet.data else np.full(len(pdet), TEAM_NONE)
            for i in range(len(pdet)):
                if not on_pitch[i]:
                    continue
                t_id = int(tids_p[i])
                team = int(teams[i])
                # Single-focus shows only the focused player's dot (matches the
                # focus-only trace); follow-all draws every player's dot.
                if focus_tid is not None and t_id != focus_tid:
                    continue
                if team in (0, 1):
                    color = TEAM_COLORS[team].as_bgr()
                else:
                    color = track_id_color(t_id) if t_id >= 0 else (150, 150, 150)
                pt_cm = xy_cm[i]
                px = int(pt_cm[0] * scale) + padding
                py = int(pt_cm[1] * scale) + padding
                cv2.circle(radar, (px, py), 8, color, -1, cv2.LINE_AA)
                cv2.circle(radar, (px, py), 8, (255, 255, 255), 1, cv2.LINE_AA)
    return radar


def annotate_motion_overlay(
    frame: np.ndarray,
    detections: sv.Detections,
    *,
    joystick_smoother: JoystickDotSmoother | None,
    speed_by_tid: dict[int, float] | None,
    distance_by_tid: dict[int, float] | None,
    show_ids: bool = False,
) -> None:
    """Draw team ellipses + instant-speed badge + cumulative-distance chip per player.

    The instant Kalman ground speed rides the joystick dot (existing speed badge) and
    the cumulative-distance chip sits above the player, so the two metric chips stack
    without overlapping. Shared by DISTANCE and PLAYER_FOCUS so the on-player metric
    chips look identical across both demos.
    """
    draw_team_ellipses(frame, detections, show_ids=show_ids)
    draw_joystick_dots(
        frame, detections, joystick_smoother,
        speed_by_tid=speed_by_tid, show_speed=speed_by_tid is not None,
    )
    if distance_by_tid is not None:
        draw_distance_labels(frame, detections, distance_by_tid)


def render_follow_all_frame(
    frame: np.ndarray,
    detections: sv.Detections,
    *,
    joystick_smoother: JoystickDotSmoother | None,
    speed_by_tid: dict[int, float] | None,
    distance_by_tid: dict[int, float] | None,
    trace_by_tid: dict[int, list[np.ndarray]],
    radar_transformer: ViewTransformer | None,
    locked_goal_defenders: tuple[int, int] | None = None,
    focus_tid: int | None = None,
    show_legend: bool = True,
    show_ids: bool = False,
) -> None:
    """Annotate every player with speed+distance chips and overlay the trace radar.

    This is the shared "follow-all" look: it backs both PLAYER_FOCUS (no --track-id)
    and DISTANCE so the two render identically apart from DISTANCE's leaderboard
    end-card. Mutates ``frame`` in place.
    """
    annotate_motion_overlay(
        frame, detections,
        joystick_smoother=joystick_smoother,
        speed_by_tid=speed_by_tid,
        distance_by_tid=distance_by_tid,
        show_ids=show_ids,
    )
    if show_legend:
        draw_speed_legend(frame)
    mini_radar = build_trace_minimap(
        detections, radar_transformer, trace_by_tid, focus_tid,
        locked_goal_defenders=locked_goal_defenders,
    )
    overlay_minimap(frame, mini_radar)


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def open_video(path: str) -> tuple[cv2.VideoCapture, float, int, int]:
    """Return (cap, fps, width, height)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, fps, w, h


# ---------------------------------------------------------------------------
# Distance / kinematics
# ---------------------------------------------------------------------------

MAX_PHYSICAL_STEP_MS = 12.5   # ~45 km/h hard cap on a single-frame step
# One shared median window (odd) applied to the pitch trajectory for BOTH distance
# integration and the visible radar trace polyline, so the shown and integrated
# trajectory stay consistent. Larger = smoother trace (and smoother distance).
HOMOGRAPHY_PITCH_SMOOTH = 9   # median window on pitch trajectory (distance + drawn trace)
HOMOGRAPHY_XY_SMOOTH = 5      # moving-average window on image feet before warping to pitch


def _smooth_xy(xy: np.ndarray, window: int) -> np.ndarray:
    """Moving-average smooth an (N, 2) trajectory."""
    if len(xy) < 2 or window <= 1:
        return xy
    pad = window // 2
    out = xy.copy()
    for k in range(2):
        col = xy[:, k]
        padded = np.pad(col, pad, mode="edge")
        out[:, k] = np.array([np.mean(padded[i:i + window]) for i in range(len(col))])
    return out


from dataclasses import dataclass as _dataclass, field as _field


@_dataclass
class PlayerTrack:
    track_id: int
    frames: list[int] = _field(default_factory=list)
    xy: list[tuple[float, float]] = _field(default_factory=list)
    box_h: list[float] = _field(default_factory=list)
    distance_m: float = 0.0
    cumulative_m: np.ndarray | None = None  # running total aligned with frames


def collect_tracks(detections_iter) -> dict[int, PlayerTrack]:
    """Accumulate per-track feet positions + box heights from a detections iterator.

    detections_iter yields (frame_idx, sv.Detections).
    """
    tracks: dict[int, PlayerTrack] = {}
    for frame_idx, dets in detections_iter:
        if dets.tracker_id is None or len(dets) == 0:
            continue
        pmask = player_mask(dets)
        if not pmask.any():
            continue
        fxy = feet_xy(dets)
        heights = dets.xyxy[:, 3] - dets.xyxy[:, 1]
        for i in np.flatnonzero(pmask):
            tid = int(dets.tracker_id[i])
            if tid < 0:
                continue
            track = tracks.setdefault(tid, PlayerTrack(tid))
            track.frames.append(int(frame_idx))
            track.xy.append((float(fxy[i, 0]), float(fxy[i, 1])))
            track.box_h.append(float(heights[i]))
    return tracks


def _smooth_trajectory(pos: np.ndarray, window: int) -> np.ndarray:
    """Median-smooth (N, 2) pitch positions; NaN-filled from track median first."""
    if len(pos) < 2 or window <= 1:
        return pos
    out = pos.copy()
    for k in range(2):
        col = out[:, k].copy()
        valid = np.isfinite(col)
        if not valid.any():
            continue
        col[~valid] = float(np.median(col[valid]))
        pad = window // 2
        padded = np.pad(col, pad, mode="edge")
        out[:, k] = np.array([np.median(padded[i:i + window]) for i in range(len(col))])
    return out


def _pitch_positions(
    xy: np.ndarray,
    frames: np.ndarray,
    frame_transforms: dict,
) -> np.ndarray:
    """Per-sample pitch positions (m) warped with that frame's homography."""
    n = len(frames)
    pos = np.full((n, 2), np.nan, dtype=np.float64)
    for i in range(n):
        t = frame_transforms.get(int(frames[i]))
        if t is None:
            continue
        pt = t.transform_points(xy[i].reshape(1, 2).astype(np.float32))[0]
        pos[i] = pt / 100.0   # cm → m
    return pos


def _distance_from_smoothed(
    pos: np.ndarray,
    frames: np.ndarray,
    fps: float,
    *,
    pitch_smooth: int = HOMOGRAPHY_PITCH_SMOOTH,
    max_step_ms: float = MAX_PHYSICAL_STEP_MS,
) -> tuple[float, np.ndarray]:
    """Smooth pitch trajectory then integrate 1-frame hops.  Returns (total_m, steps)."""
    smooth_pos = _smooth_trajectory(pos, pitch_smooth)
    n = len(frames)
    step_m = np.zeros(max(n - 1, 0), dtype=np.float64)
    for j in range(1, n):
        if np.any(np.isnan(smooth_pos[j])) or np.any(np.isnan(smooth_pos[j - 1])):
            continue
        dt = (int(frames[j]) - int(frames[j - 1])) / fps
        if dt <= 0:
            continue
        dist = float(np.linalg.norm(smooth_pos[j] - smooth_pos[j - 1]))
        if dist / dt <= max_step_ms:
            step_m[j - 1] = dist
    return float(np.sum(step_m)), step_m


def compute_kinematics(
    tracks: dict[int, PlayerTrack],
    fps: float,
    *,
    mode: str = "homography",
    frame_transforms: dict[int, Any] | None = None,
    min_frames: int = 2,
) -> dict[int, PlayerTrack]:
    """Compute cumulative distance for each track.

    Uses homography mode with gated frame_transforms for distance only.
    Internal speed is not exposed in analytics (Kalman speed is displayed instead).
    """
    for track in tracks.values():
        if len(track.frames) < min_frames:
            track.distance_m = 0.0
            track.cumulative_m = np.zeros(len(track.frames))
            continue

        xy = np.asarray(track.xy, dtype=np.float64)
        frames = np.asarray(track.frames)
        transforms = frame_transforms or {}

        if mode == "homography" and transforms:
            # Pre-smooth image feet (moving average) before warping to pitch space.
            xy = _smooth_xy(xy, HOMOGRAPHY_XY_SMOOTH)
            pos = _pitch_positions(xy, frames, transforms)
            dist_m, step_m = _distance_from_smoothed(pos, frames, fps)
            track.distance_m = dist_m
            # build cumulative array aligned with frames
            cum = np.zeros(len(frames), dtype=np.float64)
            for j in range(len(step_m)):
                cum[j + 1] = cum[j] + step_m[j]
            track.cumulative_m = cum
        else:
            track.distance_m = 0.0
            track.cumulative_m = np.zeros(len(track.frames))
    return tracks


def cumulative_distance_at_frame(track: PlayerTrack, frame_idx: int) -> float | None:
    """Running distance (m) at frame_idx after compute_kinematics."""
    if track.cumulative_m is None or frame_idx not in track.frames:
        return None
    return float(track.cumulative_m[track.frames.index(frame_idx)])
