from dataclasses import dataclass, field
from typing import Any

import numpy as np
import supervision as sv
from trackers.utils.state_representations import XCYCWHStateEstimator, XYXYStateEstimator

from sports.configs.soccer import GOALKEEPER_CLASS_ID, PLAYER_CLASS_ID

DEFAULT_MIN_SPEED_PX = 0.5
MAX_PHYSICAL_STEP_MS = 12.5
HOMOGRAPHY_PITCH_SMOOTH = 9
HOMOGRAPHY_XY_SMOOTH = 5


def feet_xy(detections: sv.Detections) -> np.ndarray:
    """Bottom-center anchor for each detection row."""
    if len(detections) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)


def player_mask(detections: sv.Detections) -> np.ndarray:
    """True for outfield players and goalkeepers."""
    if len(detections) == 0 or detections.class_id is None:
        return np.zeros(0, dtype=bool)
    cls = detections.class_id.astype(int)
    return (cls == PLAYER_CLASS_ID) | (cls == GOALKEEPER_CLASS_ID)


def _kalman_feet_velocity_from_tracklet(tracklet):
    """Extract feet-referenced Kalman velocity from a tracker tracklet."""
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


def kalman_velocity_arrays(
    detections: sv.Detections,
    tracker,
    min_speed: float = DEFAULT_MIN_SPEED_PX,
):
    """Return per-row kf_vx and kf_vy arrays aligned with detections."""
    n = len(detections)
    kf_vx = np.full(n, np.nan, dtype=np.float32)
    kf_vy = np.full(n, np.nan, dtype=np.float32)
    if n == 0 or detections.tracker_id is None:
        return kf_vx, kf_vy

    id_to_vel = {}
    for tracklet in tracker.tracks:
        tid = int(tracklet.tracker_id)
        if tid < 0:
            continue
        vel = _kalman_feet_velocity_from_tracklet(tracklet)
        if vel is None:
            continue
        if float(np.linalg.norm(vel)) < min_speed:
            continue
        id_to_vel[tid] = vel

    for i, tid in enumerate(detections.tracker_id):
        vel = id_to_vel.get(int(tid))
        if vel is None:
            continue
        kf_vx[i] = float(vel[0])
        kf_vy[i] = float(vel[1])
    return kf_vx, kf_vy


def merge_kalman_velocity(
    dets: sv.Detections,
    player_tracker,
    min_speed: float = DEFAULT_MIN_SPEED_PX,
) -> sv.Detections:
    """Merge kf_vx and kf_vy onto detections from an already-updated tracker."""
    kf_vx, kf_vy = kalman_velocity_arrays(
        dets, player_tracker, min_speed=min_speed
    )
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


class KalmanVelocitySmoother:
    """Per-track EMA on kf_vx and kf_vy; holds last value on NaN."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = float(np.clip(alpha, 0.05, 1.0))
        self._state = {}

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

    def __init__(self, alpha: float = 0.32):
        self.alpha = float(np.clip(alpha, 0.05, 1.0))
        self._offset = {}

    def smooth(self, tracker_id: int, cx: float, cy: float, px: float, py: float):
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


@dataclass
class PlayerTrack:
    track_id: int
    frames: list[int] = field(default_factory=list)
    xy: list[tuple[float, float]] = field(default_factory=list)
    box_h: list[float] = field(default_factory=list)
    distance_m: float = 0.0
    cumulative_m: np.ndarray | None = None


def collect_tracks(detections_iter) -> dict[int, PlayerTrack]:
    """Accumulate per-track feet positions from a (frame_idx, detections) iterator."""
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
        pos[i] = pt / 100.0
    return pos


def _distance_from_smoothed(
    pos: np.ndarray,
    frames: np.ndarray,
    fps: float,
    *,
    pitch_smooth: int = HOMOGRAPHY_PITCH_SMOOTH,
    max_step_ms: float = MAX_PHYSICAL_STEP_MS,
) -> tuple[float, np.ndarray]:
    """Smooth pitch trajectory then integrate 1-frame hops."""
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
    """Compute cumulative distance for each track via gated homography."""
    for track in tracks.values():
        if len(track.frames) < min_frames:
            track.distance_m = 0.0
            track.cumulative_m = np.zeros(len(track.frames))
            continue

        xy = np.asarray(track.xy, dtype=np.float64)
        frames = np.asarray(track.frames)
        transforms = frame_transforms or {}

        if mode == "homography" and transforms:
            xy = _smooth_xy(xy, HOMOGRAPHY_XY_SMOOTH)
            pos = _pitch_positions(xy, frames, transforms)
            dist_m, step_m = _distance_from_smoothed(pos, frames, fps)
            track.distance_m = dist_m
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
