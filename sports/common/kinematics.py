import numpy as np
import supervision as sv
from trackers.utils.state_representations import XCYCWHStateEstimator, XYXYStateEstimator

from sports.configs.soccer import GOALKEEPER_CLASS_ID, PLAYER_CLASS_ID

DEFAULT_MIN_SPEED_PX = 0.5


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
