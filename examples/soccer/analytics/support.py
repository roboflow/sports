"""analytics/support.py — shared non-homography plumbing for player-motion analytics.

Ported from world_cup_projects (minimal slices; no pass/possession/carrier logic).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterator  # noqa: F401

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

# ── team classification ────────────────────────────────────────────────────────
STRIDE = 60  # frames between team-fit samples (matches existing main.py)


# ---------------------------------------------------------------------------
# Tracker factory
# ---------------------------------------------------------------------------

def create_player_tracker(
    frame_rate: float,
    *,
    kind: str = "botsort",
    track_activation_threshold: float = DEFAULT_TRACK_ACTIVATION_THRESHOLD,
    high_conf_det_threshold: float = DEFAULT_HIGH_CONF_DET_THRESHOLD,
    minimum_iou_threshold_first_assoc: float = DEFAULT_MINIMUM_IOU_THRESHOLD_FIRST_ASSOC,
):
    """Return a multi-object tracker configured for football players."""
    if kind == "bytetrack":
        return ByteTrackTracker(
            frame_rate=frame_rate,
            track_activation_threshold=track_activation_threshold,
            high_conf_det_threshold=high_conf_det_threshold,
        )
    return BoTSORTTracker(
        frame_rate=frame_rate,
        track_activation_threshold=track_activation_threshold,
        high_conf_det_threshold=high_conf_det_threshold,
        minimum_iou_threshold_first_assoc=minimum_iou_threshold_first_assoc,
        enable_cmc=(kind == "botsort"),
        cmc_method="sparseOptFlow",
    )


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
# Kalman velocity helpers  (ported from world_cup_projects/common/tracking_facing.py)
# ---------------------------------------------------------------------------

def _kalman_feet_velocity_from_tracklet(tracklet) -> np.ndarray | None:
    """Extract feet-referenced Kalman velocity from a tracker tracklet."""
    est = tracklet.state_estimator
    x = est.kf.x.flatten()
    if len(x) < 6:
        return None
    if isinstance(est, XYXYStateEstimator):
        vx = (float(x[4]) + float(x[6])) / 2.0 if len(x) > 6 else float(x[4])
        vy = float(x[7]) if len(x) > 7 else float(x[5])
    elif isinstance(est, XCYCWHStateEstimator):
        vx = float(x[4])
        vy = float(x[5]) + float(x[7]) / 2.0 if len(x) > 7 else float(x[5])
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
    player_detector_fn: Callable[[np.ndarray], sv.Detections],
    *,
    device: str = "cpu",
    stride: int = STRIDE,
    max_frames: int | None = None,
) -> TeamClassifier:
    """Sample frames at STRIDE and fit TeamClassifier on player crops."""
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
        cx = int((x1 + x2) / 2)
        cy = int(y2)
        rx = max(int((x2 - x1) / 2), 1)
        ry = max(int(rx * 0.35), 1)
        cv2.ellipse(
            frame,
            (cx, cy),
            (rx, ry),
            0, 0, 360,
            color.as_bgr(),
            thickness,
            cv2.LINE_AA,
        )
        tid = int(tids[i])
        if show_ids and tid >= 0:
            cv2.putText(
                frame, str(tid),
                (cx - 6, cy - ry - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                color.as_bgr(), 1, cv2.LINE_AA,
            )


# ── radial Kalman speed badge (ported from world_cup_projects/common/visual.py) ──
_SPEED_BADGE_BG_BGR = (16, 18, 24)
SPEED_SPRINT_MS = 5.0


def _format_speed_value(speed_m_s: float) -> str:
    """Round to 1 decimal m/s — readable without false precision."""
    return f"{round(max(0.0, float(speed_m_s)), 1):.1f}"


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
    font = cv2.FONT_HERSHEY_DUPLEX
    value_scale, value_thick = 0.48, 1
    (vw, vh), baseline = cv2.getTextSize(value, font, value_scale, value_thick)
    pad_x, pad_y, rail_w = 3, 2, 2
    box_w = vw + pad_x * 2 + rail_w
    box_h = vh + baseline + pad_y * 2

    ux, uy = _speed_badge_radial(cx, cy, float(px), float(py), vx, vy, min_speed_px=min_speed_px)
    outward = float(dot_radius) + 5.0 + box_h * 0.5
    bcx, bcy = float(px) + ux * outward, float(py) + uy * outward
    x0 = int(round(bcx - box_w * 0.5))
    y0 = int(round(bcy - box_h * 0.5))
    fh, fw = frame.shape[:2]
    x0 = int(np.clip(x0, 2, max(2, fw - box_w - 2)))
    y0 = int(np.clip(y0, 2, max(2, fh - box_h - 2)))
    x1, y1 = x0 + box_w, y0 + box_h

    border = team_bgr if speed_m_s >= SPEED_SPRINT_MS else tuple(int(c * 0.7) for c in team_bgr)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), _SPEED_BADGE_BG_BGR, -1)
    cv2.rectangle(overlay, (x0, y0), (x0 + rail_w, y1), team_bgr, -1)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), border, 1, cv2.LINE_AA)
    frame[:] = cv2.addWeighted(overlay, 0.62, frame, 0.38, 0)
    _draw_text_shadow(
        frame, value, (x0 + rail_w + pad_x, y0 + pad_y + vh),
        font_scale=value_scale, color_bgr=(240, 242, 248), thickness=value_thick,
    )


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

    The dot color matches the player's team (ported from world_cup
    ``draw_kalman_joystick_dots``); referees / unassigned rows get no dot. When
    ``show_speed`` and ``speed_by_tid`` are given, a radial m/s badge rides the dot.
    """
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
        cx = (x1 + x2) / 2.0
        cy = float(y2)
        rx = max((x2 - x1) / 2.0, 1.0)
        arm = rx * arm_scale
        px = cx + (vx / speed) * arm
        py = cy + (vy / speed) * arm * 0.35
        tid = int(tids[i])
        if joystick_smoother is not None:
            px, py = joystick_smoother.smooth(tid, cx, cy, px, py)
        ipx, ipy = int(round(px)), int(round(py))
        cv2.circle(frame, (ipx, ipy), dot_radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (ipx, ipy), dot_radius, (20, 20, 20), 1, cv2.LINE_AA)
        if show_speed and speed_by_tid is not None and tid >= 0:
            spd = speed_by_tid.get(tid)
            if spd is not None and spd >= min_speed_ms:
                draw_speed_badge(
                    frame, float(spd), cx, cy, ipx, ipy, vx, vy,
                    team_bgr=color, dot_radius=dot_radius,
                )


def draw_distance_labels(
    frame: np.ndarray,
    detections: sv.Detections,
    distance_by_tid: dict[int, float],
) -> None:
    """Draw cumulative distance label above each tracked player."""
    if len(detections) == 0 or detections.tracker_id is None:
        return
    for i, tid in enumerate(detections.tracker_id):
        tid = int(tid)
        if tid < 0:
            continue
        dist_m = distance_by_tid.get(tid)
        if dist_m is None:
            continue
        xyxy = detections.xyxy[i]
        x1, y1, x2 = xyxy[0], xyxy[1], xyxy[2]
        cx = int((x1 + x2) / 2)
        top = int(y1) - 8
        label = f"{dist_m:.0f}m"
        cv2.putText(
            frame, label,
            (cx - 10, top),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            (100, 255, 100), 1, cv2.LINE_AA,
        )


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
    """Highlight each goal mouth in the defending team's color (ported from world_cup)."""
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


def draw_radar_minimap(
    frame: np.ndarray,
    detections: sv.Detections,
    transformer: ViewTransformer | None,
    *,
    minimap_scale: float = 0.065,
    padding: int = 30,
    margin_x: int = 12,
    margin_y: int = 12,
    locked_goal_defenders: tuple[int, int] | None = None,
) -> np.ndarray:
    """Overlay a radar minimap in the bottom-right corner of frame."""
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

    rh, rw = radar.shape[:2]
    fh, fw = frame.shape[:2]
    x0 = fw - rw - margin_x
    y0 = fh - rh - margin_y
    if x0 >= 0 and y0 >= 0:
        frame[y0:y0 + rh, x0:x0 + rw] = radar
    return frame


def draw_trace_on_minimap(
    radar: np.ndarray,
    trace_cm: np.ndarray,
    color_bgr: tuple[int, int, int],
    *,
    padding: int = 30,
    scale: float = 0.065,
    thickness: int = 2,
) -> np.ndarray:
    """Draw a pitch-cm polyline trace on a minimap image."""
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
# Distance / kinematics  (ported from world_cup_projects/player_stats/speed_distance.py)
# ---------------------------------------------------------------------------

MAX_PHYSICAL_STEP_MS = 12.5   # ~45 km/h hard cap on a single-frame step
HOMOGRAPHY_PITCH_SMOOTH = 5   # median window on pitch trajectory before distance integration
HOMOGRAPHY_XY_SMOOTH = 5      # moving-average window on image feet before warping to pitch


def _smooth_xy(xy: np.ndarray, window: int) -> np.ndarray:
    """Moving-average smooth an (N, 2) trajectory (ported from world_cup speed_distance)."""
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
    min_frames: int = 10,
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
