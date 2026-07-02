import os
from pathlib import Path
from typing import Callable, List

import cv2
import numpy as np
import supervision as sv
from trackers import BoTSORTTracker, ByteTrackTracker
from ultralytics import YOLO

try:
    from inference import get_model
except ImportError:
    get_model = None

from sports.common.kinematics import kalman_velocity_arrays
from sports.common.team import TeamClassifier
from sports.configs.soccer import (
    GOALKEEPER_CLASS_ID,
    PLAYER_CLASS_ID,
    REFEREE_CLASS_ID,
    TEAM_NONE,
)

PLAYER_DETECTION_MODEL_PATH = str(
    Path(__file__).resolve().parents[2]
    / "examples"
    / "soccer"
    / "data"
    / "football-player-detection.pt"
)

DEFAULT_PLAYER_MODEL_ID = "football-players-detection-3zvbc/11"

DEFAULT_TRACK_ACTIVATION_THRESHOLD = 0.55
DEFAULT_HIGH_CONF_DET_THRESHOLD = 0.6
DEFAULT_MINIMUM_IOU_THRESHOLD_FIRST_ASSOC = 0.15

REFEREE_PLAYER_IOU_THRESHOLD = 0.25
REFEREE_TRACK_IOU_THRESHOLD = 0.4
GOALKEEPER_PLAYER_IOU_THRESHOLD = 0.25

STRIDE = 60


def create_player_tracker(
    frame_rate: float,
    kind: str = "botsort",
    track_activation_threshold: float = DEFAULT_TRACK_ACTIVATION_THRESHOLD,
    high_conf_det_threshold: float = DEFAULT_HIGH_CONF_DET_THRESHOLD,
    minimum_iou_threshold_first_assoc: float = DEFAULT_MINIMUM_IOU_THRESHOLD_FIRST_ASSOC,
):
    """Return a multi-object tracker configured for football players."""
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
    tracker.reset()
    return tracker


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
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
    t0 = (
        pl_xy[players_team_id == 0].mean(axis=0)
        if (players_team_id == 0).any()
        else pl_xy.mean(axis=0)
    )
    t1 = (
        pl_xy[players_team_id == 1].mean(axis=0)
        if (players_team_id == 1).any()
        else pl_xy.mean(axis=0)
    )
    result = []
    for xy in gk_xy:
        d0 = float(np.linalg.norm(xy - t0))
        d1 = float(np.linalg.norm(xy - t1))
        result.append(0 if d0 <= d1 else 1)
    return np.array(result, dtype=int)


def _split_detection_roles(dets: sv.Detections):
    return (
        dets[dets.class_id == PLAYER_CLASS_ID],
        dets[dets.class_id == GOALKEEPER_CLASS_ID],
        dets[dets.class_id == REFEREE_CLASS_ID],
    )


def _detection_overlap_with_referees(
    subject: sv.Detections,
    referees: sv.Detections,
    iou_threshold: float = REFEREE_PLAYER_IOU_THRESHOLD,
) -> np.ndarray:
    if len(subject) == 0 or len(referees) == 0:
        return np.zeros(len(subject), dtype=bool)
    ious = sv.box_iou_batch(subject.xyxy, referees.xyxy)
    overlap = ious.max(axis=1) >= iou_threshold
    if overlap.all():
        return overlap
    rcx = (referees.xyxy[:, 0] + referees.xyxy[:, 2]) / 2.0
    rcy = (referees.xyxy[:, 1] + referees.xyxy[:, 3]) / 2.0
    x1, y1, x2, y2 = (
        subject.xyxy[:, 0],
        subject.xyxy[:, 1],
        subject.xyxy[:, 2],
        subject.xyxy[:, 3],
    )
    for i in np.flatnonzero(~overlap):
        inside = (rcx >= x1[i]) & (rcx <= x2[i]) & (rcy >= y1[i]) & (rcy <= y2[i])
        if inside.any():
            overlap[i] = True
    return overlap


def _suppress_goalkeepers_overlapping_players(
    goalkeepers: sv.Detections,
    players: sv.Detections,
    iou_threshold: float = GOALKEEPER_PLAYER_IOU_THRESHOLD,
) -> sv.Detections:
    if len(goalkeepers) == 0 or len(players) == 0:
        return goalkeepers
    ious = sv.box_iou_batch(goalkeepers.xyxy, players.xyxy)
    drop = ious.max(axis=1) >= iou_threshold
    return goalkeepers[~drop]


def filter_referees(
    dets: sv.Detections,
    blocked_tracker_ids=None,
    iou_threshold: float = REFEREE_PLAYER_IOU_THRESHOLD,
) -> sv.Detections:
    """Remove referee rows, overlapping players, and flagged tracklets."""
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


def _enforce_one_goalkeeper_per_team(
    dets: sv.Detections,
    frame_width=None,
) -> sv.Detections:
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

    def _rank(i: int):
        conf = float(dets.confidence[i]) if dets.confidence is not None else 0.0
        height = float(dets.xyxy[i, 3] - dets.xyxy[i, 1])
        return (conf, height)

    by_group = {}
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


def build_trackable_detections(raw, frame_width=None) -> sv.Detections:
    """Return players and goalkeepers to feed the tracker."""
    if raw is None or len(raw) == 0:
        return sv.Detections.empty()
    cleaned = _enforce_one_goalkeeper_per_team(raw, frame_width=frame_width)
    cleaned = filter_referees(cleaned)
    players, gks, _ = _split_detection_roles(cleaned)
    gks = _suppress_goalkeepers_overlapping_players(gks, players)
    if len(players) or len(gks):
        return sv.Detections.merge([players, gks])
    return sv.Detections.empty()


def _combine_for_referee_check(
    tracked: sv.Detections,
    referees: sv.Detections,
) -> sv.Detections:
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


def collect_referee_tracker_ids(
    frames,
    iou_threshold: float = REFEREE_TRACK_IOU_THRESHOLD,
):
    """Return tracker ids that ever strongly overlap a referee box."""
    flagged = set()
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


def drop_blocked_tracker_ids(dets: sv.Detections, blocked_tracker_ids) -> sv.Detections:
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


def create_player_detector(
    backend: str = "yolo",
    model_path=None,
    model_id: str = DEFAULT_PLAYER_MODEL_ID,
    device: str = "cpu",
    threshold: float = 0.5,
    api_key=None,
):
    """Return a callable(frame_bgr) -> sv.Detections for player detection."""
    if backend == "yolo":
        path = model_path or PLAYER_DETECTION_MODEL_PATH
        model = YOLO(str(path)).to(device=device)

        def _detect_yolo(frame: np.ndarray) -> sv.Detections:
            results = model.predict(frame, conf=threshold, verbose=False, device=device)[0]
            return sv.Detections.from_ultralytics(results)

        return _detect_yolo

    if backend == "inference":
        if get_model is None:
            raise RuntimeError("Install the 'inference' package for inference player detection.")
        key = api_key or os.environ.get("ROBOFLOW_API_KEY")
        if not key:
            raise RuntimeError("Set ROBOFLOW_API_KEY for inference player detection.")
        model = get_model(model_id=model_id, api_key=key)

        def _detect_inf(frame: np.ndarray) -> sv.Detections:
            result = model.infer(frame, confidence=threshold)[0]
            return sv.Detections.from_inference(result)

        return _detect_inf

    raise ValueError(f"Unknown player detector backend: {backend!r}")


def fit_team_classifier(
    cap: cv2.VideoCapture,
    player_detector_fn: Callable[[np.ndarray], sv.Detections] = None,
    device: str = "cpu",
    stride: int = STRIDE,
    max_frames=None,
    det_by_frame=None,
) -> TeamClassifier:
    """Sample frames at stride and fit TeamClassifier on player crops."""
    team_classifier = TeamClassifier(device=device)
    crops = []
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


def open_video(path: str):
    """Return (cap, fps, width, height)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, fps, w, h


def collect_team_frames(
    source_video_path: str,
    player_detector_fn=None,
    team_classifier=None,
    tracker=None,
    needs_frame: bool = True,
    max_frames=None,
    detections_by_frame=None,
    capture_velocity: bool = False,
):
    """Run one detect-track-classify pass and return team and referee frames."""
    cap = cv2.VideoCapture(source_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {source_video_path}")

    frames = []
    referee_frames = []
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if max_frames is not None and frame_idx > max_frames:
                break
            if detections_by_frame is not None:
                raw = detections_by_frame.get(frame_idx)
                if raw is None:
                    raw = sv.Detections.empty()
            else:
                raw = player_detector_fn(frame)
            trackable = build_trackable_detections(raw, frame_width=float(frame.shape[1]))
            _, _, refs = _split_detection_roles(raw)
            tracked = (
                tracker.update(trackable, frame=frame if needs_frame else None)
                if len(trackable)
                else sv.Detections.empty()
            )
            team_arr = np.full(len(tracked), TEAM_NONE, dtype=int)
            if len(tracked):
                t_players = tracked[tracked.class_id == PLAYER_CLASS_ID]
                if len(t_players):
                    team_arr[tracked.class_id == PLAYER_CLASS_ID] = team_classifier.predict(
                        get_crops(frame, t_players)
                    )
            data = {"team": team_arr}
            if capture_velocity:
                kf_vx, kf_vy = kalman_velocity_arrays(tracked, tracker)
                data["kf_vx"] = kf_vx
                data["kf_vy"] = kf_vy
            dets = sv.Detections(
                xyxy=tracked.xyxy,
                class_id=tracked.class_id,
                tracker_id=tracked.tracker_id,
                confidence=tracked.confidence,
                data=data,
            )
            frames.append((frame_idx, dets))
            referee_frames.append((frame_idx, _combine_for_referee_check(tracked, refs)))
    finally:
        cap.release()
    return frames, referee_frames
