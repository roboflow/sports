#!/usr/bin/env python3
"""One-off visual check: custom cv2 ellipses vs shared EllipseAnnotator + joystick dots.

Does not affect any demo mode. Writes side-by-side PNGs under data/renders/.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

_EXAMPLE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXAMPLE_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sports.annotators.motion import annotate_team_ellipses, draw_joystick_dots
from sports.configs.soccer import (
    GOALKEEPER_CLASS_ID,
    PLAYER_CLASS_ID,
    REFEREE_CLASS_ID,
    TEAM_NONE,
)

REFEREE_COLOR = sv.Color.from_hex("#FFD700")
TEAM_COLORS = [sv.Color.from_hex("#FF1493"), sv.Color.from_hex("#00BFFF")]
NEUTRAL_COLOR = sv.Color.from_hex("#CCCCCC")


def _team_color(team: int) -> sv.Color:
    if team in (0, 1):
        return TEAM_COLORS[team]
    return NEUTRAL_COLOR


def _draw_legacy_team_ellipses(
    frame: np.ndarray,
    detections: sv.Detections,
    thickness: int = 2,
) -> None:
    if len(detections) == 0 or detections.data is None:
        return
    teams = detections.data.get("team", np.full(len(detections), TEAM_NONE))
    for i, xyxy in enumerate(detections.xyxy):
        class_id = int(detections.class_id[i])
        team = int(teams[i])
        if class_id == REFEREE_CLASS_ID:
            color = REFEREE_COLOR
        else:
            color = _team_color(team)
        x1, y1, x2, y2 = xyxy
        width = float(x2 - x1)
        cx = int((x1 + x2) / 2)
        cy = int(y2)
        rx = max(int(width), 1)
        ry = max(int(0.35 * width), 1)
        cv2.ellipse(
            frame,
            (cx, cy),
            (rx, ry),
            0.0,
            -45,
            235,
            color.as_bgr(),
            thickness,
            cv2.LINE_AA,
        )


def _label_strip(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (16, 18, 24), -1)
    cv2.putText(
        out, text, (12, 26),
        cv2.FONT_HERSHEY_DUPLEX, 0.65, (240, 242, 248), 1, cv2.LINE_AA,
    )
    return out


def _render_pair(frame: np.ndarray, dets: sv.Detections) -> np.ndarray:
    left = frame.copy()
    right = frame.copy()
    _draw_legacy_team_ellipses(left, dets)
    draw_joystick_dots(left, dets)
    right = annotate_team_ellipses(right, dets)
    draw_joystick_dots(right, dets)
    left = _label_strip(left, "legacy cv2 ellipses (LINE_AA)")
    right = _label_strip(right, "sv.EllipseAnnotator (shared)")
    h = max(left.shape[0], right.shape[0])
    if left.shape[0] != h:
        left = cv2.copyMakeBorder(left, 0, h - left.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    if right.shape[0] != h:
        right = cv2.copyMakeBorder(right, 0, h - right.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    return np.hstack([left, right])


def _synthetic_detections() -> sv.Detections:
    """Players with mixed teams, velocity, and one referee."""
    xyxy = np.array([
        [120, 80, 180, 220],
        [300, 100, 370, 260],
        [520, 90, 590, 240],
        [700, 110, 760, 270],
    ], dtype=float)
    class_id = np.array([
        PLAYER_CLASS_ID, PLAYER_CLASS_ID, PLAYER_CLASS_ID, REFEREE_CLASS_ID,
    ], dtype=int)
    tracker_id = np.array([1, 2, 3, 4], dtype=int)
    team = np.array([0, 1, 0, TEAM_NONE], dtype=int)
    kf_vx = np.array([2.5, -1.8, 0.0, 0.0], dtype=np.float32)
    kf_vy = np.array([-1.2, 2.1, 0.0, 0.0], dtype=np.float32)
    return sv.Detections(
        xyxy=xyxy,
        class_id=class_id,
        tracker_id=tracker_id,
        data={"team": team, "kf_vx": kf_vx, "kf_vy": kf_vy},
    )


def _synthetic_frame() -> np.ndarray:
    frame = np.full((360, 960, 3), (34, 120, 34), dtype=np.uint8)
    for y in range(0, 360, 40):
        cv2.line(frame, (0, y), (960, y), (28, 100, 28), 1)
    return frame


def run_synthetic(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dets = _synthetic_detections()
    pair = _render_pair(_synthetic_frame(), dets)
    out = out_dir / "ellipse_compare_synthetic.png"
    cv2.imwrite(str(out), pair)
    return out


def run_video_from_cache(
    source_video_path: str,
    cache_pkl: str,
    frame_indices: list[int],
    out_dir: Path,
) -> list[Path]:
    """Render comparison frames using precomputed detections (no TeamClassifier)."""
    import pickle

    from trackers import BoTSORTTracker

    from sports.common.kinematics import kalman_velocity_arrays, merge_kalman_velocity
    from sports.configs.soccer import GOALKEEPER_CLASS_ID, PLAYER_CLASS_ID

    with open(cache_pkl, "rb") as fh:
        payload = pickle.load(fh)
    det_by_frame = payload["frames"]

    cap = cv2.VideoCapture(source_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(source_video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    tracker = BoTSORTTracker(
        frame_rate=fps,
        track_activation_threshold=0.55,
        high_conf_det_threshold=0.6,
        minimum_iou_threshold_first_assoc=0.15,
        enable_cmc=True,
        cmc_method="sparseOptFlow",
    )
    tracker.reset()

    saved: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    want = set(frame_indices)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx > max(frame_indices):
            break

        raw_dets = det_by_frame.get(frame_idx)
        if raw_dets is None:
            trackable = sv.Detections.empty()
        else:
            raw = sv.Detections(
                xyxy=raw_dets["xyxy"],
                confidence=raw_dets["confidence"],
                class_id=raw_dets["class_id"],
            )
            mask = np.isin(raw.class_id, [PLAYER_CLASS_ID, GOALKEEPER_CLASS_ID])
            trackable = raw[mask]

        tracked = (
            tracker.update(trackable, frame=frame)
            if len(trackable)
            else sv.Detections.empty()
        )
        if frame_idx not in want or len(tracked) == 0:
            continue

        cx = (tracked.xyxy[:, 0] + tracked.xyxy[:, 2]) * 0.5
        team_arr = (cx >= width * 0.5).astype(int)
        kf_vx, kf_vy = kalman_velocity_arrays(tracked, tracker)
        dets = sv.Detections(
            xyxy=tracked.xyxy,
            class_id=tracked.class_id,
            tracker_id=tracked.tracker_id,
            confidence=tracked.confidence,
            data={"team": team_arr, "kf_vx": kf_vx, "kf_vy": kf_vy},
        )

        pair = _render_pair(frame, dets)
        out = out_dir / f"ellipse_compare_frame_{frame_idx:04d}.png"
        cv2.imwrite(str(out), pair)
        saved.append(out)

    cap.release()
    return saved


def run_video_frames(
    source_video_path: str,
    frame_indices: list[int],
    out_dir: Path,
    device: str,
) -> list[Path]:
    from sports.common.video_tracking import build_video_tracking_session
    from sports.common.kinematics import merge_kalman_velocity
    from sports.common.tracking import (
        build_trackable_detections,
        create_player_tracker,
        drop_blocked_tracker_ids,
        get_crops,
        open_video,
        resolve_goalkeepers_team_id,
    )
    from sports.common.team import apply_team_lock, relock_detection_teams
    from sports.configs.soccer import PLAYER_CLASS_ID

    class Args:
        source_video_path = source_video_path
        device = device
        max_frames = max(frame_indices)
        cache = True
        cache_dir = None
        tracker = "botsort"
        player_detector = "yolo"
        player_model_path = None
        player_model_id = "football-players-detection-3zvbc/11"
        api_key = None

    fork_models = Path("/Users/alexanderbodner/Documents/roboflow/sports_fork/examples/soccer/data")
    if Args.player_model_path is None and (fork_models / "football-player-detection.pt").exists():
        Args.player_model_path = str(fork_models / "football-player-detection.pt")

    session = build_video_tracking_session(Args())
    fps, width, height = session.fps, session.width, session.height
    det_by_frame = session.det_by_frame
    team_classifier = session.team_classifier
    needs_frame = session.needs_frame
    locks = session.team_locks(gk_assignment="centroid")
    team_lock = locks.team_lock
    blocked_ids = session.blocked_ids

    tracker = create_player_tracker(fps, kind=Args.tracker)
    vel_smoother = KalmanVelocitySmoother(alpha=0.3)

    cap, _, _, _ = open_video(source_video_path)
    saved: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx > max(frame_indices):
            break
        if frame_idx not in frame_indices:
            continue

        raw_dets = det_by_frame.get(frame_idx) or sv.Detections.empty()
        trackable = build_trackable_detections(raw_dets, frame_width=float(width))
        tracked = (
            tracker.update(trackable, frame=frame if needs_frame else None)
            if len(trackable)
            else sv.Detections.empty()
        )
        tracked = drop_blocked_tracker_ids(tracked, blocked_ids)

        team_arr = np.full(len(tracked), TEAM_NONE, dtype=int)
        if len(tracked):
            t_players = tracked[tracked.class_id == PLAYER_CLASS_ID]
            if len(t_players):
                player_teams = team_classifier.predict(get_crops(frame, t_players))
                team_arr[tracked.class_id == PLAYER_CLASS_ID] = player_teams
            team_arr = apply_team_lock(team_arr, tracked.tracker_id, team_lock)
            t_gks = tracked[tracked.class_id == GOALKEEPER_CLASS_ID]
            if len(t_gks) and (team_arr == 0).any() and (team_arr == 1).any():
                gk_teams = resolve_goalkeepers_team_id(
                    tracked[tracked.class_id == PLAYER_CLASS_ID],
                    team_arr[tracked.class_id == PLAYER_CLASS_ID],
                    t_gks,
                )
                team_arr[tracked.class_id == GOALKEEPER_CLASS_ID] = gk_teams

        dets = sv.Detections(
            xyxy=tracked.xyxy,
            class_id=tracked.class_id,
            tracker_id=tracked.tracker_id,
            confidence=tracked.confidence,
            data={**(tracked.data or {}), "team": team_arr},
        )
        dets = merge_kalman_velocity(dets, tracker)
        dets = vel_smoother.smooth_detections(dets)
        dets = relock_detection_teams(dets, team_lock)

        pair = _render_pair(frame, dets)
        out = out_dir / f"ellipse_compare_frame_{frame_idx:04d}.png"
        cv2.imwrite(str(out), pair)
        saved.append(out)

    cap.release()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ellipse annotator implementations.")
    parser.add_argument(
        "--source_video_path",
        default="/Users/alexanderbodner/Documents/roboflow/sports_fork/examples/soccer/data/08fd33_0.mp4",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frames", default="30,90,150", help="Comma-separated 1-based frame indices")
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument(
        "--out_dir",
        default=str(_EXAMPLE_DIR / "data" / "renders" / "ellipse_compare"),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    paths = [run_synthetic(out_dir)]

    if not args.synthetic_only and Path(args.source_video_path).exists():
        indices = [int(x.strip()) for x in args.frames.split(",") if x.strip()]
        cache_pkl = Path(
            "/Users/alexanderbodner/Documents/roboflow/sports_fork/examples/soccer/data/cache/detections-71e584ed8a68c11c.pkl"
        )
        try:
            if cache_pkl.exists():
                paths.extend(
                    run_video_from_cache(
                        args.source_video_path, str(cache_pkl), indices, out_dir,
                    )
                )
            else:
                paths.extend(
                    run_video_frames(args.source_video_path, indices, out_dir, args.device)
                )
        except Exception as exc:
            print(f"Video comparison skipped: {exc}")
    elif not args.synthetic_only:
        print(f"Video not found: {args.source_video_path}")

    print("Wrote:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
