"""analytics/speed.py — Feature 2: Kalman ground speed with demo homography.

Displayed speed = Kalman ground speed via speed_transforms_gap_filled H map.
Shows m/s badge per tracked player + a translucent radar minimap.

Consumes a shared :class:`~analytics.clip_pipeline.ClipAnalysis` (computed if absent). When
invoked standalone (``analysis=None``) the render runs its own per-frame tracker step with
a single update and Kalman velocity read per frame. When the run-all orchestrator passes a
shared analysis, the render replays that single BoTSORT pass so tracking runs only once
for the run-all.
"""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

from analytics.clip_pipeline import ClipAnalysis, compute_clip_analysis
from analytics.goalkeepers import apply_goalkeeper_frame
from analytics.class_ids import GOALKEEPER_CLASS_ID, PLAYER_CLASS_ID, TEAM_NONE
from analytics.player_motion import (
    JoystickDotSmoother,
    KalmanSpeedDisplaySmoother,
    KalmanVelocitySmoother,
    merge_kalman_velocity,
    build_trackable_detections,
    create_player_tracker,
    draw_joystick_dots,
    draw_radar_minimap,
    draw_speed_legend,
    draw_team_ellipses,
    drop_blocked_tracker_ids,
    feet_xy,
    get_crops,
    kalman_ground_speed_m_s,
    open_video,
    resolve_goalkeepers_team_id,
)
from analytics.teams import apply_team_lock, relock_detection_teams

_GK_ASSIGNMENT = "goal_distance"


def run_speed(args, analysis: ClipAnalysis | None = None) -> None:
    """Render per-player Kalman ground-speed badges (m/s) via gap-filled H."""
    if analysis is not None:
        _run_speed_replay(args, analysis)
        return
    _run_speed_standalone(args, compute_clip_analysis(args, need_homography=True))


def _speed_by_tid(
    dets: sv.Detections,
    transformer,
    fps: float,
    speed_smoother: KalmanSpeedDisplaySmoother,
    *,
    only_tid: int | None = None,
) -> dict[int, float]:
    """Per-player smoothed Kalman ground speed (m/s) from the gap-filled speed H."""
    speed_by_tid: dict[int, float] = {}
    if dets.tracker_id is None or transformer is None or dets.data is None:
        return speed_by_tid
    kf_vx = dets.data.get("kf_vx")
    kf_vy = dets.data.get("kf_vy")
    if kf_vx is None:
        return speed_by_tid
    fxy = feet_xy(dets)
    for i, tid in enumerate(dets.tracker_id):
        tid = int(tid)
        if tid < 0:
            continue
        if only_tid is not None and tid != only_tid:
            continue
        vx, vy = float(kf_vx[i]), float(kf_vy[i])
        if not (np.isfinite(vx) and np.isfinite(vy)):
            continue
        speed_ms = kalman_ground_speed_m_s(
            fxy[i], np.array([vx, vy], dtype=np.float64), transformer, fps=fps,
        )
        if speed_ms is not None:
            speed_by_tid[tid] = speed_smoother.smooth(tid, float(speed_ms))
    return speed_by_tid


def _draw_speed_overlay(
    frame: np.ndarray,
    dets: sv.Detections,
    speed_by_tid: dict[int, float],
    joy_smoother: JoystickDotSmoother,
    *,
    show_ids: bool = False,
    show_legend: bool = True,
) -> None:
    """Team ellipses + speed badges + optional legend (in-place)."""
    draw_team_ellipses(frame, dets, show_ids=show_ids)
    draw_joystick_dots(
        frame, dets, joy_smoother,
        speed_by_tid=speed_by_tid, show_speed=True,
    )
    if show_legend:
        draw_speed_legend(frame)


def _draw_speed_frame(
    frame: np.ndarray,
    dets: sv.Detections,
    speed_by_tid: dict[int, float],
    joy_smoother: JoystickDotSmoother,
    radar_t,
    locked_goal_defenders,
    *,
    show_ids: bool = False,
) -> np.ndarray:
    """Team ellipses + speed badges + translucent radar minimap (shared by both paths)."""
    annotated = frame.copy()
    _draw_speed_overlay(
        annotated, dets, speed_by_tid, joy_smoother, show_ids=show_ids,
    )
    # radar minimap: no-mirror keypoint-radar H, with goal shading from the clip lock.
    if radar_t is not None:
        draw_radar_minimap(
            annotated, dets, radar_t,
            locked_goal_defenders=locked_goal_defenders,
        )
    return annotated


def _run_speed_standalone(args, analysis: ClipAnalysis) -> None:
    """Standalone render: own per-frame tracker step (single update per frame)."""
    fps, width, height = analysis.fps, analysis.width, analysis.height
    team_classifier = analysis.team_classifier
    det_by_frame = analysis.det_by_frame
    needs_frame = analysis.needs_frame
    metric = analysis.metric
    gap_filled = analysis.gap_filled
    radar_h_by_frame = analysis.radar_h_by_frame
    locks = analysis.locks(_GK_ASSIGNMENT)
    team_lock, gk_lock = locks.team_lock, locks.gk_lock
    locked_goal_defenders = locks.locked_goal_defenders
    blocked_ids = analysis.blocked_ids
    show_ids = bool(getattr(args, "show_track_ids", False))

    tracker = create_player_tracker(fps, kind=args.tracker)
    vel_smoother = KalmanVelocitySmoother(alpha=0.3)
    speed_smoother = KalmanSpeedDisplaySmoother(alpha=0.3)
    joy_smoother = JoystickDotSmoother(alpha=0.32)

    cap, _, _, _ = open_video(args.source_video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    with sv.VideoSink(args.target_video_path, sv.VideoInfo(width, height, fps)) as sink:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if args.max_frames is not None and frame_idx > args.max_frames:
                break

            raw_dets = det_by_frame.get(frame_idx)
            if raw_dets is None:
                raw_dets = sv.Detections.empty()

            # Referees excluded; at most one goalkeeper per team (shared trackable builder).
            trackable = build_trackable_detections(raw_dets, frame_width=float(width))
            tracked = (
                tracker.update(trackable, frame=frame if needs_frame else None)
                if len(trackable) else sv.Detections.empty()
            )
            tracked = drop_blocked_tracker_ids(tracked, blocked_ids)

            # ── team classification ────────────────────────────────────────
            team_arr = np.full(len(tracked), TEAM_NONE, dtype=int)
            if len(tracked):
                t_players = tracked[tracked.class_id == PLAYER_CLASS_ID]
                if len(t_players):
                    pl_teams = team_classifier.predict(get_crops(frame, t_players))
                    team_arr[tracked.class_id == PLAYER_CLASS_ID] = pl_teams
                team_arr = apply_team_lock(team_arr, tracked.class_id, tracked.tracker_id, team_lock)
                if _GK_ASSIGNMENT == "centroid":
                    t_gks = tracked[tracked.class_id == GOALKEEPER_CLASS_ID]
                    if len(t_gks) and (team_arr == 0).any() and (team_arr == 1).any():
                        gk_teams = resolve_goalkeepers_team_id(
                            tracked[tracked.class_id == PLAYER_CLASS_ID],
                            team_arr[tracked.class_id == PLAYER_CLASS_ID],
                            t_gks,
                        )
                        team_arr[tracked.class_id == GOALKEEPER_CLASS_ID] = gk_teams

            tracked = sv.Detections(
                xyxy=tracked.xyxy,
                class_id=tracked.class_id,
                tracker_id=tracked.tracker_id,
                confidence=tracked.confidence,
                data={**(tracked.data or {}), "team": team_arr},
            )

            # ── goal-distance GK assignment (per-frame + clip lock) ────────
            if _GK_ASSIGNMENT == "goal_distance":
                tracked = apply_goalkeeper_frame(
                    tracked, metric.radar_transforms.get(frame_idx), gk_lock
                )

            # ── Kalman velocity (read after the single update above) ───────
            dets = merge_kalman_velocity(tracked, tracker)
            dets = vel_smoother.smooth_detections(dets)
            # Clip-level lock has the final say so class-flipping keepers stay one colour.
            dets = relock_detection_teams(dets, team_lock)

            # ── per-player Kalman ground speed (gap-filled H) ──────────────
            speed_by_tid = _speed_by_tid(
                dets, gap_filled.get(frame_idx), fps, speed_smoother
            )

            annotated = _draw_speed_frame(
                frame, dets, speed_by_tid, joy_smoother,
                radar_h_by_frame.get(frame_idx), locked_goal_defenders,
                show_ids=show_ids,
            )
            sink.write_frame(annotated)

    cap.release()
    print(f"Wrote {args.target_video_path}")


def _run_speed_replay(args, analysis: ClipAnalysis) -> None:
    """Run-all render: replay the shared single BoTSORT pass (no second tracker)."""
    fps, width, height = analysis.fps, analysis.width, analysis.height
    gap_filled = analysis.gap_filled
    radar_h_by_frame = analysis.radar_h_by_frame
    locks = analysis.locks(_GK_ASSIGNMENT)
    locked_goal_defenders = locks.locked_goal_defenders
    show_ids = bool(getattr(args, "show_track_ids", False))

    vel_smoother = KalmanVelocitySmoother(alpha=0.3)
    speed_smoother = KalmanSpeedDisplaySmoother(alpha=0.3)
    joy_smoother = JoystickDotSmoother(alpha=0.32)
    tracked_lookup = analysis.tracked_by_frame()

    cap, _, _, _ = open_video(args.source_video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    with sv.VideoSink(args.target_video_path, sv.VideoInfo(width, height, fps)) as sink:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if args.max_frames is not None and frame_idx > args.max_frames:
                break

            tracked = tracked_lookup.get(frame_idx)
            if tracked is None:
                tracked = sv.Detections.empty()
            dets = analysis.decorate_replay_frame(
                frame_idx, tracked,
                gk_assignment=_GK_ASSIGNMENT, locks=locks, vel_smoother=vel_smoother,
            )
            speed_by_tid = _speed_by_tid(
                dets, gap_filled.get(frame_idx), fps, speed_smoother
            )
            annotated = _draw_speed_frame(
                frame, dets, speed_by_tid, joy_smoother,
                radar_h_by_frame.get(frame_idx), locked_goal_defenders,
                show_ids=show_ids,
            )
            sink.write_frame(annotated)

    cap.release()
    print(f"Wrote {args.target_video_path}")
