import cv2
import numpy as np
import supervision as sv

from sports.annotators.motion import draw_joystick_dots, draw_team_ellipses
from sports.common.clip import ClipAnalysis, compute_clip_analysis
from sports.common.kinematics import (
    JoystickDotSmoother,
    KalmanVelocitySmoother,
    merge_kalman_velocity,
)
from sports.common.tracking import (
    build_trackable_detections,
    create_player_tracker,
    drop_blocked_tracker_ids,
    get_crops,
    open_video,
    resolve_goalkeepers_team_id,
)
from sports.common.team import apply_team_lock, relock_detection_teams
from sports.configs.soccer import GOALKEEPER_CLASS_ID, PLAYER_CLASS_ID, TEAM_NONE


def run_direction(args, analysis=None) -> None:
    """Render team-colored ellipses and image-space Kalman direction dots."""
    if analysis is not None:
        _run_direction_replay(args, analysis)
        return
    _run_direction_standalone(args, compute_clip_analysis(args))


def _run_direction_standalone(args, analysis: ClipAnalysis) -> None:
    fps, width, height = analysis.fps, analysis.width, analysis.height
    team_classifier = analysis.team_classifier
    det_by_frame = analysis.det_by_frame
    needs_frame = analysis.needs_frame
    locks = analysis.locks()
    team_lock = locks.team_lock
    blocked_ids = analysis.blocked_ids

    tracker = create_player_tracker(fps, kind=args.tracker)
    vel_smoother = KalmanVelocitySmoother(alpha=0.3)
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
                    player_crops = get_crops(frame, t_players)
                    player_teams = team_classifier.predict(player_crops)
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

            tracked_with_teams = sv.Detections(
                xyxy=tracked.xyxy,
                class_id=tracked.class_id,
                tracker_id=tracked.tracker_id,
                confidence=tracked.confidence,
                data={**(tracked.data or {}), "team": team_arr},
            )

            dets_with_vel = merge_kalman_velocity(tracked_with_teams, tracker)
            dets_with_vel = vel_smoother.smooth_detections(dets_with_vel)
            dets_with_vel = relock_detection_teams(dets_with_vel, team_lock)

            annotated = frame.copy()
            draw_team_ellipses(annotated, dets_with_vel)
            draw_joystick_dots(annotated, dets_with_vel, joy_smoother)
            sink.write_frame(annotated)

    cap.release()
    print(f"Wrote {args.target_video_path}")


def _run_direction_replay(args, analysis: ClipAnalysis) -> None:
    fps, width, height = analysis.fps, analysis.width, analysis.height
    locks = analysis.locks()

    vel_smoother = KalmanVelocitySmoother(alpha=0.3)
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
                frame_idx, tracked, locks=locks, vel_smoother=vel_smoother,
            )

            annotated = frame.copy()
            draw_team_ellipses(annotated, dets)
            draw_joystick_dots(annotated, dets, joy_smoother)
            sink.write_frame(annotated)

    cap.release()
    print(f"Wrote {args.target_video_path}")
