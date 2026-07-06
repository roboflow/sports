import cv2
import numpy as np
import supervision as sv

from sports.annotators.motion import (
    annotate_team_ellipses,
    build_trace_minimap,
    dim_frame,
    draw_distance_end_card,
    draw_distance_labels,
    draw_joystick_dots,
    draw_speed_legend,
    overlay_minimap,
    spotlight,
)
from sports.common.kinematics import (
    JoystickDotSmoother,
    KalmanVelocitySmoother,
    cumulative_distance_at_frame,
    feet_xy,
)
from sports.common.tracking import open_video
from sports.common.video_tracking import VideoTrackingSession, build_video_tracking_session
from speed import (
    KalmanSpeedDisplaySmoother,
    _annotate_speed_overlay,
    _speed_by_tid,
)


def run_distance(args, session=None) -> None:
    """Render direction dots + distance chips, trace minimap, and distance leaderboard end-card."""
    if session is None:
        session = build_video_tracking_session(args, need_homography=True)
    _render_speed_distance_traces(args, session, show_speed=False, append_end_card=True)


def _annotate_player_overlay(
    frame: np.ndarray,
    dets: sv.Detections,
    joy_smoother: JoystickDotSmoother,
    speed_by_tid: dict[int, float],
    *,
    show_speed: bool,
) -> None:
    """Team ellipses + direction dots; optional m/s badges when show_speed."""
    if show_speed:
        _annotate_speed_overlay(
            frame, dets, speed_by_tid, joy_smoother, show_legend=False,
        )
    else:
        annotate_team_ellipses(frame, dets)
        draw_joystick_dots(frame, dets, joy_smoother)


def _render_speed_distance_traces(
    args,
    session: VideoTrackingSession,
    *,
    show_speed: bool = True,
    focus_tid: int | None = None,
    append_end_card: bool = False,
) -> None:
    fps, width, height = session.fps, session.width, session.height
    gap_filled = session.gap_filled_transforms_by_frame
    minimap_transforms = session.minimap_transforms_by_frame
    locks = session.team_locks()
    raw_tracks = session.tracks
    tracked_lookup = session.tracked_by_frame()

    vel_smoother = KalmanVelocitySmoother(alpha=0.3)
    speed_smoother = KalmanSpeedDisplaySmoother(alpha=0.3)
    joy_smoother = JoystickDotSmoother(alpha=0.32)
    trace_by_tid: dict[int, list[np.ndarray]] = {}

    cap, _, _, _ = open_video(args.source_video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    try:
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
                dets = session.apply_replay_teams(
                    tracked,
                    locks=locks,
                    vel_smoother=vel_smoother,
                )

                radar_h = minimap_transforms.get(frame_idx)
                if radar_h is not None and dets.tracker_id is not None:
                    fxy = feet_xy(dets)
                    xy_cm = radar_h.transform_points(fxy.astype(np.float32))
                    for i, tid in enumerate(dets.tracker_id):
                        tid = int(tid)
                        if tid < 0:
                            continue
                        if focus_tid is not None and tid != focus_tid:
                            continue
                        trace_by_tid.setdefault(tid, []).append(xy_cm[i].copy())

                speed_by_tid: dict[int, float] = {}
                if show_speed:
                    speed_by_tid = _speed_by_tid(
                        dets, gap_filled.get(frame_idx), fps, speed_smoother,
                        only_tid=focus_tid,
                    )

                dist_by_tid: dict[int, float] = {}
                if dets.tracker_id is not None:
                    for tid in dets.tracker_id:
                        tid = int(tid)
                        track = raw_tracks.get(tid)
                        if track is None:
                            continue
                        dist = cumulative_distance_at_frame(track, frame_idx)
                        if dist is not None:
                            dist_by_tid[tid] = dist

                radar_transformer = radar_h
                if focus_tid is not None:
                    spotlight_mask = (
                        dets.tracker_id == focus_tid
                        if dets.tracker_id is not None
                        else np.zeros(len(dets), dtype=bool)
                    )
                    visible = bool(spotlight_mask.any())
                    if visible:
                        box = dets.xyxy[spotlight_mask][0]
                        cx_f = int((box[0] + box[2]) / 2)
                        cy_f = int(box[3])
                        annotated = spotlight(frame, cx_f, cy_f)
                    else:
                        annotated = dim_frame(frame)
                    marked = dets[spotlight_mask]
                    _annotate_player_overlay(
                        annotated, marked, joy_smoother, speed_by_tid, show_speed=show_speed,
                    )
                    draw_distance_labels(annotated, marked, dist_by_tid)
                    mini_radar = build_trace_minimap(
                        dets, radar_transformer, trace_by_tid, focus_tid,
                    )
                    overlay_minimap(annotated, mini_radar)
                else:
                    annotated = frame.copy()
                    _annotate_player_overlay(
                        annotated, dets, joy_smoother, speed_by_tid, show_speed=show_speed,
                    )
                    draw_distance_labels(annotated, dets, dist_by_tid)
                    if show_speed:
                        draw_speed_legend(annotated)
                    mini_radar = build_trace_minimap(
                        dets, radar_transformer, trace_by_tid, None,
                    )
                    overlay_minimap(annotated, mini_radar)
                sink.write_frame(annotated)

            if append_end_card:
                end_card = draw_distance_end_card(width, height, raw_tracks)
                n_end_frames = max(1, int(fps * 3))
                for _ in range(n_end_frames):
                    sink.write_frame(end_card)
    finally:
        cap.release()

    print(f"Wrote {args.target_video_path}")
