import cv2
import numpy as np
import supervision as sv

from sports.annotators.motion import (
    annotate_team_ellipses,
    draw_joystick_dots,
    draw_radar_minimap,
    draw_speed_legend,
)
from sports.common.kinematics import (
    JoystickDotSmoother,
    KalmanSpeedDisplaySmoother,
    KalmanVelocitySmoother,
    feet_xy,
    kalman_ground_speed_m_s,
)
from sports.common.tracking import open_video
from sports.common.video_tracking import VideoTrackingSession, build_video_tracking_session


def run_speed(args, session=None) -> None:
    """Render per-player Kalman ground-speed badges (m/s) with a radar minimap."""
    if session is None:
        session = build_video_tracking_session(args, need_homography=True)
    _render_speed(args, session)


def _speed_by_tid(
    dets: sv.Detections,
    transformer,
    fps: float,
    speed_smoother: KalmanSpeedDisplaySmoother,
) -> dict[int, float]:
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
        vx, vy = float(kf_vx[i]), float(kf_vy[i])
        if not (np.isfinite(vx) and np.isfinite(vy)):
            continue
        speed_ms = kalman_ground_speed_m_s(
            fxy[i], np.array([vx, vy], dtype=np.float64), transformer, fps=fps,
        )
        if speed_ms is not None:
            speed_by_tid[tid] = speed_smoother.smooth(tid, float(speed_ms))
    return speed_by_tid


def _render_speed(args, session: VideoTrackingSession) -> None:
    fps, width, height = session.fps, session.width, session.height
    gap_filled = session.gap_filled_transforms_by_frame
    minimap_transforms = session.minimap_transforms_by_frame
    locks = session.team_locks()

    vel_smoother = KalmanVelocitySmoother(alpha=0.3)
    speed_smoother = KalmanSpeedDisplaySmoother(alpha=0.3)
    joy_smoother = JoystickDotSmoother(alpha=0.32)
    tracked_lookup = session.tracked_by_frame()

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
                speed_by_tid = _speed_by_tid(
                    dets, gap_filled.get(frame_idx), fps, speed_smoother,
                )

                annotated = frame.copy()
                annotated = annotate_team_ellipses(annotated, dets)
                draw_joystick_dots(
                    annotated, dets, joy_smoother,
                    speed_by_tid=speed_by_tid, show_speed=True,
                )
                draw_speed_legend(annotated)
                draw_radar_minimap(
                    annotated, dets, minimap_transforms.get(frame_idx),
                )
                sink.write_frame(annotated)
    finally:
        cap.release()

    print(f"Wrote {args.target_video_path}")
