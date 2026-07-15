import cv2
import supervision as sv

from sports.annotators.motion import annotate_team_ellipses, draw_joystick_dots
from sports.common.kinematics import JoystickDotSmoother, KalmanVelocitySmoother
from sports.common.tracking import open_video
from sports.common.video_tracking import (
    VideoTrackingSession,
    build_video_tracking_session,
)


def run_direction(args, session=None) -> None:
    """Render team-colored ellipses and image-space Kalman direction dots."""
    if session is None:
        session = build_video_tracking_session(args)
    _render_direction(args, session)


def _render_direction(args, session: VideoTrackingSession) -> None:
    """Draw direction overlays using precomputed video tracking."""
    fps, width, height = session.fps, session.width, session.height
    locks = session.team_locks()
    vel_smoother = KalmanVelocitySmoother(alpha=0.3)
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
                    frame_idx=frame_idx,
                )

                annotated = frame.copy()
                annotated = annotate_team_ellipses(annotated, dets)
                draw_joystick_dots(annotated, dets, joy_smoother)
                sink.write_frame(annotated)
    finally:
        cap.release()

    print(f"Wrote {args.target_video_path}")
