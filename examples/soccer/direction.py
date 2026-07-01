import cv2
import supervision as sv

from sports.annotators.motion import annotate_team_ellipses, draw_joystick_dots
from sports.common.clip import ClipAnalysis, compute_clip_analysis
from sports.common.kinematics import JoystickDotSmoother, KalmanVelocitySmoother
from sports.common.tracking import open_video


def run_direction(args, analysis=None) -> None:
    """Render team-colored ellipses and image-space Kalman direction dots."""
    if analysis is None:
        analysis = compute_clip_analysis(args)
    _render_direction(args, analysis)


def _render_direction(args, analysis: ClipAnalysis) -> None:
    """Draw direction overlays using precomputed clip tracking."""
    fps, width, height = analysis.fps, analysis.width, analysis.height
    locks = analysis.locks()
    vel_smoother = KalmanVelocitySmoother(alpha=0.3)
    joy_smoother = JoystickDotSmoother(alpha=0.32)
    tracked_lookup = analysis.tracked_by_frame()

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
                dets = analysis.decorate_replay_frame(
                    frame_idx, tracked, locks=locks, vel_smoother=vel_smoother,
                )

                annotated = frame.copy()
                annotated = annotate_team_ellipses(annotated, dets)
                draw_joystick_dots(annotated, dets, joy_smoother)
                sink.write_frame(annotated)
    finally:
        cap.release()

    print(f"Wrote {args.target_video_path}")
