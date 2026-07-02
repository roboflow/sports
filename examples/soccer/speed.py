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
    KalmanVelocitySmoother,
    feet_xy,
)
from sports.common.tracking import open_video
from sports.common.video_tracking import VideoTrackingSession, build_video_tracking_session
from sports.common.view import ViewTransformer


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
