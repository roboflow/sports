"""analytics/speed.py — Feature 2: Kalman ground speed with demo homography.

Displayed speed = Kalman ground speed via speed_transforms_gap_filled H map.
Shows m/s + km/h badge per tracked player.
"""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

from analytics.cache import (
    FrameCache,
    build_or_load_detections,
    build_or_load_keypoints,
)
from analytics.goalkeepers import apply_goalkeeper_frame, compute_clip_locks
from analytics.homography import MetricContext, build_metric_from_maps
from analytics.support import (
    GOALKEEPER_CLASS_ID,
    PLAYER_CLASS_ID,
    REFEREE_CLASS_ID,
    TEAM_NONE,
    KalmanSpeedDisplaySmoother,
    KalmanVelocitySmoother,
    JoystickDotSmoother,
    attach_kalman_velocity,
    create_player_detector,
    create_pitch_keypoint_detector,
    create_player_tracker,
    draw_speed_legend,
    draw_team_ellipses,
    draw_joystick_dots,
    draw_radar_minimap,
    feet_xy,
    fit_team_classifier,
    get_crops,
    kalman_ground_speed_m_s,
    open_video,
    resolve_goalkeepers_team_id,
    MS_TO_KMH,
)
from analytics.teams import apply_team_lock, relock_detection_teams


def run_speed(args) -> None:
    """Render per-player Kalman ground-speed badges (m/s + km/h) via gap-filled H."""
    cap, fps, width, height = open_video(args.source_video_path)

    player_model_id = getattr(args, "player_model_id", "football-players-detection-3zvbc/11")
    pitch_model_id = getattr(args, "pitch_model_id", "football-field-detection-f07vi/15")
    def _make_player_detector():
        return create_player_detector(
            backend=args.player_detector,
            model_path=getattr(args, "player_model_path", None),
            model_id=player_model_id,
            device=args.device,
            api_key=getattr(args, "api_key", None),
        )

    def _make_pitch_detector():
        return create_pitch_keypoint_detector(
            backend=args.pitch_detector,
            model_path=getattr(args, "pitch_model_path", None),
            model_id=pitch_model_id,
            device=args.device,
            api_key=getattr(args, "api_key", None),
        )

    # On-disk cache: detections + pitch keypoints are computed once, reused after.
    cache = FrameCache(
        args.source_video_path,
        cache_dir=getattr(args, "cache_dir", None),
        enabled=getattr(args, "cache", True),
        player_backend=args.player_detector,
        player_model_id=player_model_id,
        pitch_backend=args.pitch_detector,
        pitch_model_id=pitch_model_id,
    )
    det_by_frame = build_or_load_detections(
        args.source_video_path, _make_player_detector, cache, max_frames=args.max_frames
    )
    kp_by_frame = build_or_load_keypoints(
        args.source_video_path, _make_pitch_detector, cache, max_frames=args.max_frames
    )

    print("Fitting team classifier…")
    team_classifier = fit_team_classifier(
        cap, device=args.device, max_frames=args.max_frames, det_by_frame=det_by_frame
    )

    print("Building pitch homography maps…")
    metric: MetricContext = build_metric_from_maps(
        kp_by_frame, detections_by_frame=det_by_frame, pitch_confidence=0.9
    )
    gap_filled = metric.speed_transforms_gap_filled(0.9)
    # Single source of truth for the minimap: ungated keypoint-radar H (labelled
    # correspondences keep it upright), with the gated radar H as a fallback.
    radar_h_by_frame = metric.keypoint_radar_transforms(0.9)

    gk_assignment = getattr(args, "gk_assignment", "goal_distance")
    needs_frame = args.tracker in ("botsort", "botsort_nocmc")
    print("Building clip locks (team-id + goalkeeper)…")
    team_lock, gk_lock, locked_goal_defenders = compute_clip_locks(
        args.source_video_path,
        team_classifier=team_classifier,
        tracker=create_player_tracker(fps, kind=args.tracker),
        needs_frame=needs_frame,
        gk_assignment=gk_assignment,
        metric=metric,
        max_frames=args.max_frames,
        detections_by_frame=det_by_frame,
    )

    tracker = create_player_tracker(fps, kind=args.tracker)
    vel_smoother = KalmanVelocitySmoother(alpha=0.3)
    speed_smoother = KalmanSpeedDisplaySmoother(alpha=0.3)
    joy_smoother = JoystickDotSmoother(alpha=0.32)

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
            players = raw_dets[raw_dets.class_id == PLAYER_CLASS_ID]
            gks = raw_dets[raw_dets.class_id == GOALKEEPER_CLASS_ID]

            trackable = sv.Detections.merge([players, gks]) if (len(players) or len(gks)) else sv.Detections.empty()
            tracked = (
                tracker.update(trackable, frame=frame if needs_frame else None)
                if len(trackable) else sv.Detections.empty()
            )

            # ── team classification ────────────────────────────────────────
            team_arr = np.full(len(tracked), TEAM_NONE, dtype=int)
            if len(tracked):
                t_players = tracked[tracked.class_id == PLAYER_CLASS_ID]
                if len(t_players):
                    pl_teams = team_classifier.predict(get_crops(frame, t_players))
                    team_arr[tracked.class_id == PLAYER_CLASS_ID] = pl_teams
                team_arr = apply_team_lock(team_arr, tracked.class_id, tracked.tracker_id, team_lock)
                if gk_assignment == "centroid":
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
            if gk_assignment == "goal_distance":
                tracked = apply_goalkeeper_frame(
                    tracked, metric.radar_transforms.get(frame_idx), gk_lock
                )

            # ── Kalman velocity ────────────────────────────────────────────
            dets = attach_kalman_velocity(tracked, tracker, needs_frame=needs_frame, image=frame)
            dets = vel_smoother.smooth_detections(dets)
            # Clip-level lock has the final say so class-flipping keepers stay one colour.
            dets = relock_detection_teams(dets, team_lock)

            # ── per-player Kalman ground speed (gap-filled H) ──────────────
            transformer = gap_filled.get(frame_idx)
            speed_by_tid: dict[int, float] = {}
            if dets.tracker_id is not None and transformer is not None:
                fxy = feet_xy(dets)
                for i, tid in enumerate(dets.tracker_id):
                    tid = int(tid)
                    if tid < 0 or dets.data is None:
                        continue
                    kf_vx = dets.data.get("kf_vx")
                    kf_vy = dets.data.get("kf_vy")
                    if kf_vx is None:
                        continue
                    vx, vy = float(kf_vx[i]), float(kf_vy[i])
                    if not (np.isfinite(vx) and np.isfinite(vy)):
                        continue
                    speed_ms = kalman_ground_speed_m_s(
                        fxy[i], np.array([vx, vy], dtype=np.float64),
                        transformer, fps=fps,
                    )
                    if speed_ms is not None:
                        speed_by_tid[tid] = speed_smoother.smooth(tid, float(speed_ms))

            # ── render ─────────────────────────────────────────────────────
            annotated = frame.copy()
            draw_team_ellipses(annotated, dets)
            draw_joystick_dots(
                annotated, dets, joy_smoother,
                speed_by_tid=speed_by_tid, show_speed=True,
            )
            draw_speed_legend(annotated)

            # radar minimap: ungated keypoint H (upright), gated radar H as fallback
            radar_t = radar_h_by_frame.get(frame_idx) or metric.radar_transforms.get(frame_idx)
            if radar_t is not None:
                draw_radar_minimap(
                    annotated, dets, radar_t,
                    locked_goal_defenders=locked_goal_defenders,
                )

            sink.write_frame(annotated)

    cap.release()
    print(f"Wrote {args.target_video_path}")
