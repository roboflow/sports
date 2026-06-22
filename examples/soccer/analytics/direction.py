"""analytics/direction.py — Feature 1: BoTSORT tracking + team colors + joystick direction dots.

No homography. Image-space Kalman velocity only.
"""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

from analytics.goalkeepers import compute_clip_locks
from analytics.support import (
    GOALKEEPER_CLASS_ID,
    PLAYER_CLASS_ID,
    REFEREE_CLASS_ID,
    TEAM_NONE,
    JoystickDotSmoother,
    KalmanVelocitySmoother,
    attach_kalman_velocity,
    create_player_detector,
    create_player_tracker,
    draw_joystick_dots,
    draw_team_ellipses,
    fit_team_classifier,
    get_crops,
    open_video,
    resolve_goalkeepers_team_id,
)
from analytics.teams import apply_team_lock


def run_direction(args) -> None:
    """Render team-colored ellipses + image-space Kalman direction dots."""
    cap, fps, width, height = open_video(args.source_video_path)

    player_detector_fn = create_player_detector(
        backend=args.player_detector,
        model_path=getattr(args, "player_model_path", None),
        model_id=getattr(args, "player_model_id", "football-players-detection-3zvbc/11"),
        device=args.device,
        api_key=getattr(args, "api_key", None),
    )

    print("Fitting team classifier…")
    team_classifier = fit_team_classifier(
        cap,
        player_detector_fn,
        device=args.device,
        max_frames=args.max_frames,
    )

    needs_frame = args.tracker in ("botsort", "botsort_nocmc")

    # Clip-level team-id majority lock (DIRECTION has no homography, so no GK goal lock).
    print("Building team-id stabilization lock…")
    team_lock, _, _ = compute_clip_locks(
        args.source_video_path,
        player_detector_fn=player_detector_fn,
        team_classifier=team_classifier,
        tracker=create_player_tracker(fps, kind=args.tracker),
        needs_frame=needs_frame,
        gk_assignment="centroid",
        metric=None,
        max_frames=args.max_frames,
    )

    tracker = create_player_tracker(fps, kind=args.tracker)
    vel_smoother = KalmanVelocitySmoother(alpha=0.3)
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

            raw_dets = player_detector_fn(frame)

            # ── track ──────────────────────────────────────────────────────
            players = raw_dets[raw_dets.class_id == PLAYER_CLASS_ID]
            gks = raw_dets[raw_dets.class_id == GOALKEEPER_CLASS_ID]
            refs = raw_dets[raw_dets.class_id == REFEREE_CLASS_ID]

            trackable = sv.Detections.merge([players, gks]) if (len(players) or len(gks)) else sv.Detections.empty()
            tracked = (
                tracker.update(trackable, frame=frame if needs_frame else None)
                if len(trackable) else sv.Detections.empty()
            )

            # ── team classification ────────────────────────────────────────
            # DIRECTION has no pitch homography by design, so the goal-distance GK
            # path (which needs pitch coords) is not available here: regardless of
            # --gk-assignment we always use the centroid rule for goalkeepers.
            team_arr = np.full(len(tracked), TEAM_NONE, dtype=int)
            if len(tracked):
                t_players = tracked[tracked.class_id == PLAYER_CLASS_ID]
                if len(t_players):
                    player_crops = get_crops(frame, t_players)
                    player_teams = team_classifier.predict(player_crops)
                    team_arr[tracked.class_id == PLAYER_CLASS_ID] = player_teams
                team_arr = apply_team_lock(team_arr, tracked.class_id, tracked.tracker_id, team_lock)
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

            # ── attach Kalman velocity + smooth ────────────────────────────
            dets_with_vel = attach_kalman_velocity(
                tracked_with_teams,
                tracker,
                needs_frame=needs_frame,
                image=frame,
            )
            dets_with_vel = vel_smoother.smooth_detections(dets_with_vel)

            # ── render (team-colored dots; no track-id numbers in direction) ──
            annotated = frame.copy()
            draw_team_ellipses(annotated, dets_with_vel, show_ids=False)
            draw_joystick_dots(annotated, dets_with_vel, joy_smoother)

            sink.write_frame(annotated)

    cap.release()
    print(f"Wrote {args.target_video_path}")
