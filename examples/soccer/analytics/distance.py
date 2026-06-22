"""analytics/distance.py — Feature 3: cumulative distance + leaderboard end-card.

Two-pass: collect_tracks → compute_kinematics(mode="homography", gated H) →
per-frame cumulative-meters labels + end-card distance ranking.
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
    TEAM_NONE,
    KalmanVelocitySmoother,
    attach_kalman_velocity,
    collect_tracks,
    compute_kinematics,
    create_player_detector,
    create_pitch_keypoint_detector,
    create_player_tracker,
    cumulative_distance_at_frame,
    draw_distance_labels,
    draw_team_ellipses,
    fit_team_classifier,
    get_crops,
    open_video,
    resolve_goalkeepers_team_id,
)
from analytics.teams import apply_team_lock


def _build_end_card(
    width: int,
    height: int,
    tracks: dict,
    *,
    n_top: int = 10,
) -> np.ndarray:
    """Black end-card with distance leaderboard (distance ranking only)."""
    card = np.zeros((height, width, 3), dtype=np.uint8)
    title = "DISTANCE LEADERBOARD"
    cv2.putText(card, title, (40, 60), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(card, (40, 80), (width - 40, 80), (120, 120, 120), 1)

    ranked = sorted(
        ((tid, t.distance_m) for tid, t in tracks.items()),
        key=lambda x: x[1],
        reverse=True,
    )[:n_top]

    y = 130
    for rank, (tid, dist_m) in enumerate(ranked, 1):
        label = f"#{rank:2d}   Track {tid:4d}   {dist_m:7.1f} m"
        color = (255, 215, 0) if rank == 1 else (200, 200, 200)
        cv2.putText(card, label, (60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 1, cv2.LINE_AA)
        y += 46

    cv2.putText(
        card,
        "Distance measured via gated pitch homography",
        (40, height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (90, 90, 90),
        1,
        cv2.LINE_AA,
    )
    return card


def run_distance(args) -> None:
    """Two-pass: collect detections → kinematics → render cumulative distance + end-card."""
    cap, fps, width, height = open_video(args.source_video_path)

    player_model_id = getattr(args, "player_model_id", "football-players-detection-3zvbc/11")
    pitch_model_id = getattr(args, "pitch_model_id", "football-field-detection-f07vi/15")
    player_detector_fn = create_player_detector(
        backend=args.player_detector,
        model_path=getattr(args, "player_model_path", None),
        model_id=player_model_id,
        device=args.device,
        api_key=getattr(args, "api_key", None),
    )
    pitch_detector_fn = create_pitch_keypoint_detector(
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
        args.source_video_path, player_detector_fn, cache, max_frames=args.max_frames
    )
    kp_by_frame = build_or_load_keypoints(
        args.source_video_path, pitch_detector_fn, cache, max_frames=args.max_frames
    )

    print("Fitting team classifier…")
    team_classifier = fit_team_classifier(
        cap, device=args.device, max_frames=args.max_frames, det_by_frame=det_by_frame
    )

    print("Building pitch homography maps…")
    metric: MetricContext = build_metric_from_maps(
        kp_by_frame, detections_by_frame=det_by_frame, pitch_confidence=0.9
    )
    speed_transforms = metric.speed_transforms

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

    # ── First pass: collect detections for kinematics ──────────────────────
    print("First pass: collecting tracks…")
    tracker_pass1 = create_player_tracker(fps, kind=args.tracker)

    def _iter_detections():
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if args.max_frames is not None and frame_idx > args.max_frames:
                break
            raw = det_by_frame.get(frame_idx)
            if raw is None:
                raw = sv.Detections.empty()
            players = raw[raw.class_id == PLAYER_CLASS_ID]
            gks = raw[raw.class_id == GOALKEEPER_CLASS_ID]
            trackable = sv.Detections.merge([players, gks]) if (len(players) or len(gks)) else sv.Detections.empty()
            tracked = (
                tracker_pass1.update(trackable, frame=frame if needs_frame else None)
                if len(trackable) else sv.Detections.empty()
            )
            yield frame_idx, tracked

    raw_tracks = collect_tracks(_iter_detections())
    compute_kinematics(
        raw_tracks,
        fps,
        mode="homography",
        frame_transforms=speed_transforms,
    )

    # ── Second pass: render ────────────────────────────────────────────────
    print("Second pass: rendering…")
    tracker_pass2 = create_player_tracker(fps, kind=args.tracker)
    vel_smoother = KalmanVelocitySmoother(alpha=0.3)

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

            raw = det_by_frame.get(frame_idx)
            if raw is None:
                raw = sv.Detections.empty()
            players = raw[raw.class_id == PLAYER_CLASS_ID]
            gks = raw[raw.class_id == GOALKEEPER_CLASS_ID]
            trackable = sv.Detections.merge([players, gks]) if (len(players) or len(gks)) else sv.Detections.empty()
            tracked = (
                tracker_pass2.update(trackable, frame=frame if needs_frame else None)
                if len(trackable) else sv.Detections.empty()
            )

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

            if gk_assignment == "goal_distance":
                tracked = apply_goalkeeper_frame(
                    tracked, metric.radar_transforms.get(frame_idx), gk_lock
                )
            dets = attach_kalman_velocity(tracked, tracker_pass2, needs_frame=needs_frame, image=frame)
            dets = vel_smoother.smooth_detections(dets)

            # cumulative distance per tid at this frame
            dist_by_tid: dict[int, float] = {}
            if dets.tracker_id is not None:
                for tid in dets.tracker_id:
                    tid = int(tid)
                    t = raw_tracks.get(tid)
                    if t is None:
                        continue
                    d = cumulative_distance_at_frame(t, frame_idx)
                    if d is not None:
                        dist_by_tid[tid] = d

            annotated = frame.copy()
            draw_team_ellipses(annotated, dets)
            draw_distance_labels(annotated, dets, dist_by_tid)
            sink.write_frame(annotated)

        # ── end-card (3 seconds) ───────────────────────────────────────────
        end_card = _build_end_card(width, height, raw_tracks)
        n_end_frames = max(1, int(fps * 3))
        for _ in range(n_end_frames):
            sink.write_frame(end_card)

    cap.release()
    print(f"Wrote {args.target_video_path}")
