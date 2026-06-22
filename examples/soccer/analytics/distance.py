"""analytics/distance.py — Feature 3: cumulative distance, focus-all look + end-card.

Two-pass: collect_tracks → compute_kinematics(mode="homography", gated H). The per-frame
render reuses the shared PLAYER_FOCUS follow-all look (all players annotated with instant
speed + cumulative-distance chips, plus a translucent trace radar). DISTANCE stays distinct
from PLAYER_FOCUS follow-all by appending its distance leaderboard end-card.

Speed source:    Kalman ground speed + speed_transforms_gap_filled (instant chips).
Distance source: compute_kinematics(mode="homography", gated speed_transforms).
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
from analytics.homography import (
    MetricContext,
    build_metric_from_maps,
    build_radar_homography_map,
    valid_pitch_cm,
)
from analytics.support import (
    GOALKEEPER_CLASS_ID,
    PLAYER_CLASS_ID,
    TEAM_NONE,
    KalmanSpeedDisplaySmoother,
    KalmanVelocitySmoother,
    JoystickDotSmoother,
    collect_tracks,
    compute_kinematics,
    create_player_detector,
    create_pitch_keypoint_detector,
    create_player_tracker,
    cumulative_distance_at_frame,
    feet_xy,
    fit_team_classifier,
    get_crops,
    kalman_ground_speed_m_s,
    kalman_velocity_arrays,
    open_video,
    render_follow_all_frame,
    resolve_goalkeepers_team_id,
)
from analytics.teams import apply_team_lock, relock_detection_teams


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
    """Two-pass: collect detections → kinematics → render follow-all look + end-card."""
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
    speed_transforms = metric.speed_transforms
    gap_filled = metric.speed_transforms_gap_filled(0.9)
    # Single source of truth for the minimap (traces AND live dots): the gated,
    # orientation-locked radar H, with an orientation-locked keypoint H fallback.
    radar_h_by_frame = build_radar_homography_map(metric, confidence=0.9)

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
    speed_smoother = KalmanSpeedDisplaySmoother(alpha=0.3)
    joy_smoother = JoystickDotSmoother(alpha=0.32)

    # per-tid growing trace in pitch-cm coordinates (radar polylines)
    trace_by_tid: dict[int, list[np.ndarray]] = {}

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
            # Read Kalman velocity from the tracker already advanced above; a second
            # tracker.update() here would inflate/fragment ids and break cross-pass
            # distance attribution.
            kf_vx, kf_vy = kalman_velocity_arrays(tracked, tracker_pass2)
            data = dict(tracked.data) if tracked.data else {}
            data["kf_vx"] = kf_vx
            data["kf_vy"] = kf_vy
            dets = sv.Detections(
                xyxy=tracked.xyxy,
                class_id=tracked.class_id,
                tracker_id=tracked.tracker_id,
                confidence=tracked.confidence,
                data=data,
            )
            dets = vel_smoother.smooth_detections(dets)
            # Clip-level lock has the final say so class-flipping keepers stay one colour.
            dets = relock_detection_teams(dets, team_lock)

            # ── update trace buffers (radar H → pitch cm) ──────────────────
            radar_h = radar_h_by_frame.get(frame_idx)
            if radar_h is not None and dets.tracker_id is not None:
                fxy = feet_xy(dets)
                xy_cm = radar_h.transform_points(fxy.astype(np.float32))
                for i, tid in enumerate(dets.tracker_id):
                    tid = int(tid)
                    if tid < 0:
                        continue
                    if not valid_pitch_cm(xy_cm[i:i + 1], margin_cm=80.0)[0]:
                        continue
                    trace_by_tid.setdefault(tid, []).append(xy_cm[i].copy())

            # ── Kalman ground speed per player (gap-filled H) ──────────────
            speed_t = gap_filled.get(frame_idx)
            speed_by_tid: dict[int, float] = {}
            if speed_t is not None and dets.tracker_id is not None and dets.data is not None:
                fxy = feet_xy(dets)
                kf_vx = dets.data.get("kf_vx")
                kf_vy = dets.data.get("kf_vy")
                if kf_vx is not None:
                    for i, tid in enumerate(dets.tracker_id):
                        tid = int(tid)
                        if tid < 0:
                            continue
                        vx, vy = float(kf_vx[i]), float(kf_vy[i])
                        if not (np.isfinite(vx) and np.isfinite(vy)):
                            continue
                        s = kalman_ground_speed_m_s(
                            fxy[i], np.array([vx, vy], dtype=np.float64),
                            speed_t, fps=fps,
                        )
                        if s is not None:
                            speed_by_tid[tid] = speed_smoother.smooth(tid, float(s))

            # ── cumulative distance per tid at this frame ──────────────────
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

            # ── render: shared follow-all look (chips + trace radar) ───────
            annotated = frame.copy()
            render_follow_all_frame(
                annotated, dets,
                joystick_smoother=joy_smoother,
                speed_by_tid=speed_by_tid,
                distance_by_tid=dist_by_tid,
                trace_by_tid=trace_by_tid,
                radar_transformer=radar_h_by_frame.get(frame_idx),
                locked_goal_defenders=locked_goal_defenders,
                show_legend=True,
            )
            sink.write_frame(annotated)

        # ── end-card (3 seconds) — keeps DISTANCE distinct from follow-all ──
        end_card = _build_end_card(width, height, raw_tracks)
        n_end_frames = max(1, int(fps * 3))
        for _ in range(n_end_frames):
            sink.write_frame(end_card)

    cap.release()
    print(f"Wrote {args.target_video_path}")
