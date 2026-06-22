"""analytics/player_focus.py — Feature 4: per-player focus with radar traces.

Default (no --track-id): annotate all players, per-tracker-id colored radar traces.
With --track-id N: spotlight that player, dim background, HUD with live speed +
cumulative distance, radar minimap with that player's trace.

Speed source:    Kalman ground speed + speed_transforms_gap_filled.
Distance source: compute_kinematics(mode="homography", gated speed_transforms).
"""

from __future__ import annotations

import colorsys

import cv2
import numpy as np
import supervision as sv

from analytics.cache import (
    FrameCache,
    build_or_load_detections,
    build_or_load_keypoints,
)
from analytics.goalkeepers import apply_goalkeeper_frame, compute_clip_locks
from analytics.homography import MetricContext, build_metric_from_maps, valid_pitch_cm
from analytics.support import (
    GOALKEEPER_CLASS_ID,
    PLAYER_CLASS_ID,
    TEAM_NONE,
    KalmanSpeedDisplaySmoother,
    KalmanVelocitySmoother,
    JoystickDotSmoother,
    MS_TO_KMH,
    attach_kalman_velocity,
    kalman_velocity_arrays,
    collect_tracks,
    compute_kinematics,
    create_player_detector,
    create_pitch_keypoint_detector,
    create_player_tracker,
    cumulative_distance_at_frame,
    draw_goals_on_pitch,
    draw_joystick_dots,
    draw_speed_legend,
    draw_team_ellipses,
    draw_trace_on_minimap,
    feet_xy,
    fit_team_classifier,
    get_crops,
    kalman_ground_speed_m_s,
    open_video,
    player_mask,
    resolve_goalkeepers_team_id,
    track_id_color,
    TEAM_COLORS,
)
from analytics.teams import apply_team_lock
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.configs.soccer import SoccerPitchConfiguration


_PITCH_CONFIG = SoccerPitchConfiguration()
_MINIMAP_SCALE = 0.065
_MINIMAP_PAD = 30


def _dim_frame(frame: np.ndarray, level: float = 0.22) -> np.ndarray:
    return np.clip(frame.astype(np.float32) * level, 0, 255).astype(np.uint8)


def _spotlight(frame: np.ndarray, cx: int, cy: int, radius: int = 200) -> np.ndarray:
    """Spotlight: dim the frame everywhere except within radius of (cx, cy)."""
    mask = np.zeros(frame.shape[:2], dtype=np.float32)
    cv2.circle(mask, (cx, cy), radius, 1.0, -1, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (0, 0), radius // 4)
    mask = np.clip(mask, 0, 1)[..., np.newaxis]
    dim = _dim_frame(frame)
    return np.where(mask > 0.05, (mask * frame.astype(np.float32) + (1 - mask) * dim.astype(np.float32)).astype(np.uint8), dim)


def _draw_hud(
    frame: np.ndarray,
    *,
    track_id: int,
    speed_ms: float | None,
    distance_m: float | None,
    visible: bool,
) -> None:
    h = frame.shape[0]
    x, y = 18, h - 150
    # title — name the subject without exposing the raw tracker id
    cv2.putText(frame, "FOCUS PLAYER", (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.8, (180, 80, 255), 2, cv2.LINE_AA)
    status_color = (80, 255, 80) if visible else (80, 80, 200)
    cv2.putText(frame, "on pitch" if visible else "occluded", (x, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1, cv2.LINE_AA)
    speed_str = f"speed:  {speed_ms:.1f} m/s  ({speed_ms * MS_TO_KMH:.1f} km/h)" if speed_ms is not None else "speed:  —"
    dist_str = f"dist:   {distance_m:.1f} m" if distance_m is not None else "dist:   —"
    cv2.putText(frame, speed_str, (x, y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 80), 1, cv2.LINE_AA)
    cv2.putText(frame, dist_str, (x, y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 220, 80), 1, cv2.LINE_AA)


def _build_trace_minimap(
    detections: sv.Detections,
    transformer,
    trace_by_tid: dict[int, list[np.ndarray]],
    focus_tid: int | None,
    *,
    config: SoccerPitchConfiguration = _PITCH_CONFIG,
    scale: float = _MINIMAP_SCALE,
    padding: int = _MINIMAP_PAD,
    locked_goal_defenders: tuple[int, int] | None = None,
) -> np.ndarray:
    """Build radar minimap with per-track colored traces (and defended-goal shading)."""
    radar = draw_pitch(config=config, padding=padding, scale=scale)
    if locked_goal_defenders is not None:
        left_def, right_def = locked_goal_defenders
        if left_def in (0, 1) and right_def in (0, 1):
            radar = draw_goals_on_pitch(
                config, left_defender_team=left_def, right_defender_team=right_def,
                team_colors=TEAM_COLORS, padding=padding, scale=scale, pitch=radar,
            )
    # draw traces
    for tid, pts in trace_by_tid.items():
        if len(pts) < 2:
            continue
        trace = np.stack(pts, axis=0)
        if focus_tid is not None and tid != focus_tid:
            color = (60, 60, 60)
        else:
            color = track_id_color(tid)
        radar = draw_trace_on_minimap(radar, trace, color, padding=padding, scale=scale)

    # draw current player positions
    if transformer is not None and len(detections):
        pmask = player_mask(detections)
        if pmask.any():
            pdet = detections[pmask]
            xy = feet_xy(pdet).astype(np.float32)
            xy_cm = transformer.transform_points(xy)
            # Drop obviously off-pitch warps (a few metres beyond the lines are kept).
            on_pitch = valid_pitch_cm(xy_cm, config, margin_cm=-300.0)
            tids_p = pdet.tracker_id if pdet.tracker_id is not None else np.full(len(pdet), -1)
            teams = pdet.data.get("team", np.full(len(pdet), TEAM_NONE)) if pdet.data else np.full(len(pdet), TEAM_NONE)
            for i in range(len(pdet)):
                if not on_pitch[i]:
                    continue
                t_id = int(tids_p[i])
                team = int(teams[i])
                if focus_tid is not None and t_id != focus_tid:
                    color = (60, 60, 60)
                elif team in (0, 1):
                    color = TEAM_COLORS[team].as_bgr()
                else:
                    color = track_id_color(t_id) if t_id >= 0 else (150, 150, 150)
                pt_cm = xy_cm[i]
                px = int(pt_cm[0] * scale) + padding
                py = int(pt_cm[1] * scale) + padding
                cv2.circle(radar, (px, py), 8, color, -1, cv2.LINE_AA)
                cv2.circle(radar, (px, py), 8, (255, 255, 255), 1, cv2.LINE_AA)
    return radar


def run_player_focus(args) -> None:
    """Render player focus: spotlight (with --track-id) or full radar traces (default)."""
    cap, fps, width, height = open_video(args.source_video_path)
    focus_tid: int | None = getattr(args, "track_id", None)

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
    # Single source of truth for the minimap (traces AND live dots): ungated keypoint H.
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

    # ── First pass: collect tracks for distance kinematics ─────────────────
    print("First pass: collecting tracks…")
    tracker_p1 = create_player_tracker(fps, kind=args.tracker)

    def _iter_pass1():
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fi = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            fi += 1
            if args.max_frames is not None and fi > args.max_frames:
                break
            raw = det_by_frame.get(fi)
            if raw is None:
                raw = sv.Detections.empty()
            players = raw[raw.class_id == PLAYER_CLASS_ID]
            gks = raw[raw.class_id == GOALKEEPER_CLASS_ID]
            trackable = sv.Detections.merge([players, gks]) if (len(players) or len(gks)) else sv.Detections.empty()
            tracked = (
                tracker_p1.update(trackable, frame=frame if needs_frame else None)
                if len(trackable) else sv.Detections.empty()
            )
            yield fi, tracked

    all_tracks = collect_tracks(_iter_pass1())
    compute_kinematics(
        all_tracks,
        fps,
        mode="homography",
        frame_transforms=speed_transforms,
    )

    # ── Second pass: render ────────────────────────────────────────────────
    print("Second pass: rendering…")
    tracker_p2 = create_player_tracker(fps, kind=args.tracker)
    vel_smoother = KalmanVelocitySmoother(alpha=0.3)
    speed_smoother = KalmanSpeedDisplaySmoother(alpha=0.3)
    joy_smoother = JoystickDotSmoother(alpha=0.32)

    # per-tid growing trace in pitch-cm coordinates
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
                tracker_p2.update(trackable, frame=frame if needs_frame else None)
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
            # tracker.update() here would inflate/fragment tracker ids each frame and
            # break the --track-id spotlight (and cross-pass distance attribution).
            kf_vx, kf_vy = kalman_velocity_arrays(tracked, tracker_p2)
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

            # ── update trace buffers (radar H → pitch cm) ──────────────────
            radar_h = radar_h_by_frame.get(frame_idx)
            if radar_h is not None and dets.tracker_id is not None:
                fxy = feet_xy(dets)
                xy_cm = radar_h.transform_points(fxy.astype(np.float32))
                for i, tid in enumerate(dets.tracker_id):
                    tid = int(tid)
                    if tid < 0:
                        continue
                    trace_by_tid.setdefault(tid, []).append(xy_cm[i].copy())

            # ── Kalman speed for focus player (gap-filled H) ───────────────
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
                        if focus_tid is not None and tid != focus_tid:
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

            # ── render ─────────────────────────────────────────────────────
            if focus_tid is not None:
                # spotlight mode: dim the scene, then restore the focus player area and
                # annotate ONLY the focus player so the others stay dimmed and unmarked
                focus_mask = dets.tracker_id == focus_tid if dets.tracker_id is not None else np.zeros(len(dets), dtype=bool)
                visible = bool(focus_mask.any())
                if visible:
                    cx_f = int((dets.xyxy[focus_mask][0, 0] + dets.xyxy[focus_mask][0, 2]) / 2)
                    cy_f = int((dets.xyxy[focus_mask][0, 1] + dets.xyxy[focus_mask][0, 3]) / 2)
                    annotated = _spotlight(frame, cx_f, cy_f, radius=200)
                else:
                    annotated = _dim_frame(frame)
                marked = dets[focus_mask]
            else:
                annotated = frame.copy()
                marked = dets

            # Raw track-id numbers look noisy on the broadcast frame; never label them
            # in focus modes (the HUD names the focused player instead).
            draw_team_ellipses(annotated, marked, show_ids=False)
            draw_joystick_dots(
                annotated, marked, joy_smoother,
                speed_by_tid=speed_by_tid, show_speed=True,
            )
            if focus_tid is None:
                draw_speed_legend(annotated)

            # ── radar minimap with traces ──────────────────────────────────
            # Live dots use the SAME keypoint-radar H as the traces (one coordinate
            # frame), falling back to the gated radar H only when it is unavailable.
            radar_transformer = radar_h_by_frame.get(frame_idx) or metric.radar_transforms.get(frame_idx)
            mini_radar = _build_trace_minimap(
                dets,
                radar_transformer,
                trace_by_tid,
                focus_tid,
                locked_goal_defenders=locked_goal_defenders,
            )
            rh, rw = mini_radar.shape[:2]
            fh, fw = annotated.shape[:2]
            x0 = fw - rw - 12
            y0 = fh - rh - 12
            if x0 >= 0 and y0 >= 0:
                annotated[y0:y0 + rh, x0:x0 + rw] = mini_radar

            # ── HUD (focus mode only) ──────────────────────────────────────
            if focus_tid is not None:
                track = all_tracks.get(focus_tid)
                dist_m = cumulative_distance_at_frame(track, frame_idx) if track else None
                _draw_hud(
                    annotated,
                    track_id=focus_tid,
                    speed_ms=speed_by_tid.get(focus_tid),
                    distance_m=dist_m,
                    visible=(dets.tracker_id is not None and (dets.tracker_id == focus_tid).any()),
                )

            sink.write_frame(annotated)

    cap.release()
    print(f"Wrote {args.target_video_path}")
