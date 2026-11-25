"""analytics/distance.py — Feature 3: cumulative distance, focus-all look + end-card.

Consumes a shared :class:`~analytics.clip_pipeline.ClipAnalysis` (computed if absent): the
single BoTSORT pass, gated homographies and per-track cumulative-distance kinematics are
read from it rather than recomputed. The per-frame render reuses the shared SPEED_AND_DISTANCE
follow-all look (all players annotated with instant speed + cumulative-distance chips, plus
a translucent trace radar). DISTANCE stays distinct from SPEED_AND_DISTANCE follow-all by
appending its distance leaderboard end-card.

Speed source:    Kalman ground speed + speed_transforms_gap_filled (instant chips).
Distance source: compute_kinematics(mode="homography", gated speed_transforms).
"""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

from analytics.clip_pipeline import ClipAnalysis, compute_clip_analysis
from analytics.speed import _draw_speed_overlay, _speed_by_tid
from analytics.player_motion import (
    JoystickDotSmoother,
    KalmanSpeedDisplaySmoother,
    KalmanVelocitySmoother,
    build_trace_minimap,
    cumulative_distance_at_frame,
    draw_distance_labels,
    draw_speed_legend,
    feet_xy,
    open_video,
    overlay_minimap,
)

_GK_ASSIGNMENT = "goal_distance"


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


def run_distance(args, analysis: ClipAnalysis | None = None) -> None:
    """Render the follow-all look + distance leaderboard, replaying the shared analysis.

    Reuses the shared :class:`ClipAnalysis` (computed if not supplied): the single BoTSORT
    pass, gated homographies and per-track kinematics are read from it. Because the shared
    pass's tracked boxes / ids and captured single-update Kalman velocity match what the
    standalone two-pass DISTANCE produced, the output is byte-for-byte equivalent whether
    invoked standalone (``analysis=None``) or shared by the run-all orchestrator.
    """
    if analysis is None:
        analysis = compute_clip_analysis(args, need_homography=True)

    fps, width, height = analysis.fps, analysis.width, analysis.height
    locks = analysis.locks(_GK_ASSIGNMENT)
    locked_goal_defenders = locks.locked_goal_defenders
    gap_filled = analysis.gap_filled
    radar_h_by_frame = analysis.radar_h_by_frame

    # Per-track cumulative-distance kinematics (collected + integrated once, shared).
    raw_tracks = analysis.tracks
    tracked_lookup = analysis.tracked_by_frame()

    vel_smoother = KalmanVelocitySmoother(alpha=0.3)
    speed_smoother = KalmanSpeedDisplaySmoother(alpha=0.3)
    joy_smoother = JoystickDotSmoother(alpha=0.32)

    # per-tid growing trace in pitch-cm coordinates (radar polylines)
    trace_by_tid: dict[int, list[np.ndarray]] = {}

    cap, _, _, _ = open_video(args.source_video_path)
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

            tracked = tracked_lookup.get(frame_idx)
            if tracked is None:
                tracked = sv.Detections.empty()
            dets = analysis.decorate_replay_frame(
                frame_idx, tracked,
                gk_assignment=_GK_ASSIGNMENT, locks=locks, vel_smoother=vel_smoother,
            )

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

            speed_by_tid = _speed_by_tid(
                dets, gap_filled.get(frame_idx), fps, speed_smoother,
            )

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
            _draw_speed_overlay(
                annotated, dets, speed_by_tid, joy_smoother, show_legend=False,
            )
            draw_distance_labels(annotated, dets, dist_by_tid)
            draw_speed_legend(annotated)
            mini_radar = build_trace_minimap(
                dets, radar_h_by_frame.get(frame_idx), trace_by_tid, None,
                locked_goal_defenders=locked_goal_defenders,
            )
            overlay_minimap(annotated, mini_radar)
            sink.write_frame(annotated)

        # ── end-card (3 seconds) — keeps DISTANCE distinct from follow-all ──
        end_card = _build_end_card(width, height, raw_tracks)
        n_end_frames = max(1, int(fps * 3))
        for _ in range(n_end_frames):
            sink.write_frame(end_card)

    cap.release()
    print(f"Wrote {args.target_video_path}")
