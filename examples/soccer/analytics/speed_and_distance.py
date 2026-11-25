"""analytics/speed_and_distance.py — Speed + distance overlay with radar traces.

Default (no --track-id): annotate all players with instant speed + cumulative
distance chips and per-tracker-id colored radar traces (the shared "follow-all" look).
With --track-id N: spotlight that player, dim the background, show the same on-player
speed + distance chips for the spotlighted player, and a radar minimap with that player's
trace (no lateral HUD panel).

Speed source:    Kalman ground speed + speed_transforms_gap_filled.
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


def _dim_frame(frame: np.ndarray, level: float = 0.22) -> np.ndarray:
    return np.clip(frame.astype(np.float32) * level, 0, 255).astype(np.uint8)


# Single-player spotlight geometry. The Gaussian falloff sigma scales with the
# radius, so the lit region keeps a small fully-bright core and a soft gradient
# into the dimmed background instead of a hard bright disc — the spotlighted player
# reads as a tight highlight rather than an oversized circle.
SPOTLIGHT_RADIUS = 210
SPOTLIGHT_STRENGTH = 0.88


def _spotlight(
    frame: np.ndarray,
    cx: int,
    cy: int,
    radius: int = SPOTLIGHT_RADIUS,
    strength: float = SPOTLIGHT_STRENGTH,
) -> np.ndarray:
    """Spotlight: dim the frame, then restore the spotlighted player within a soft circle."""
    dimmed = _dim_frame(frame)
    mask = np.zeros(frame.shape[:2], dtype=np.float32)
    cv2.circle(mask, (cx, cy), radius, 1.0, -1, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=radius * 0.38)
    mask = (mask[..., np.newaxis] * strength).astype(np.float32)
    out = dimmed.astype(np.float32) * (1.0 - mask) + frame.astype(np.float32) * mask
    return np.clip(out, 0, 255).astype(np.uint8)


def run_speed_and_distance(args, analysis: ClipAnalysis | None = None) -> None:
    """Render speed + distance overlay: spotlight (with --track-id) or full radar traces (default).

    Reuses the shared :class:`ClipAnalysis` (computed if not supplied): the single BoTSORT
    pass, gated homographies and per-track kinematics are read from it. As with DISTANCE the
    shared pass's tracked boxes / ids and captured single-update Kalman velocity match the
    standalone two-pass output, so the result is byte-for-byte equivalent whether invoked
    standalone (``analysis=None``) or shared by the run-all orchestrator.
    """
    raw_track_id = getattr(args, "track_id", None)
    spotlight_tid: int | None = int(raw_track_id) if raw_track_id is not None else None

    if analysis is None:
        analysis = compute_clip_analysis(args, need_homography=True)

    fps, width, height = analysis.fps, analysis.width, analysis.height
    locks = analysis.locks(_GK_ASSIGNMENT)
    locked_goal_defenders = locks.locked_goal_defenders
    gap_filled = analysis.gap_filled
    radar_h_by_frame = analysis.radar_h_by_frame

    # Distance is computed for every tracked player (not just the spotlight id) so the
    # follow-all view can show every player's accumulated distance, and the spotlight
    # view can read the spotlighted player's distance from the same kinematics.
    all_tracks = analysis.tracks
    tracked_lookup = analysis.tracked_by_frame()

    vel_smoother = KalmanVelocitySmoother(alpha=0.3)
    speed_smoother = KalmanSpeedDisplaySmoother(alpha=0.3)
    joy_smoother = JoystickDotSmoother(alpha=0.32)

    # per-tid growing trace in pitch-cm coordinates
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
                    if spotlight_tid is not None and tid != spotlight_tid:
                        continue
                    trace_by_tid.setdefault(tid, []).append(xy_cm[i].copy())

            # ── Kalman ground speed per player (gap-filled H) ──────────────
            # Follow-all shows speed on every player; spotlight mode filters via only_tid.
            speed_by_tid = _speed_by_tid(
                dets, gap_filled.get(frame_idx), fps, speed_smoother,
                only_tid=spotlight_tid,
            )

            # ── cumulative distance per player at this frame ───────────────
            dist_by_tid: dict[int, float] = {}
            if dets.tracker_id is not None:
                for tid in dets.tracker_id:
                    tid = int(tid)
                    track = all_tracks.get(tid)
                    if track is None:
                        continue
                    d = cumulative_distance_at_frame(track, frame_idx)
                    if d is not None:
                        dist_by_tid[tid] = d

            # ── render ─────────────────────────────────────────────────────
            radar_transformer = radar_h_by_frame.get(frame_idx)
            if spotlight_tid is not None:
                # spotlight mode: dim the scene, restore the spotlighted player area, and show
                # that player's speed + distance chips (no lateral HUD panel). The radar shows
                # only the spotlighted player's trace + dot via spotlight_tid.
                spotlight_mask = dets.tracker_id == spotlight_tid if dets.tracker_id is not None else np.zeros(len(dets), dtype=bool)
                visible = bool(spotlight_mask.any())
                if visible:
                    box = dets.xyxy[spotlight_mask][0]
                    cx_f = int((box[0] + box[2]) / 2)
                    cy_f = int(box[3])  # feet (ground contact) anchor
                    annotated = _spotlight(frame, cx_f, cy_f)
                else:
                    annotated = _dim_frame(frame)
                marked = dets[spotlight_mask]
                _draw_speed_overlay(
                    annotated, marked, speed_by_tid, joy_smoother, show_legend=False,
                )
                draw_distance_labels(annotated, marked, dist_by_tid)
                mini_radar = build_trace_minimap(
                    dets, radar_transformer, trace_by_tid, spotlight_tid,
                    locked_goal_defenders=locked_goal_defenders,
                )
                overlay_minimap(annotated, mini_radar)
            else:
                # follow-all: speed + distance chips on every player + trace radar.
                annotated = frame.copy()
                _draw_speed_overlay(
                    annotated, dets, speed_by_tid, joy_smoother, show_legend=False,
                )
                draw_distance_labels(annotated, dets, dist_by_tid)
                draw_speed_legend(annotated)
                mini_radar = build_trace_minimap(
                    dets, radar_transformer, trace_by_tid, None,
                    locked_goal_defenders=locked_goal_defenders,
                )
                overlay_minimap(annotated, mini_radar)

            sink.write_frame(annotated)

    cap.release()
    print(f"Wrote {args.target_video_path}")
