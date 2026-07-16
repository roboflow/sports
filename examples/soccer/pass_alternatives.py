"""PASS_ALTERNATIVES mode: freeze moments with ranked open pass lanes."""

from __future__ import annotations

import supervision as sv

from sports.annotators.passing import (
    CARRIER_SHADOW_BGR,
    annotate_ball,
    annotate_pass_players,
    draw_carrier_ground_ellipse,
    draw_hud_bar,
    draw_pass_alternatives_overlay,
    draw_radar_minimap,
)
from sports.common.kinematics import feet_xy
from sports.common.pass_alternatives import (
    DEFAULT_FINAL_OPTION_EXTRA_SECONDS,
    DEFAULT_FREEZE_SECONDS,
    DEFAULT_OPTION_REVEAL_SECONDS,
    DEFAULT_SLOWDOWN_RAMP_SECONDS,
    slowdown_hold_count,
)
from sports.common.pass_pitch import lane_scoring_transformer_for_frame
from sports.common.possession import find_control_carrier
from sports.common.tracking import open_video
from sports.common.video_tracking import VideoTrackingSession, build_video_tracking_session


def run_pass_alternatives(args, session: VideoTrackingSession | None = None) -> None:
    """Render pass-alternative freeze frames to ``args.target_video_path``."""
    if session is None:
        session = build_video_tracking_session(args, need_homography=True)
    _render_pass_alternatives(args, session)


def _annotate_live(
    frame,
    dets,
    *,
    radar_h,
    locked_goals,
) -> object:
    image = frame.copy()
    image = annotate_pass_players(image, dets, show_tracker_ids=True)
    image = annotate_ball(image, dets)
    carrier = find_control_carrier(dets, transformer=radar_h)
    if carrier is not None:
        draw_carrier_ground_ellipse(
            image,
            feet_xy(dets)[carrier.index],
            transformer=radar_h,
            color_bgr=CARRIER_SHADOW_BGR,
            radius_m=0.5,
            alpha=0.42,
            filled=True,
            thickness=1,
        )
    image = draw_radar_minimap(
        image,
        dets,
        radar_h,
        locked_goal_defenders=locked_goals,
    )
    return draw_hud_bar(image, "PASS ALTERNATIVES")


def _render_pass_alternatives(args, session: VideoTrackingSession) -> None:
    locks = session.team_locks()
    locked_goals = locks.locked_goal_defenders
    pass_by_frame = session.pass_by_frame
    freeze_events = session.pass_alternative_events()
    events_by_frame = {e.frame_idx: e for e in freeze_events}
    event_frames = sorted(events_by_frame)
    fps = float(session.fps)
    width, height = session.width, session.height
    ramp_frames = max(1, int(round(DEFAULT_SLOWDOWN_RAMP_SECONDS * fps)))
    reveal_frames = max(4, int(round(DEFAULT_OPTION_REVEAL_SECONDS * fps)))
    gap_filled = (
        session.gap_filled_transforms_by_frame if session.kp_by_frame is not None else {}
    )
    minimap_transforms = session.minimap_transforms_by_frame
    pitch_confidence = 0.9

    print(f"PASS_ALTERNATIVES: {len(freeze_events)} freeze events")

    cap, _, _, _ = open_video(args.source_video_path)
    try:
        with sv.VideoSink(args.target_video_path, sv.VideoInfo(width, height, fps)) as sink:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if session.max_frames is not None and frame_idx > session.max_frames:
                    break
                dets = pass_by_frame.get(frame_idx)
                if dets is None:
                    continue

                kps = (session.kp_by_frame or {}).get(frame_idx)
                radar_h = minimap_transforms.get(frame_idx)
                lane_h = lane_scoring_transformer_for_frame(
                    gap_filled, frame_idx, kps, pitch_confidence=pitch_confidence
                ) or radar_h

                live = _annotate_live(
                    frame, dets, radar_h=radar_h, locked_goals=locked_goals,
                )
                frames_until = next(
                    (ef - frame_idx for ef in event_frames if ef >= frame_idx), None
                )
                hold = (
                    slowdown_hold_count(frames_until, ramp_frames=ramp_frames)
                    if frames_until is not None
                    else 1
                )
                for _ in range(hold):
                    sink.write_frame(live)

                if frame_idx not in events_by_frame:
                    continue

                event = events_by_frame[frame_idx]
                n_options = min(3, len(event.options))
                phases: list[tuple[int, int]] = [(0, reveal_frames)]
                phases.extend((i, reveal_frames) for i in range(1, n_options + 1))
                min_freeze = sum(h for _, h in phases)
                extra_hold = max(
                    0, int(round(DEFAULT_FREEZE_SECONDS * fps)) - min_freeze
                )
                final_extra = max(
                    4, int(round(DEFAULT_FINAL_OPTION_EXTRA_SECONDS * fps))
                )
                if phases:
                    phases[-1] = (
                        phases[-1][0],
                        phases[-1][1] + extra_hold + final_extra,
                    )
                for revealed, phase_hold in phases:
                    for step in range(phase_hold):
                        progress = (step + 1) / max(phase_hold, 1)
                        overlay = draw_pass_alternatives_overlay(
                            frame,
                            dets,
                            event,
                            revealed_options=revealed,
                            reveal_progress=progress,
                            transformer=lane_h,
                            locked_goal_defenders=locked_goals,
                            metric=True,
                        )
                        sink.write_frame(overlay)
    finally:
        cap.release()
    print(f"Wrote {args.target_video_path}")
