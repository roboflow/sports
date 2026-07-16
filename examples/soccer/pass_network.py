"""PASS_NETWORK mode: completed passes, collaboration web, and summary end-card.

Optional ``--show-predictions`` freezes on detected pass releases and reveals
ranked open lanes (detect + suggest combined demo).
"""

from __future__ import annotations

from pathlib import Path

import supervision as sv

from sports.annotators.passing import (
    CARRIER_SHADOW_BGR,
    annotate_ball,
    annotate_pass_players,
    draw_carrier_ground_ellipse,
    draw_collaboration_web,
    draw_hud_bar,
    draw_pass_alternatives_overlay,
    draw_pass_network_end_card,
    draw_pass_network_frame_overlays,
    draw_radar_minimap,
)
from sports.common.kinematics import feet_xy
from sports.common.pass_alternatives import PassEvent
from sports.common.pass_network import build_pass_network
from sports.common.possession import carrier_from_tracker_id, find_control_carrier
from sports.common.tracking import open_video
from sports.common.video_tracking import VideoTrackingSession, build_video_tracking_session


def run_pass_network(args, session: VideoTrackingSession | None = None) -> None:
    """Render pass detection + collaboration overlay to ``args.target_video_path``."""
    if session is None:
        session = build_video_tracking_session(args, need_homography=True)
    _render_pass_network(args, session)


def _prediction_freeze_phases(fps: float, n_options: int) -> list[tuple[int, int]]:
    reveal_frames = max(4, int(round(0.6 * fps)))
    phases: list[tuple[int, int]] = [(0, reveal_frames)]
    phases.extend((i, reveal_frames) for i in range(1, n_options + 1))
    min_freeze = sum(h for _, h in phases)
    extra_hold = max(0, int(round(2.5 * fps)) - min_freeze)
    final_extra = max(4, int(round(1.0 * fps)))
    if phases:
        phases[-1] = (phases[-1][0], phases[-1][1] + extra_hold + final_extra)
    return phases


def _render_pass_network(args, session: VideoTrackingSession) -> None:
    locks = session.team_locks()
    locked_goals = locks.locked_goal_defenders
    pass_by_frame = session.pass_by_frame
    scan = session.pass_scan()
    network = build_pass_network(
        Path(session.source_video_path).stem,
        list(scan.passes),
        list(scan.turnovers),
        metric=True,
    )
    fps = float(session.fps)
    width, height = session.width, session.height
    end_hold_frames = max(int(fps * 2), 1)
    minimap_transforms = session.minimap_transforms_by_frame
    gap_filled = (
        session.gap_filled_transforms_by_frame if session.kp_by_frame is not None else {}
    )
    events_by_frame = {e.frame_idx: e for e in network.passes}
    show_predictions = bool(getattr(args, "show_predictions", False))
    freeze_quality_threshold = float(getattr(args, "freeze_quality_threshold", 0.0))
    scorer = session.pass_scorer if show_predictions else None

    print(
        f"PASS_NETWORK: {network.n_passes} passes, {network.n_turnovers} turnovers, "
        f"{len(network.links)} collaboration links"
        + (" (+ prediction freezes)" if show_predictions else "")
    )

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

                image = frame.copy()
                radar_h = minimap_transforms.get(frame_idx)
                lane_h = gap_filled.get(frame_idx) or radar_h

                image = annotate_pass_players(image, dets, show_tracker_ids=True)
                image = annotate_ball(image, dets)

                carrier = find_control_carrier(dets, transformer=radar_h)
                if carrier is not None:
                    draw_carrier_ground_ellipse(
                        image,
                        feet_xy(dets)[carrier.index],
                        transformer=lane_h,
                        color_bgr=CARRIER_SHADOW_BGR,
                        radius_m=0.5,
                        alpha=0.42,
                        filled=True,
                        thickness=1,
                    )

                draw_collaboration_web(image, dets, frame_idx, network.passes)
                draw_pass_network_frame_overlays(
                    image,
                    dets,
                    frame_idx,
                    network.passes,
                    network.turnovers,
                    fps,
                    transformer=lane_h,
                )

                if (
                    show_predictions
                    and scorer is not None
                    and frame_idx in events_by_frame
                ):
                    event = events_by_frame[frame_idx]
                    qs = event.quality_score
                    if qs is not None and qs >= freeze_quality_threshold:
                        freeze_carrier = carrier_from_tracker_id(dets, event.passer_tid)
                        if freeze_carrier is not None:
                            options = scorer.top_options(
                                frame_idx, dets, freeze_carrier, k=3
                            )
                            if options:
                                freeze_event = PassEvent(
                                    frame_idx=frame_idx,
                                    carrier=freeze_carrier,
                                    options=options,
                                    top_score=options[0].score,
                                )
                                for revealed, phase_hold in _prediction_freeze_phases(
                                    fps, min(3, len(options))
                                ):
                                    for step in range(phase_hold):
                                        progress = (step + 1) / max(phase_hold, 1)
                                        overlay = draw_pass_alternatives_overlay(
                                            frame,
                                            dets,
                                            freeze_event,
                                            revealed_options=revealed,
                                            reveal_progress=progress,
                                            transformer=lane_h,
                                            locked_goal_defenders=locked_goals,
                                            metric=True,
                                            hud_title=(
                                                "PASS NETWORK  -  detected pass"
                                                " + open lanes"
                                            ),
                                        )
                                        sink.write_frame(overlay)

                image = draw_radar_minimap(
                    image,
                    dets,
                    radar_h,
                    locked_goal_defenders=locked_goals,
                )
                image = draw_hud_bar(
                    image,
                    f"PASS NETWORK  passes={network.n_passes}  turnovers={network.n_turnovers}",
                )
                sink.write_frame(image)

            end_card = draw_pass_network_end_card(
                (width, height), network, top_n=5,
            )
            for _ in range(end_hold_frames):
                sink.write_frame(end_card)
    finally:
        cap.release()
    print(f"Wrote {args.target_video_path}")
