"""PASS_NETWORK mode: completed passes, collaboration web, and summary end-card."""

from __future__ import annotations

from pathlib import Path

import cv2
import supervision as sv

from sports.annotators.passing import (
    CARRIER_SHADOW_BGR,
    annotate_ball,
    annotate_pass_players,
    draw_carrier_ground_ellipse,
    draw_collaboration_web,
    draw_hud_bar,
    draw_pass_network_end_card,
    draw_pass_network_frame_overlays,
    draw_radar_minimap,
)
from sports.common.kinematics import feet_xy
from sports.common.pass_network import build_pass_network
from sports.common.possession import find_control_carrier
from sports.common.tracking import open_video
from sports.common.video_tracking import VideoTrackingSession, build_video_tracking_session


def run_pass_network(args, session: VideoTrackingSession | None = None) -> None:
    """Render pass detection + collaboration overlay to ``args.target_video_path``."""
    if session is None:
        session = build_video_tracking_session(args, need_homography=True)
    _render_pass_network(args, session)


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
    # Short end-card (~2s) with top collaborators (accepted plan recommendation).
    end_hold_frames = max(int(fps * 2), 1)
    minimap_transforms = session.minimap_transforms_by_frame
    gap_filled = (
        session.gap_filled_transforms_by_frame if session.kp_by_frame is not None else {}
    )

    print(
        f"PASS_NETWORK: {network.n_passes} passes, {network.n_turnovers} turnovers, "
        f"{len(network.links)} collaboration links"
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
