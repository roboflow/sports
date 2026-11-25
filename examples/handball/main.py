import argparse
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(
    parent for parent in [CURRENT_DIR, *CURRENT_DIR.parents]
    if (parent / "sports").is_dir()
)
sys.path.insert(0, str(REPO_ROOT))

from sports.annotators.handball import (
    draw_court,
    draw_paths_on_court,
    draw_points_on_court,
)
from sports.common.view import ViewTransformer
from sports.configs.handball import HandballCourtConfiguration


PARENT_DIR = str(CURRENT_DIR)
DEFAULT_TARGET_DIR = os.path.join(PARENT_DIR, "data")

PLAYER_DETECTION_MODEL_PATH = os.path.join(
    PARENT_DIR, "data/handball-player-detection.pt"
)
BALL_DETECTION_MODEL_PATH = os.path.join(
    PARENT_DIR, "data/handball-ball-detection.pt"
)
COURT_DETECTION_MODEL_PATH = os.path.join(
    PARENT_DIR, "data/handball-court-keypoint-detection.pt"
)

CONFIG = HandballCourtConfiguration()
TEAM_COLORS = ["#E53935", "#1E88E5", "#FDD835"]

VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.WHITE,
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(TEAM_COLORS),
    thickness=2,
)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(TEAM_COLORS),
    text_color=sv.Color.WHITE,
    text_padding=5,
    text_thickness=1,
)
BALL_ANNOTATOR = sv.CircleAnnotator(
    color=sv.Color.from_hex(TEAM_COLORS[2]),
    thickness=2,
)


class Mode(Enum):
    """
    Enum class representing different modes for Handball AI examples.
    """

    COURT_RENDERING = "COURT_RENDERING"
    POINT_RENDERING = "POINT_RENDERING"
    PATH_RENDERING = "PATH_RENDERING"
    COURT_DETECTION = "COURT_DETECTION"
    PLAYER_DETECTION = "PLAYER_DETECTION"
    BALL_DETECTION = "BALL_DETECTION"
    RADAR = "RADAR"
    ALL_RENDERINGS = "ALL_RENDERINGS"


def ensure_target_dir(target_dir: str) -> None:
    os.makedirs(target_dir, exist_ok=True)


def require_video_paths(
    source_video_path: Optional[str],
    target_video_path: Optional[str],
    mode: Mode,
) -> Tuple[str, str]:
    if source_video_path is None or target_video_path is None:
        raise ValueError(
            f"{mode.value} requires --source_video_path and --target_video_path."
        )
    return source_video_path, target_video_path


def labels_from_result(result) -> List[str]:
    detections = sv.Detections.from_ultralytics(result)
    names = result.names
    return [names[class_id] for class_id in detections.class_id]


def detection_class_ids(result, class_names: Set[str]) -> List[int]:
    return [
        class_id
        for class_id, class_name in result.names.items()
        if class_name.lower() in class_names
    ]


def keypoint_correspondences(
    keypoints: sv.KeyPoints,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    court_vertices = np.array(CONFIG.vertices, dtype=np.float32)
    detected_xy = keypoints.xy[0].astype(np.float32)
    point_count = min(len(detected_xy), len(court_vertices))

    source = detected_xy[:point_count]
    target = court_vertices[:point_count]
    mask = (source[:, 0] > 1) & (source[:, 1] > 1)
    return source[mask], target[mask], mask


def write_video(
    source_video_path: str,
    target_video_path: str,
    frame_generator: Iterator[np.ndarray],
) -> None:
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in frame_generator:
            sink.write_frame(frame)


def save_image(target_dir: str, file_name: str, image: np.ndarray) -> str:
    ensure_target_dir(target_dir)
    target_path = os.path.join(target_dir, file_name)
    cv2.imwrite(target_path, image)
    return target_path


def render_court() -> np.ndarray:
    return draw_court(
        config=CONFIG,
        background_color=sv.Color(38, 132, 170),
        line_color=sv.Color.WHITE,
        goal_color=sv.Color.RED,
        padding=80,
        line_thickness=6,
        scale=0.2,
    )


def render_points() -> np.ndarray:
    court = render_court()
    players = {
        0: np.array([
            [650, 750],
            [1050, 1220],
            [1500, 530],
            [2300, 1450],
            [2850, 760],
            [3350, 1200],
        ]),
        1: np.array([
            [700, 1240],
            [1150, 560],
            [1700, 1460],
            [2350, 640],
            [2900, 1300],
            [3300, 840],
        ]),
        2: np.array([[2100, 980]]),
    }

    for team_id, xy in players.items():
        court = draw_points_on_court(
            config=CONFIG,
            xy=xy,
            face_color=sv.Color.from_hex(TEAM_COLORS[team_id]),
            edge_color=sv.Color.WHITE,
            radius=12,
            thickness=3,
            padding=80,
            scale=0.2,
            court=court,
        )

    return court


def render_paths() -> np.ndarray:
    court = render_points()
    paths = [
        np.array([[650, 750], [950, 840], [1280, 760], [1600, 900], [2100, 980]]),
        np.array([[3300, 840], [3020, 900], [2700, 1040], [2400, 980], [2100, 980]]),
        np.array([[1050, 1220], [1350, 1120], [1680, 1240], [1950, 1100]]),
    ]

    return draw_paths_on_court(
        config=CONFIG,
        paths=paths,
        color=sv.Color.WHITE,
        thickness=4,
        padding=80,
        scale=0.2,
        court=court,
    )


def run_court_detection(
    source_video_path: str,
    device: str,
    model_path: str,
) -> Iterator[np.ndarray]:
    court_detection_model = YOLO(model_path).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    for frame in frame_generator:
        result = court_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)
        label_count = min(len(CONFIG.labels), keypoints.xy.shape[1])

        annotated_frame = frame.copy()
        annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(
            annotated_frame,
            keypoints,
            CONFIG.labels[:label_count],
        )
        yield annotated_frame


def run_player_detection(
    source_video_path: str,
    device: str,
    model_path: str,
) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(model_path).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        labels = labels_from_result(result)

        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels=labels
        )
        yield annotated_frame


def run_ball_detection(
    source_video_path: str,
    device: str,
    model_path: str,
) -> Iterator[np.ndarray]:
    ball_detection_model = YOLO(model_path).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    for frame in frame_generator:
        result = ball_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        labels = labels_from_result(result)

        annotated_frame = frame.copy()
        annotated_frame = BALL_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels=labels
        )
        yield annotated_frame


def run_radar(
    source_video_path: str,
    device: str,
    player_model_path: str,
    court_model_path: str,
) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(player_model_path).to(device=device)
    court_detection_model = YOLO(court_model_path).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    for frame in frame_generator:
        court_result = court_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(court_result)
        source, target, _ = keypoint_correspondences(keypoints)

        player_result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(player_result)
        labels = labels_from_result(player_result)

        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels=labels
        )

        if len(source) < 4:
            yield annotated_frame
            continue

        player_class_ids = detection_class_ids(
            player_result, {"player", "goalkeeper"}
        )
        player_mask = np.isin(detections.class_id, player_class_ids)
        players = detections[player_mask]
        if len(players) == 0:
            yield annotated_frame
            continue

        transformer = ViewTransformer(source=source, target=target)
        xy = players.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        transformed_xy = transformer.transform_points(points=xy)

        radar = draw_court(config=CONFIG)
        radar = draw_points_on_court(
            config=CONFIG,
            xy=transformed_xy,
            face_color=sv.Color.from_hex(TEAM_COLORS[0]),
            edge_color=sv.Color.WHITE,
            radius=16,
            court=radar,
        )

        h, w, _ = frame.shape
        radar = sv.resize_image(radar, (w // 2, h // 2))
        radar_h, radar_w, _ = radar.shape
        rect = sv.Rect(
            x=w // 2 - radar_w // 2,
            y=h - radar_h,
            width=radar_w,
            height=radar_h,
        )
        yield sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)


def run_rendering_mode(target_dir: str, mode: Mode) -> Dict[str, str]:
    renderers = {
        Mode.COURT_RENDERING: ("handball-court.png", render_court),
        Mode.POINT_RENDERING: ("handball-points.png", render_points),
        Mode.PATH_RENDERING: ("handball-paths.png", render_paths),
    }

    if mode == Mode.ALL_RENDERINGS:
        selected_modes = [
            Mode.COURT_RENDERING,
            Mode.POINT_RENDERING,
            Mode.PATH_RENDERING,
        ]
    else:
        selected_modes = [mode]

    output_paths = {}
    for selected_mode in selected_modes:
        file_name, renderer = renderers[selected_mode]
        output_paths[selected_mode.value] = save_image(
            target_dir=target_dir,
            file_name=file_name,
            image=renderer(),
        )

    return output_paths


def main(
    source_video_path: Optional[str],
    target_video_path: Optional[str],
    target_dir: str,
    device: str,
    mode: Mode,
    player_model_path: str,
    ball_model_path: str,
    court_model_path: str,
) -> Optional[Dict[str, str]]:
    if mode in {
        Mode.COURT_RENDERING,
        Mode.POINT_RENDERING,
        Mode.PATH_RENDERING,
        Mode.ALL_RENDERINGS,
    }:
        return run_rendering_mode(target_dir=target_dir, mode=mode)

    source_video_path, target_video_path = require_video_paths(
        source_video_path, target_video_path, mode
    )

    if mode == Mode.COURT_DETECTION:
        frame_generator = run_court_detection(
            source_video_path=source_video_path,
            device=device,
            model_path=court_model_path,
        )
    elif mode == Mode.PLAYER_DETECTION:
        frame_generator = run_player_detection(
            source_video_path=source_video_path,
            device=device,
            model_path=player_model_path,
        )
    elif mode == Mode.BALL_DETECTION:
        frame_generator = run_ball_detection(
            source_video_path=source_video_path,
            device=device,
            model_path=ball_model_path,
        )
    elif mode == Mode.RADAR:
        frame_generator = run_radar(
            source_video_path=source_video_path,
            device=device,
            player_model_path=player_model_path,
            court_model_path=court_model_path,
        )
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented.")

    write_video(
        source_video_path=source_video_path,
        target_video_path=target_video_path,
        frame_generator=frame_generator,
    )
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--source_video_path", type=str)
    parser.add_argument("--target_video_path", type=str)
    parser.add_argument("--target_dir", type=str, default=DEFAULT_TARGET_DIR)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mode", type=Mode, default=Mode.COURT_RENDERING)
    parser.add_argument(
        "--player_model_path", type=str, default=PLAYER_DETECTION_MODEL_PATH
    )
    parser.add_argument("--ball_model_path", type=str, default=BALL_DETECTION_MODEL_PATH)
    parser.add_argument(
        "--court_model_path", type=str, default=COURT_DETECTION_MODEL_PATH
    )
    args = parser.parse_args()

    paths = main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        target_dir=args.target_dir,
        device=args.device,
        mode=args.mode,
        player_model_path=args.player_model_path,
        ball_model_path=args.ball_model_path,
        court_model_path=args.court_model_path,
    )
    if paths is not None:
        for mode, target_path in paths.items():
            print(f"{mode}: {target_path}")
