import cv2
import argparse
import supervision as sv
from ultralytics import YOLO
from sports.configs.soccer import SoccerFieldConfiguration
from enum import Enum
from typing import Iterator
import numpy as np


class Mode(Enum):
    PITCH_DETECTION = 'PITCH_DETECTION'
    PLAYER_DETECTION = 'PLAYER_DETECTION'


PLAYER_DETECTION_MODEL_PATH = 'examples/soccer/data/football-player-detection-v9.pt'
PITCH_DETECTION_MODEL_PATH = 'examples/soccer/data/football-pitch-detection-v9.pt'
CONFIG = SoccerFieldConfiguration()

VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    border_radius=5,
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_thickness=1,
    text_scale=0.5,
    text_padding=5
)
COLOR_PALETTE = sv.ColorPalette.from_hex(['#FF1493', '#00BFFF', '#FF6347', '#FFD700'])
BOX_ANNOTATOR = sv.BoxAnnotator(color=COLOR_PALETTE, thickness=1)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLOR_PALETTE,
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5
)


def run_pitch_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(
            annotated_frame, keypoints, CONFIG.labels)
        yield annotated_frame


def run_player_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = LABEL_ANNOTATOR.annotate(annotated_frame, detections)
        yield annotated_frame


def main(source_video_path: str, target_video_path: str, device: str, mode: Mode) -> None:
    if mode == Mode.PITCH_DETECTION:
        frame_generator = run_pitch_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_DETECTION:
        frame_generator = run_player_detection(
            source_video_path=source_video_path, device=device)
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented.")

    for frame in frame_generator:
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--source_video_path', type=str)
    parser.add_argument('--target_video_path', type=str)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--mode', type=Mode, default=Mode.PLAYER_DETECTION)
    args = parser.parse_args()
    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device,
        mode=args.mode
    )
