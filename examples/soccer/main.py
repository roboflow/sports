import cv2
import argparse
import supervision as sv
from ultralytics import YOLO

from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerFieldConfiguration
from enum import Enum
from typing import Iterator
import numpy as np


class Mode(Enum):
    PITCH_DETECTION = 'PITCH_DETECTION'
    PITCH_PROJECTION = 'PITCH_PROJECTION'
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    PLAYER_TRACKING = 'PLAYER_TRACKING'


PLAYER_DETECTION_MODEL_PATH = 'examples/soccer/data/football-player-detection-v9.pt'
PITCH_DETECTION_MODEL_PATH = 'examples/soccer/data/football-pitch-detection-v9.pt'
CONFIG = SoccerFieldConfiguration(width=6800, length=10500)

VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.from_hex('#FFFFFF'),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
EDGE_ANNOTATOR = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    thickness=2,
    edges=CONFIG.edges,
)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(['#FF1493', '#00BFFF', '#FF6347', '#FFD700']),
    thickness=2
)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(['#FF1493', '#00BFFF', '#FF6347', '#FFD700']),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)


def get_projected_pitch_vertices(keypoints: sv.KeyPoints) -> np.ndarray:
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    view_transformer = ViewTransformer(
        source=np.array(CONFIG.vertices)[mask].astype(np.float32),
        target=keypoints.xy[0][mask].astype(np.float32)
    )
    return view_transformer.transform_points(
        points=np.array(CONFIG.vertices)
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


def run_pitch_projection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    pitch_projection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = pitch_projection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)
        projected_vertices = get_projected_pitch_vertices(keypoints)
        projected_keypoints = sv.KeyPoints(xy=np.array([projected_vertices]))

        annotated_frame = frame.copy()
        annotated_frame = EDGE_ANNOTATOR.annotate(annotated_frame, projected_keypoints)
        annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(annotated_frame, keypoints)
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
    elif mode == Mode.PITCH_PROJECTION:
        frame_generator = run_pitch_projection(
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
