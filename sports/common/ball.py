from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from sports.common.kinematics import feet_xy
from sports.configs.soccer import BALL_CLASS_ID

DEFAULT_BALL_MODEL_PATH: str | None = None


class BallAnnotator:
    """
    A class to annotate frames with circles of varying radii and colors.

    Attributes:
        radius (int): The maximum radius of the circles to be drawn.
        buffer (deque): A deque buffer to store recent coordinates for annotation.
        color_palette (sv.ColorPalette): A color palette for the circles.
        thickness (int): The thickness of the circle borders.
    """

    def __init__(self, radius: int, buffer_size: int = 5, thickness: int = 2):

        self.color_palette = sv.ColorPalette.from_matplotlib('jet', buffer_size)
        self.buffer = deque(maxlen=buffer_size)
        self.radius = radius
        self.thickness = thickness

    def interpolate_radius(self, i: int, max_i: int) -> int:
        """
        Interpolates the radius between 1 and the maximum radius based on the index.

        Args:
            i (int): The current index in the buffer.
            max_i (int): The maximum index in the buffer.

        Returns:
            int: The interpolated radius.
        """
        if max_i == 1:
            return self.radius
        return int(1 + i * (self.radius - 1) / (max_i - 1))

    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """
        Annotates the frame with circles based on detections.

        Args:
            frame (np.ndarray): The frame to annotate.
            detections (sv.Detections): The detections containing coordinates.

        Returns:
            np.ndarray: The annotated frame.
        """
        xy = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER).astype(int)
        self.buffer.append(xy)
        for i, xy in enumerate(self.buffer):
            color = self.color_palette.by_idx(i)
            interpolated_radius = self.interpolate_radius(i, len(self.buffer))
            for center in xy:
                frame = cv2.circle(
                    img=frame,
                    center=tuple(center),
                    radius=interpolated_radius,
                    color=color.as_bgr(),
                    thickness=self.thickness
                )
        return frame


class BallTracker:
    """
    A class used to track a soccer ball's position across video frames.

    The BallTracker class maintains a buffer of recent ball positions and uses this
    buffer to predict the ball's position in the current frame by selecting the
    detection closest to the average position (centroid) of the recent positions.

    Attributes:
        buffer (collections.deque): A deque buffer to store recent ball positions.
    """
    def __init__(self, buffer_size: int = 10):
        self.buffer = deque(maxlen=buffer_size)

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Updates the buffer with new detections and returns the detection closest to the
        centroid of recent positions.

        Args:
            detections (sv.Detections): The current frame's ball detections.

        Returns:
            sv.Detections: The detection closest to the centroid of recent positions.
            If there are no detections, returns the input detections.
        """
        xy = detections.get_anchors_coordinates(sv.Position.CENTER)
        self.buffer.append(xy)

        if len(detections) == 0:
            return detections

        centroid = np.mean(np.concatenate(self.buffer), axis=0)
        distances = np.linalg.norm(xy - centroid, axis=1)
        index = np.argmin(distances)
        return detections[[index]]


def _default_ball_model_path() -> str:
    return str(Path(__file__).resolve().parents[2] / "examples/soccer/data/football-ball-detection.pt")


def _ball_centers(balls: sv.Detections) -> np.ndarray:
    """Ground positions (cx, y2) for each ball detection."""
    cx = (balls.xyxy[:, 0] + balls.xyxy[:, 2]) / 2
    cy = balls.xyxy[:, 3]
    return np.column_stack([cx, cy])


def select_best_ball(
    balls: sv.Detections,
    *,
    players: sv.Detections | None = None,
) -> sv.Detections:
    """Pick one ball box: nearest player feet when ambiguous, else highest confidence."""
    if len(balls) <= 1:
        return balls
    if players is not None and len(players):
        feet = feet_xy(players)
        if len(feet):
            centers = _ball_centers(balls)
            dists = np.linalg.norm(
                centers[:, None, :] - feet[None, :, :], axis=2
            ).min(axis=1)
            pick = int(np.argmin(dists))
            return balls[pick : pick + 1]
    if balls.confidence is None:
        return balls[:1]
    pick = int(np.argmax(balls.confidence))
    return balls[pick : pick + 1]


def _ball_row_for_merge(ball: sv.Detections, players: sv.Detections) -> sv.Detections:
    """Make a single ball detection compatible with tracked player rows for merge."""
    if len(ball) == 0:
        return ball
    ball = ball[:1]
    kwargs: dict = {
        "xyxy": ball.xyxy,
        "class_id": ball.class_id,
        "confidence": ball.confidence,
    }
    if players.tracker_id is not None:
        kwargs["tracker_id"] = np.array([-1], dtype=int)
    if players.data:
        ball_data: dict = {}
        n_players = len(players)
        for key, val in players.data.items():
            arr = np.asarray(val)
            if arr.ndim == 0:
                ball_data[key] = arr
            elif len(arr) == n_players:
                if np.issubdtype(arr.dtype, np.floating):
                    ball_data[key] = np.array([np.nan], dtype=arr.dtype)
                else:
                    ball_data[key] = np.array([-1], dtype=arr.dtype)
            else:
                ball_data[key] = arr[:1]
        kwargs["data"] = ball_data
    return sv.Detections(**kwargs)


def attach_ball(
    dets: sv.Detections,
    ball_dets: sv.Detections,
) -> sv.Detections:
    """Append the best ball row to a player/GK detections frame."""
    ball = select_best_ball(ball_dets, players=dets)
    if len(ball) == 0:
        return dets
    ball = _ball_row_for_merge(ball, dets)
    return sv.Detections.merge([dets, ball])


def create_ball_detector(
    *,
    device: str = "cpu",
    model_path: str | None = None,
    threshold: float = 0.2,
) -> Callable[[np.ndarray], sv.Detections]:
    """Return ``detect(frame_bgr) -> sv.Detections`` for the football ball YOLO model."""
    path = model_path or DEFAULT_BALL_MODEL_PATH or _default_ball_model_path()
    model = YOLO(str(path)).to(device=device)

    def _detect(frame: np.ndarray) -> sv.Detections:
        results = model.predict(frame, conf=threshold, verbose=False, device=device)[0]
        dets = sv.Detections.from_ultralytics(results)
        if len(dets) == 0:
            return dets
        return sv.Detections(
            xyxy=dets.xyxy,
            confidence=dets.confidence,
            class_id=np.full(len(dets), BALL_CLASS_ID, dtype=int),
        )

    return _detect
