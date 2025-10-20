from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import supervision as sv
from ultralytics import YOLO


@dataclass(frozen=True)
class Keyframe:
    """
    Representation of a crop keyframe.

    Attributes:
        timestamp_s: Time offset from the beginning of the clip in seconds.
        offset_px: Horizontal crop offset in pixels from the left side.
    """

    timestamp_s: float
    offset_px: int

    def as_pair(self) -> Tuple[float, int]:
        """Return the keyframe as a ``(t, o)`` tuple."""
        return self.timestamp_s, self.offset_px

    def as_dict(self) -> dict:
        """Return the keyframe as a dictionary with ``t`` and ``o``."""
        return {"t": self.timestamp_s, "o": self.offset_px}


class _OffsetSmoother:
    """Low-pass filter with bounded slew rate to avoid jitter and jump cuts."""

    def __init__(
        self,
        fps: float,
        alpha: float,
        max_speed_px_per_s: float,
    ) -> None:
        self.alpha = np.clip(alpha, 0.0, 1.0)
        self.max_delta_per_frame = (
            max_speed_px_per_s / fps if fps > 0 else max_speed_px_per_s
        )
        self.value: Optional[float] = None

    def reset(self) -> None:
        self.value = None

    def update(self, target: float) -> float:
        if self.value is None:
            self.value = target
            return self.value

        filtered = self.value + self.alpha * (target - self.value)
        delta = filtered - self.value
        if abs(delta) > self.max_delta_per_frame:
            delta = np.sign(delta) * self.max_delta_per_frame
        self.value += delta
        return self.value


def _rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Ramer–Douglas–Peucker simplification for 2D points.

    Args:
        points: Array of shape (N, 2) containing ordered points (x, y).
        epsilon: Maximum perpendicular distance tolerated.

    Returns:
        Simplified array of points.
    """
    if points.shape[0] <= 2:
        return points

    start = points[0]
    end = points[-1]
    segment = end - start
    segment_norm = np.linalg.norm(segment)

    if segment_norm == 0:
        distances = np.linalg.norm(points[1:-1] - start, axis=1)
    else:
        cross = np.cross(segment, start - points[1:-1])
        distances = np.abs(cross) / segment_norm

    if distances.size == 0:
        return points[[0, -1]]

    index = np.argmax(distances)
    max_distance = distances[index]

    if max_distance > epsilon:
        left = _rdp(points[: index + 2], epsilon)
        right = _rdp(points[index + 1 :], epsilon)
        return np.vstack((left[:-1], right))
    return points[[0, -1]]


class KeyframeGenerator:
    """
    Generate crop keyframes for converting a 16:9 video into a centered square crop.
    """

    def __init__(
        self,
        player_model_path: str,
        device: str = "cpu",
        *,
        ball_model_path: Optional[str] = None,
        stride: int = 1,
        crop_width_px: int = 1080,
        margin_px: int = 32,
        smoothing_alpha: float = 0.25,
        max_speed_px_per_s: float = 480.0,
        compression_epsilon_px: float = 12.0,
        player_confidence_threshold: float = 0.35,
        ball_confidence_threshold: float = 0.25,
        player_imgsz: int = 1280,
        ball_imgsz: int = 640,
        ball_slice_wh: Tuple[int, int] = (640, 640),
    ) -> None:
        """
        Args:
            player_model_path: Path to the Ultralytics YOLO player detection model.
            device: Torch device string.
            ball_model_path: Path to the ball detection model (optional).
            stride: Frame stride used when sampling the video.
            crop_width_px: Width of the target square crop.
            margin_px: Additional padding to keep around detected players/ball.
            smoothing_alpha: Exponential smoothing coefficient for offsets.
            max_speed_px_per_s: Maximum allowed change of offset per second.
            compression_epsilon_px: Maximum pixel error after RDP compression.
            player_confidence_threshold: Confidence cut-off for player boxes.
            ball_confidence_threshold: Confidence cut-off for ball detections.
            player_imgsz: Inference resolution for player detection.
            ball_imgsz: Inference resolution for ball detection.
            ball_slice_wh: Slice size for tiled ball inference.
        """
        self.player_model = YOLO(player_model_path).to(device=device)
        self.player_imgsz = player_imgsz
        self.player_confidence_threshold = player_confidence_threshold

        self.ball_model: Optional[YOLO] = None
        self.ball_slicer: Optional[sv.InferenceSlicer] = None
        self.ball_confidence_threshold = ball_confidence_threshold
        if ball_model_path:
            self.ball_model = YOLO(ball_model_path).to(device=device)

            def callback(image_slice: np.ndarray) -> sv.Detections:
                result = self.ball_model(
                    image_slice, imgsz=ball_imgsz, verbose=False
                )[0]
                detections = sv.Detections.from_ultralytics(result)
                if len(detections) == 0:
                    return detections
                mask = detections.confidence >= self.ball_confidence_threshold
                return detections[mask]

            self.ball_slicer = sv.InferenceSlicer(
                callback=callback,
                overlap_filter=sv.OverlapFilter.NONE,
                slice_wh=ball_slice_wh,
            )

        self.stride = max(1, stride)
        self.crop_width_px = crop_width_px
        self.margin_px = margin_px
        self.smoothing_alpha = smoothing_alpha
        self.max_speed_px_per_s = max_speed_px_per_s
        self.compression_epsilon_px = compression_epsilon_px

    def generate(self, source_video_path: str) -> List[Keyframe]:
        """
        Generate keyframes for the provided video.

        Args:
            source_video_path: Path to the input video file.

        Returns:
            List of Keyframe objects containing timestamps and offsets.
        """
        video_info = sv.VideoInfo.from_video_path(source_video_path)
        crop_width_px = min(self.crop_width_px, video_info.width)
        max_offset = max(0, video_info.width - crop_width_px)

        smoother = _OffsetSmoother(
            fps=video_info.fps,
            alpha=self.smoothing_alpha,
            max_speed_px_per_s=self.max_speed_px_per_s,
        )

        timestamps: List[float] = []
        offsets: List[float] = []
        frame_indices: List[int] = []

        frame_generator = sv.get_video_frames_generator(
            source_path=source_video_path,
            stride=self.stride,
        )

        frame_number = 0
        for frame in frame_generator:
            boxes = self._collect_action_boxes(frame)
            target_offset = self._compute_target_offset(
                boxes=boxes,
                frame_width=video_info.width,
                crop_width=crop_width_px,
                max_offset=max_offset,
            )
            smoothed_offset = smoother.update(target_offset)

            timestamps.append(frame_number / video_info.fps)
            offsets.append(smoothed_offset)
            frame_indices.append(frame_number)

            frame_number += self.stride

        if not offsets:
            return []

        keyframe_indices = self._compress_offsets(
            frame_indices=np.array(frame_indices),
            offsets=np.array(offsets),
            epsilon=self.compression_epsilon_px,
        )

        keyframes: List[Keyframe] = []
        for idx in keyframe_indices:
            timestamp = timestamps[idx]
            offset = int(round(offsets[idx]))
            keyframes.append(Keyframe(timestamp_s=timestamp, offset_px=offset))

        # Ensure the last frame is included.
        if keyframes and keyframes[-1].timestamp_s < timestamps[-1]:
            keyframes.append(
                Keyframe(
                    timestamp_s=timestamps[-1], offset_px=int(round(offsets[-1]))
                )
            )
        return keyframes

    def _collect_action_boxes(self, frame: np.ndarray) -> np.ndarray:
        boxes: List[np.ndarray] = []

        player_result = self.player_model(frame, imgsz=self.player_imgsz, verbose=False)[
            0
        ]
        player_detections = sv.Detections.from_ultralytics(player_result)
        if len(player_detections) > 0:
            mask = player_detections.confidence >= self.player_confidence_threshold
            if mask.any():
                boxes.append(player_detections.xyxy[mask])

        if self.ball_slicer is not None:
            ball_detections = self.ball_slicer(frame).with_nms(threshold=0.1)
            if len(ball_detections) > 0:
                boxes.append(ball_detections.xyxy)

        if not boxes:
            return np.empty((0, 4), dtype=np.float32)
        return np.vstack(boxes).astype(np.float32)

    def _compute_target_offset(
        self,
        boxes: np.ndarray,
        frame_width: int,
        crop_width: int,
        max_offset: int,
    ) -> float:
        if boxes.size == 0:
            return max_offset / 2.0

        x1 = boxes[:, 0]
        x2 = boxes[:, 2]
        centers = 0.5 * (x1 + x2)
        widths = np.maximum(x2 - x1, 1.0)
        weights = widths

        weighted_center = float(np.average(centers, weights=weights))
        desired_offset = weighted_center - crop_width / 2.0
        desired_offset = float(np.clip(desired_offset, 0, max_offset))

        min_x = float(x1.min() - self.margin_px)
        max_x = float(x2.max() + self.margin_px)

        lower_bound = max(0.0, max_x - crop_width)
        upper_bound = min(float(max_offset), min_x)

        if lower_bound > upper_bound:
            # The action spans wider than the crop; favor desired offset but clamp.
            return float(np.clip(desired_offset, 0.0, float(max_offset)))

        constrained_offset = float(
            np.clip(desired_offset, lower_bound, upper_bound)
        )
        return constrained_offset

    def _compress_offsets(
        self,
        frame_indices: np.ndarray,
        offsets: np.ndarray,
        epsilon: float,
    ) -> Sequence[int]:
        points = np.column_stack((frame_indices.astype(float), offsets))
        simplified = _rdp(points, epsilon)

        # Map simplified points back to original indices by nearest neighbour.
        idx_lookup = []
        for point in simplified:
            frame_idx = point[0]
            nearest = int(np.argmin(np.abs(frame_indices - frame_idx)))
            idx_lookup.append(nearest)

        # Keep indices sorted and unique.
        return sorted(dict.fromkeys(idx_lookup))
