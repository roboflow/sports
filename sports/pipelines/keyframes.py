from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import supervision as sv
from ultralytics.models.sam import SAM3VideoSemanticPredictor


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Keyframe:
    """A single crop keyframe.

    Attributes:
        timestamp_s: Time offset from the start of the clip in seconds.
        offset_px: Horizontal crop offset in pixels from the left edge.
    """

    timestamp_s: float
    offset_px: int

    def as_pair(self) -> Tuple[float, int]:
        """Return ``(timestamp_s, offset_px)``."""
        return self.timestamp_s, self.offset_px

    def as_dict(self) -> dict:
        """Return ``{"t": timestamp_s, "o": offset_px}``."""
        return {"t": self.timestamp_s, "o": self.offset_px}


@dataclass
class SportConfig:
    """SAM 3 prompt configuration for a sport.

    Attributes:
        player_prompts: Text concepts describing players/athletes on the field.
        ball_prompts: Text concepts describing the ball or puck.
        ball_weight: Multiplier applied to ball box widths when computing the
            weighted crop centre. Higher values cause the crop to track the
            ball more aggressively than the player cluster.
        conf: SAM 3 detection confidence threshold.
    """

    player_prompts: List[str]
    ball_prompts: List[str]
    ball_weight: float = 3.0
    conf: float = 0.25

    @property
    def all_prompts(self) -> List[str]:
        """All prompts ordered players-first then ball — matches SAM 3 class indices."""
        return self.player_prompts + self.ball_prompts

    @property
    def n_player_classes(self) -> int:
        return len(self.player_prompts)


# ---------------------------------------------------------------------------
# Pre-built sport configs
# ---------------------------------------------------------------------------

FOOTBALL = SportConfig(
    player_prompts=["football player", "soccer player", "referee"],
    ball_prompts=["football", "soccer ball"],
    ball_weight=3.0,
    conf=0.25,
)

TENNIS = SportConfig(
    player_prompts=["tennis player"],
    ball_prompts=["tennis ball"],
    # Ball dominates in tennis — it determines shot direction and rally phase.
    # Players move little relative to the baseline; the ball crosses the full width.
    ball_weight=8.0,
    # Lower threshold because a fast-moving tennis ball is often motion-blurred.
    conf=0.15,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _OffsetSmoother:
    """Low-pass filter with bounded slew rate to prevent jitter and jump cuts."""

    def __init__(self, fps: float, alpha: float, max_speed_px_per_s: float) -> None:
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.max_delta_per_frame = max_speed_px_per_s / fps if fps > 0 else max_speed_px_per_s
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
            delta = float(np.sign(delta)) * self.max_delta_per_frame
        self.value += delta
        return self.value


def _rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer–Douglas–Peucker polyline simplification.

    Args:
        points: Array of shape ``(N, 2)``.
        epsilon: Maximum perpendicular distance allowed before splitting.

    Returns:
        Simplified array of shape ``(M, 2)`` where M ≤ N.
    """
    if points.shape[0] <= 2:
        return points

    start, end = points[0], points[-1]
    segment = end - start
    norm = np.linalg.norm(segment)

    if norm == 0:
        distances = np.linalg.norm(points[1:-1] - start, axis=1)
    else:
        cross = np.cross(segment, start - points[1:-1])
        distances = np.abs(cross) / norm

    if distances.size == 0:
        return points[[0, -1]]

    idx = int(np.argmax(distances))
    if distances[idx] > epsilon:
        left = _rdp(points[: idx + 2], epsilon)
        right = _rdp(points[idx + 1 :], epsilon)
        return np.vstack((left[:-1], right))
    return points[[0, -1]]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class KeyframeGenerator:
    """Generate crop keyframes from a sports video using SAM 3.

    SAM 3 detects and tracks players and the ball across the entire video
    using text concept prompts — no pre-trained sport-specific YOLO weights
    required.  The per-frame detections are fed through a smoothing filter
    and RDP compression to produce a compact list of
    ``(timestamp_s, offset_px)`` keyframes compatible with the LIGR
    ``highlightCropRequest`` API.

    Args:
        sport: A :class:`SportConfig` instance — use ``FOOTBALL`` or
            ``TENNIS``, or build a custom one.
        model_path: Path to the ``sam3.pt`` checkpoint. Download from
            https://huggingface.co/facebook/sam3 and place alongside this
            script (weights are not auto-downloaded).
        device: PyTorch device string.  Use ``"cuda"`` on GPU instances.
        crop_width_px: Width of the target crop in pixels.  For a 1:1 square
            crop of 1920-wide footage use ``1080``.
        margin_px: Extra padding around detected boxes when constraining the
            crop window to keep objects fully visible.
        smoothing_alpha: EWA smoothing coefficient (0 = frozen, 1 = instant).
        max_speed_px_per_s: Maximum crop offset change per second. Prevents
            whip-pans when the ball crosses the frame quickly.
        epsilon_frac: RDP tolerance as a fraction of frame width. ``0.008``
            (0.8 %) keeps meaningful inflections while discarding noise.

    Example::

        gen = KeyframeGenerator(sport=TENNIS)
        keyframes = gen.generate("rally_highlight.mp4")
        # → [{"t": 0.0, "o": 420}, {"t": 3.2, "o": 680}, ...]
    """

    def __init__(
        self,
        sport: SportConfig,
        model_path: str = "sam3.pt",
        device: str = "cuda",
        *,
        crop_width_px: int = 1080,
        margin_px: int = 32,
        smoothing_alpha: float = 0.25,
        max_speed_px_per_s: float = 480.0,
        epsilon_frac: float = 0.008,
    ) -> None:
        self.sport = sport
        self.crop_width_px = crop_width_px
        self.margin_px = margin_px
        self.smoothing_alpha = smoothing_alpha
        self.max_speed_px_per_s = max_speed_px_per_s
        self.epsilon_frac = epsilon_frac

        self._predictor = SAM3VideoSemanticPredictor(
            overrides=dict(
                conf=sport.conf,
                task="segment",
                mode="predict",
                model=model_path,
                device=device,
                verbose=False,
                half=device != "cpu",
            )
        )

    def generate(self, source_video_path: str) -> List[Keyframe]:
        """Run SAM 3 on *source_video_path* and return crop keyframes.

        Args:
            source_video_path: Path to the input video file.

        Returns:
            Sorted list of :class:`Keyframe` objects ready to pass to the
            LIGR ``highlightCropRequest`` mutation as ``crops``.
        """
        video_info = sv.VideoInfo.from_video_path(source_video_path)
        crop_width_px = min(self.crop_width_px, video_info.width)
        max_offset = max(0, video_info.width - crop_width_px)
        # RDP epsilon is relative to frame width so it scales with resolution.
        epsilon_px = video_info.width * self.epsilon_frac

        smoother = _OffsetSmoother(
            fps=video_info.fps,
            alpha=self.smoothing_alpha,
            max_speed_px_per_s=self.max_speed_px_per_s,
        )

        timestamps: List[float] = []
        offsets: List[float] = []
        frame_indices: List[int] = []

        # SAM 3 streams results one frame at a time while maintaining internal
        # tracking state across frames — bridging missed detections automatically.
        results = self._predictor(
            source=source_video_path,
            text=self.sport.all_prompts,
            stream=True,
        )

        for frame_idx, result in enumerate(results):
            player_boxes, ball_boxes = self._split_boxes(result)
            target_offset = self._compute_target_offset(
                player_boxes=player_boxes,
                ball_boxes=ball_boxes,
                frame_width=video_info.width,
                crop_width=crop_width_px,
                max_offset=max_offset,
            )
            timestamps.append(frame_idx / video_info.fps)
            offsets.append(smoother.update(target_offset))
            frame_indices.append(frame_idx)

        if not offsets:
            return []

        keyframe_indices = self._compress_offsets(
            frame_indices=np.array(frame_indices),
            offsets=np.array(offsets),
            epsilon=epsilon_px,
        )

        keyframes: List[Keyframe] = []
        for idx in keyframe_indices:
            keyframes.append(
                Keyframe(timestamp_s=timestamps[idx], offset_px=int(round(offsets[idx])))
            )

        # Always include the final frame so the crop doesn't hang at the last keyframe.
        if keyframes and keyframes[-1].timestamp_s < timestamps[-1]:
            keyframes.append(
                Keyframe(timestamp_s=timestamps[-1], offset_px=int(round(offsets[-1])))
            )

        return keyframes

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_boxes(self, result) -> Tuple[np.ndarray, np.ndarray]:
        """Split a SAM 3 frame result into player boxes and ball boxes.

        SAM 3 assigns class indices matching the prompt order in ``all_prompts``.
        Indices ``0..n_player_classes-1`` → players; the rest → ball concepts.
        """
        empty = np.empty((0, 4), dtype=np.float32)

        if result.boxes is None or len(result.boxes) == 0:
            return empty, empty

        xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        cls = result.boxes.cls.cpu().numpy().astype(int)

        player_mask = cls < self.sport.n_player_classes
        return xyxy[player_mask], xyxy[~player_mask]

    def _compute_target_offset(
        self,
        player_boxes: np.ndarray,
        ball_boxes: np.ndarray,
        frame_width: int,
        crop_width: int,
        max_offset: int,
    ) -> float:
        """Compute the ideal crop left-offset for one frame.

        The ball receives ``sport.ball_weight`` × more influence than a player
        box of the same width, so the crop follows the ball when visible.
        """
        all_boxes: List[np.ndarray] = []
        all_weights: List[np.ndarray] = []

        if len(player_boxes) > 0:
            w = np.maximum(player_boxes[:, 2] - player_boxes[:, 0], 1.0)
            all_boxes.append(player_boxes)
            all_weights.append(w)

        if len(ball_boxes) > 0:
            w = np.maximum(ball_boxes[:, 2] - ball_boxes[:, 0], 1.0)
            all_boxes.append(ball_boxes)
            all_weights.append(w * self.sport.ball_weight)

        if not all_boxes:
            return max_offset / 2.0

        boxes = np.vstack(all_boxes)
        weights = np.concatenate(all_weights)

        x1, x2 = boxes[:, 0], boxes[:, 2]
        centers = 0.5 * (x1 + x2)
        weighted_center = float(np.average(centers, weights=weights))
        desired_offset = float(np.clip(weighted_center - crop_width / 2.0, 0, max_offset))

        # Hard constraint: no detected object should be cropped out of frame.
        min_x = float(x1.min() - self.margin_px)
        max_x = float(x2.max() + self.margin_px)
        lower_bound = max(0.0, max_x - crop_width)
        upper_bound = min(float(max_offset), min_x)

        if lower_bound > upper_bound:
            # Action wider than crop window — centre on weighted action.
            return float(np.clip(desired_offset, 0.0, float(max_offset)))

        return float(np.clip(desired_offset, lower_bound, upper_bound))

    def _compress_offsets(
        self,
        frame_indices: np.ndarray,
        offsets: np.ndarray,
        epsilon: float,
    ) -> Sequence[int]:
        """RDP-compress the offset timeline, returning surviving frame indices."""
        points = np.column_stack((frame_indices.astype(float), offsets))
        simplified = _rdp(points, epsilon)

        idx_lookup = []
        for point in simplified:
            nearest = int(np.argmin(np.abs(frame_indices - point[0])))
            idx_lookup.append(nearest)

        return sorted(dict.fromkeys(idx_lookup))
