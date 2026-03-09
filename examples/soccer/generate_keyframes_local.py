"""Local keyframe generator using YOLO-World (no SAM3 needed).

Runs on CPU or MPS (Apple Silicon). Uses YOLO-World open-vocab detection
instead of SAM3, so no special weights required.

Usage:
    python3.11 generate_keyframes_local.py \\
        --source_video_path ~/Documents/video.mp4 \\
        --output_path keyframes.json \\
        --crop_width 1440   # 4:3 of 1080 height
"""
import argparse
import json
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLOWorld


# ---------------------------------------------------------------------------
# Smoothing + RDP
# ---------------------------------------------------------------------------

class _OffsetSmoother:
    """Exponential smoother with per-frame dynamic alpha and speed cap."""

    def __init__(self) -> None:
        self.value: Optional[float] = None

    def update(self, target: float, alpha: float, max_delta_per_frame: float) -> float:
        if self.value is None:
            self.value = target
            return self.value
        filtered = self.value + alpha * (target - self.value)
        delta = filtered - self.value
        if abs(delta) > max_delta_per_frame:
            delta = float(np.sign(delta)) * max_delta_per_frame
        self.value += delta
        return self.value


def _rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
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
        left = _rdp(points[:idx + 2], epsilon)
        right = _rdp(points[idx + 1:], epsilon)
        return np.vstack((left[:-1], right))
    return points[[0, -1]]


# ---------------------------------------------------------------------------
# Camera motion detection (optical flow)
# ---------------------------------------------------------------------------

# Downsample factor for optical flow — faster with minimal accuracy loss
_FLOW_SCALE = 0.25

# Camera pan speed thresholds in px/s (full resolution):
#   below STILL → treat camera as stationary, full corrections apply
#   above MOVING → camera clearly panning, suppress our corrections
_CAM_STILL_PX_S = 15.0
_CAM_MOVING_PX_S = 80.0


def _camera_pan_px_s(prev_gray_small: np.ndarray, curr_gray_small: np.ndarray, fps: float) -> float:
    """Estimate broadcast camera x-pan speed in px/s (full resolution).

    Returns a large value when optical flow fails (fast pan / motion blur) so
    that the caller treats an untrackable frame as a definite camera move.
    """
    h, w = prev_gray_small.shape
    ys = np.linspace(h * 0.1, h * 0.9, 8, dtype=np.float32)
    xs = np.linspace(w * 0.1, w * 0.9, 12, dtype=np.float32)
    pts = np.array([[x, y] for y in ys for x in xs], dtype=np.float32).reshape(-1, 1, 2)

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray_small, curr_gray_small, pts, None)
    if status is None:
        return _CAM_MOVING_PX_S * 2  # can't track → assume large camera move

    good = status.ravel() == 1
    if good.sum() < 8:
        return _CAM_MOVING_PX_S * 2  # too few tracks → assume large camera move

    dx_scaled = float(np.median(next_pts[good, 0, 0] - pts[good, 0, 0]))
    return abs(dx_scaled / _FLOW_SCALE) * fps


# ---------------------------------------------------------------------------
# Ball velocity estimation
# ---------------------------------------------------------------------------

_VEL_WINDOW = 15        # frames (~0.5s at 30fps)
_BALL_VEL_LOW_PX_S = 60.0
_BALL_VEL_HIGH_PX_S = 350.0
_BALL_RANGE_TIGHT_PX = 80.0

# Ball velocity only reduces reactivity (lazy when still), never amplifies above baseline.
# We don't want to chase the ball aggressively — the broadcast camera already does that.
_ALPHA_MULT_LAZY = 0.25   # stationary ball → 25% of base alpha
_ALPHA_MULT_ACTIVE = 1.0  # moving ball → full base alpha (no amplification)
_SPEED_MULT_LAZY = 0.25
_SPEED_MULT_ACTIVE = 1.0


def _ball_velocity_scale(ball_cx_history: Deque[float], fps: float) -> float:
    """Return 0.0 (stationary) … 1.0 (sprinting) from recent ball x positions."""
    n = len(ball_cx_history)
    if n < 3:
        return 0.5

    xs = np.array(ball_cx_history)
    if float(xs.max() - xs.min()) < _BALL_RANGE_TIGHT_PX:
        return 0.0

    slope_px_per_frame = float(np.polyfit(np.arange(n, dtype=float), xs, 1)[0])
    vel_px_s = abs(slope_px_per_frame) * fps
    return float(np.clip((vel_px_s - _BALL_VEL_LOW_PX_S) / (_BALL_VEL_HIGH_PX_S - _BALL_VEL_LOW_PX_S), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PLAYER_CLASSES = ["football player", "soccer player", "referee"]
BALL_CLASSES = ["football", "soccer ball"]
BALL_WEIGHT = 3.0


def generate_keyframes(
    source_video_path: str,
    crop_width_px: int = 1440,
    margin_px: int = 32,
    base_alpha: float = 0.18,
    base_speed: float = 800.0,
    epsilon_frac: float = 0.008,
    conf: float = 0.15,
    device: str = "mps",
) -> List[dict]:
    model = YOLOWorld("yolov8x-worldv2.pt")
    model.set_classes(PLAYER_CLASSES + BALL_CLASSES)
    n_player = len(PLAYER_CLASSES)

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    fps = video_info.fps
    width = video_info.width
    crop_w = min(crop_width_px, width)
    max_offset = max(0, width - crop_w)
    epsilon_px = width * epsilon_frac

    pan_frac = max_offset / width if width > 0 else 0.5
    base_alpha_scaled = base_alpha * pan_frac
    base_speed_scaled = base_speed * pan_frac

    smoother = _OffsetSmoother()
    ball_cx_history: Deque[float] = deque(maxlen=_VEL_WINDOW)
    prev_gray_small: Optional[np.ndarray] = None

    timestamps: List[float] = []
    offsets: List[float] = []
    frame_indices: List[int] = []

    results = model.track(
        source=source_video_path,
        conf=conf,
        stream=True,
        device=device,
        verbose=False,
        tracker="bytetrack.yaml",
        persist=True,
    )

    for frame_idx, result in enumerate(results):
        if frame_idx % 30 == 0:
            print(f"  frame {frame_idx}/{video_info.total_frames}", flush=True)

        # --- Camera motion detection via optical flow ---
        cam_pan_speed = 0.0
        if result.orig_img is not None:
            curr_gray = cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2GRAY)
            curr_gray_small = cv2.resize(curr_gray, None, fx=_FLOW_SCALE, fy=_FLOW_SCALE)
            if prev_gray_small is not None:
                cam_pan_speed = _camera_pan_px_s(prev_gray_small, curr_gray_small, fps)
            prev_gray_small = curr_gray_small

        # cam_suppress: 0 = camera still (full corrections), 1 = camera panning (freeze)
        cam_suppress = float(np.clip(
            (cam_pan_speed - _CAM_STILL_PX_S) / (_CAM_MOVING_PX_S - _CAM_STILL_PX_S),
            0.0, 1.0
        ))

        # --- Object detection ---
        player_boxes = np.empty((0, 4), dtype=np.float32)
        ball_boxes = np.empty((0, 4), dtype=np.float32)

        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float32)
            cls = result.boxes.cls.cpu().numpy().astype(int)
            player_mask = cls < n_player
            player_boxes = xyxy[player_mask]
            ball_boxes = xyxy[~player_mask]

        # Update ball velocity history
        if len(ball_boxes) > 0:
            ball_cx = float(np.mean(0.5 * (ball_boxes[:, 0] + ball_boxes[:, 2])))
            ball_cx_history.append(ball_cx)

        # --- Adaptive smoothing: ball velocity + camera motion ---
        vel_scale = _ball_velocity_scale(ball_cx_history, fps)
        alpha_mult = _ALPHA_MULT_LAZY + vel_scale * (_ALPHA_MULT_ACTIVE - _ALPHA_MULT_LAZY)
        speed_mult = _SPEED_MULT_LAZY + vel_scale * (_SPEED_MULT_ACTIVE - _SPEED_MULT_LAZY)

        # When broadcast camera is panning, nearly freeze our crop to avoid fighting
        # the director's move. 5% minimum retained so we don't drift forever.
        suppress_factor = 1.0 - cam_suppress * 0.95
        effective_alpha = float(np.clip(base_alpha_scaled * alpha_mult * suppress_factor, 0.0, 1.0))
        effective_max_delta = (base_speed_scaled * speed_mult * suppress_factor) / fps

        # --- Desired crop target ---
        all_boxes_list = []
        all_weights_list = []
        if len(player_boxes) > 0:
            w = np.maximum(player_boxes[:, 2] - player_boxes[:, 0], 1.0)
            all_boxes_list.append(player_boxes)
            all_weights_list.append(w)
        if len(ball_boxes) > 0:
            w = np.maximum(ball_boxes[:, 2] - ball_boxes[:, 0], 1.0)
            all_boxes_list.append(ball_boxes)
            all_weights_list.append(w * BALL_WEIGHT)

        if all_boxes_list:
            boxes = np.vstack(all_boxes_list)
            weights = np.concatenate(all_weights_list)
            x1, x2 = boxes[:, 0], boxes[:, 2]
            centers = 0.5 * (x1 + x2)
            weighted_center = float(np.average(centers, weights=weights))
            desired = float(np.clip(weighted_center - crop_w / 2.0, 0, max_offset))
            min_x = float(x1.min() - margin_px)
            max_x = float(x2.max() + margin_px)
            lower = max(0.0, max_x - crop_w)
            upper = min(float(max_offset), min_x)
            if lower > upper:
                target = float(np.clip(desired, 0.0, float(max_offset)))
            else:
                target = float(np.clip(desired, lower, upper))
        else:
            target = max_offset / 2.0

        timestamps.append(frame_idx / fps)
        offsets.append(smoother.update(target, effective_alpha, effective_max_delta))
        frame_indices.append(frame_idx)

    if not offsets:
        return []

    fi = np.array(frame_indices)
    offs = np.array(offsets)
    points = np.column_stack((fi.astype(float), offs))
    simplified = _rdp(points, epsilon_px)

    idx_lookup = []
    for point in simplified:
        nearest = int(np.argmin(np.abs(fi - point[0])))
        idx_lookup.append(nearest)
    kf_indices = sorted(dict.fromkeys(idx_lookup))

    keyframes = [{"t": timestamps[i], "o": int(round(offsets[i]))} for i in kf_indices]

    if keyframes and keyframes[-1]["t"] < timestamps[-1]:
        keyframes.append({"t": timestamps[-1], "o": int(round(offsets[-1]))})

    return keyframes


def main():
    parser = argparse.ArgumentParser(description="Generate crop keyframes using YOLO-World (local, no SAM3).")
    parser.add_argument("--source_video_path", required=True)
    parser.add_argument("--output_path", help="JSON output path (prints to stdout if omitted)")
    parser.add_argument("--crop_width", type=int, default=1440, help="Crop width in pixels (default 1440 = 4:3 of 1080h)")
    parser.add_argument("--device", default="mps", help="torch device: mps, cpu, cuda")
    parser.add_argument("--conf", type=float, default=0.15)
    args = parser.parse_args()

    print(f"[local-crop] video={args.source_video_path} crop_width={args.crop_width}px device={args.device}", flush=True)
    keyframes = generate_keyframes(
        source_video_path=args.source_video_path,
        crop_width_px=args.crop_width,
        device=args.device,
        conf=args.conf,
    )
    print(f"[local-crop] generated {len(keyframes)} keyframes", flush=True)

    if args.output_path:
        Path(args.output_path).write_text(json.dumps(keyframes, indent=2))
        print(f"[local-crop] written to {args.output_path}", flush=True)
    else:
        for kf in keyframes:
            print(f"{kf['t']:.3f},{kf['o']}")


if __name__ == "__main__":
    main()
