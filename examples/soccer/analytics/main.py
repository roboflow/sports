"""analytics/main.py — CLI entry point for player-motion analytics.

Run from examples/soccer/:
    python analytics/main.py DIRECTION --source_video_path input.mp4 --target_video_path out.mp4
    python analytics/main.py SPEED     --source_video_path input.mp4 --target_video_path out.mp4
    python analytics/main.py DISTANCE  --source_video_path input.mp4 --target_video_path out.mp4
    python analytics/main.py PLAYER_FOCUS --source_video_path input.mp4 --target_video_path out.mp4
    python analytics/main.py PLAYER_FOCUS --source_video_path input.mp4 --target_video_path out.mp4 --track-id 7
"""

from __future__ import annotations

import argparse
import sys
from enum import Enum
from pathlib import Path

# Make analytics/ importable as a package when run as a script.
_HERE = Path(__file__).resolve().parent
_SOCCER = _HERE.parent
if str(_SOCCER) not in sys.path:
    sys.path.insert(0, str(_SOCCER))


class Mode(Enum):
    DIRECTION = "DIRECTION"
    SPEED = "SPEED"
    DISTANCE = "DISTANCE"
    PLAYER_FOCUS = "PLAYER_FOCUS"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Player-motion analytics (direction / speed / distance / player_focus)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "mode",
        type=str,
        choices=[m.value for m in Mode],
        help="Analytics mode",
    )

    # ── shared video flags ──────────────────────────────────────────────────
    parser.add_argument("--source_video_path", required=True, help="Input video file")
    parser.add_argument("--target_video_path", required=True, help="Output video file")
    parser.add_argument("--device", default="cpu", help="Torch device (cpu / cuda / mps)")
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Cap frames processed (None = all)"
    )

    # ── tracker + detector selection ────────────────────────────────────────
    parser.add_argument(
        "--tracker",
        default="botsort",
        choices=("bytetrack", "botsort", "botsort_nocmc"),
        help="Multi-object tracker backend",
    )
    parser.add_argument(
        "--player-detector",
        dest="player_detector",
        default="yolo",
        choices=("yolo", "inference"),
        help="Player detection backend",
    )
    parser.add_argument(
        "--pitch-detector",
        dest="pitch_detector",
        default="yolo",
        choices=("yolo", "inference"),
        help="Pitch keypoint detection backend (used by SPEED / DISTANCE / PLAYER_FOCUS)",
    )

    # ── model path / id overrides ────────────────────────────────────────────
    parser.add_argument(
        "--player-model-path",
        dest="player_model_path",
        default=None,
        help="Path to YOLO player detection .pt (overrides default data/ path)",
    )
    parser.add_argument(
        "--pitch-model-path",
        dest="pitch_model_path",
        default=None,
        help="Path to YOLO pitch keypoint .pt (overrides default data/ path)",
    )
    parser.add_argument(
        "--player-model-id",
        dest="player_model_id",
        default="football-players-detection-3zvbc/11",
        help="Roboflow Inference model id for player detection",
    )
    parser.add_argument(
        "--pitch-model-id",
        dest="pitch_model_id",
        default="football-field-detection-f07vi/15",
        help="Roboflow Inference model id for pitch keypoints",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help="Roboflow API key (also read from ROBOFLOW_API_KEY env var)",
    )

    # ── PLAYER_FOCUS ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--track-id",
        dest="track_id",
        type=int,
        default=None,
        help="(PLAYER_FOCUS) Spotlight one tracker id; omit to follow all players",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    mode = Mode(args.mode)

    if mode == Mode.DIRECTION:
        from analytics.direction import run_direction
        run_direction(args)
    elif mode == Mode.SPEED:
        from analytics.speed import run_speed
        run_speed(args)
    elif mode == Mode.DISTANCE:
        from analytics.distance import run_distance
        run_distance(args)
    elif mode == Mode.PLAYER_FOCUS:
        from analytics.player_focus import run_player_focus
        run_player_focus(args)
    else:
        parser.error(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
