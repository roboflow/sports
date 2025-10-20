import argparse
import json
import os
from typing import List

from sports.pipelines import KeyframeGenerator, Keyframe

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(
    PARENT_DIR, "data/football-player-detection.pt"
)
BALL_DETECTION_MODEL_PATH = os.path.join(
    PARENT_DIR, "data/football-ball-detection.pt"
)


def write_output(path: str, keyframes: List[Keyframe]) -> None:
    payload = [keyframe.as_dict() for keyframe in keyframes]
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate (t, o) crop keyframes for soccer footage."
    )
    parser.add_argument("--source_video_path", type=str, required=True)
    parser.add_argument(
        "--output_path",
        type=str,
        help="Where to write keyframes JSON. Prints to stdout if omitted.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--crop_width", type=int, default=1080)
    parser.add_argument("--margin", type=int, default=32)
    parser.add_argument("--smoothing_alpha", type=float, default=0.25)
    parser.add_argument("--max_speed", type=float, default=480.0)
    parser.add_argument("--epsilon", type=float, default=12.0)
    parser.add_argument("--player_confidence", type=float, default=0.35)
    parser.add_argument("--ball_confidence", type=float, default=0.25)
    parser.add_argument(
        "--disable_ball",
        action="store_true",
        help="Skip ball detection when computing action bounds.",
    )
    args = parser.parse_args()

    ball_model_path = None if args.disable_ball else BALL_DETECTION_MODEL_PATH

    generator = KeyframeGenerator(
        player_model_path=PLAYER_DETECTION_MODEL_PATH,
        ball_model_path=ball_model_path,
        device=args.device,
        stride=args.stride,
        crop_width_px=args.crop_width,
        margin_px=args.margin,
        smoothing_alpha=args.smoothing_alpha,
        max_speed_px_per_s=args.max_speed,
        compression_epsilon_px=args.epsilon,
        player_confidence_threshold=args.player_confidence,
        ball_confidence_threshold=args.ball_confidence,
    )

    keyframes = generator.generate(args.source_video_path)

    if args.output_path:
        write_output(args.output_path, keyframes)
    else:
        for keyframe in keyframes:
            timestamp, offset = keyframe.as_pair()
            print(f"{timestamp:.3f},{offset}")


if __name__ == "__main__":
    main()
