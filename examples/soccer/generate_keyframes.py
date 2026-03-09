import argparse
import json
import sys
from typing import List

from sports.pipelines import FOOTBALL, TENNIS, Keyframe, KeyframeGenerator, SportConfig

SPORTS: dict[str, SportConfig] = {
    "football": FOOTBALL,
    "soccer": FOOTBALL,
    "tennis": TENNIS,
}


def write_output(path: str, keyframes: List[Keyframe]) -> None:
    payload = [keyframe.as_dict() for keyframe in keyframes]
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate (t, o) crop keyframes for sports footage using SAM 3."
    )
    parser.add_argument("--source_video_path", type=str, required=True)
    parser.add_argument(
        "--sport",
        type=str,
        default="football",
        choices=list(SPORTS.keys()),
        help="Sport type. Default: football.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Where to write keyframes JSON. Prints to stdout if omitted.",
    )
    parser.add_argument("--model_path", type=str, default="sam3.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--crop_width", type=int, default=1080)
    parser.add_argument("--margin", type=int, default=32)
    parser.add_argument("--smoothing_alpha", type=float, default=0.25)
    parser.add_argument("--max_speed", type=float, default=480.0)
    parser.add_argument(
        "--epsilon_frac",
        type=float,
        default=0.008,
        help="RDP tolerance as fraction of frame width (default 0.008 = 0.8%%).",
    )
    args = parser.parse_args()

    sport = SPORTS[args.sport]

    generator = KeyframeGenerator(
        sport=sport,
        model_path=args.model_path,
        device=args.device,
        crop_width_px=args.crop_width,
        margin_px=args.margin,
        smoothing_alpha=args.smoothing_alpha,
        max_speed_px_per_s=args.max_speed,
        epsilon_frac=args.epsilon_frac,
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
