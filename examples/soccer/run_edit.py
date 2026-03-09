#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate keyframes and render slide-transitions edit")
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_video", required=True)
    parser.add_argument("--device", default="cpu", help="cuda, mps, or cpu")
    parser.add_argument("--transition", type=float, default=0.5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--crop_width", type=int, default=1080)
    parser.add_argument("--margin", type=int, default=32)
    parser.add_argument("--smoothing_alpha", type=float, default=0.25)
    parser.add_argument("--max_speed", type=float, default=480.0)
    parser.add_argument("--epsilon", type=float, default=12.0)
    parser.add_argument("--disable_ball", action="store_true")
    args = parser.parse_args()

    # Resolve paths
    here = os.path.dirname(os.path.abspath(__file__))
    gen_py = os.path.join(here, "generate_keyframes.py")
    rend_py = os.path.join(here, "render_from_keyframes.py")

    with tempfile.TemporaryDirectory() as td:
        keyframes_path = os.path.join(td, "keyframes.json")

        # 1) Generate keyframes JSON
        gen_cmd = [
            sys.executable,
            gen_py,
            "--source_video_path",
            args.input_video,
            "--output_path",
            keyframes_path,
            "--device",
            args.device,
            "--stride",
            str(args.stride),
            "--crop_width",
            str(args.crop_width),
            "--margin",
            str(args.margin),
            "--smoothing_alpha",
            str(args.smoothing_alpha),
            "--max_speed",
            str(args.max_speed),
            "--epsilon",
            str(args.epsilon),
        ]
        if args.disable_ball:
            gen_cmd.append("--disable_ball")

        subprocess.check_call(gen_cmd)

        # 2) Render single video with slide transitions
        rend_cmd = [
            sys.executable,
            rend_py,
            "--input_video",
            args.input_video,
            "--keyframes_json",
            keyframes_path,
            "--output",
            args.output_video,
            "--transition",
            str(args.transition),
        ]
        subprocess.check_call(rend_cmd)

        print(f"Wrote {args.output_video}")


if __name__ == "__main__":
    main()




