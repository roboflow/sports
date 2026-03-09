from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod Serverless handler. Expects JSON body like:
    {
        "input": {
            "input_path": "https://.../clip.mp4" or "s3://bucket/key",
            "output_path": "s3://bucket/out.mp4",
            "device": "cuda",
            "transition": 0.5,
            "stride": 1,
            "crop_width": 1080,
            "margin": 32,
            "smoothing_alpha": 0.25,
            "max_speed": 480.0,
            "epsilon": 12.0,
            "disable_ball": false
        }
    }
    """
    payload = event.get("input") or {}
    input_path = payload.get("input_path")
    output_path = payload.get("output_path")
    device = payload.get("device", "cuda")
    transition = float(payload.get("transition", 0.5))
    stride = int(payload.get("stride", 1))
    crop_width = int(payload.get("crop_width", 1080))
    margin = int(payload.get("margin", 32))
    smoothing_alpha = float(payload.get("smoothing_alpha", 0.25))
    max_speed = float(payload.get("max_speed", 480.0))
    epsilon = float(payload.get("epsilon", 12.0))
    disable_ball = bool(payload.get("disable_ball", False))
    player_model_s3 = payload.get("player_model_s3")
    ball_model_s3 = payload.get("ball_model_s3")

    if not input_path or not output_path:
        return {"error": "input_path and output_path are required"}

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # examples/soccer
    gen_py = os.path.join(here, "generate_keyframes.py")
    rend_py = os.path.join(here, "render_from_keyframes.py")
    data_dir = os.path.join(here, "data")
    os.makedirs(data_dir, exist_ok=True)
    player_dst = os.path.join(data_dir, "football-player-detection.pt")
    ball_dst = os.path.join(data_dir, "football-ball-detection.pt")

    with tempfile.TemporaryDirectory() as td:
        inp = os.path.join(td, "input.mp4")
        outp = os.path.join(td, "output.mp4")
        keyf = os.path.join(td, "keyframes.json")

        # Download input (supports http(s) via curl; add s3 logic if needed)
        if input_path.startswith("http://") or input_path.startswith("https://"):
            subprocess.check_call(["curl", "-L", "-o", inp, input_path])
        elif input_path.startswith("s3://"):
            # Requires awscli present and credentials
            subprocess.check_call(["aws", "s3", "cp", input_path, inp])
        else:
            # Assume it's accessible path mounted in pod
            subprocess.check_call(["cp", input_path, inp])

        # Optionally fetch models from S3
        if player_model_s3:
            subprocess.check_call(["aws", "s3", "cp", player_model_s3, player_dst])
        if ball_model_s3:
            subprocess.check_call(["aws", "s3", "cp", ball_model_s3, ball_dst])

        gen_cmd = [
            sys.executable,
            gen_py,
            "--source_video_path",
            inp,
            "--output_path",
            keyf,
            "--device",
            device,
            "--stride",
            str(stride),
            "--crop_width",
            str(crop_width),
            "--margin",
            str(margin),
            "--smoothing_alpha",
            str(smoothing_alpha),
            "--max_speed",
            str(max_speed),
            "--epsilon",
            str(epsilon),
        ]
        if disable_ball or not os.path.exists(ball_dst):
            gen_cmd.append("--disable_ball")
        subprocess.check_call(gen_cmd)

        rend_cmd = [
            sys.executable,
            rend_py,
            "--input_video",
            inp,
            "--keyframes_json",
            keyf,
            "--output",
            outp,
            "--transition",
            str(transition),
        ]
        subprocess.check_call(rend_cmd)

        if output_path.startswith("s3://"):
            subprocess.check_call(["aws", "s3", "cp", outp, output_path])
        else:
            subprocess.check_call(["cp", outp, output_path])

        return {"output_path": output_path}


