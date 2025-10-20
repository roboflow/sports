from __future__ import annotations

import os
import tempfile
from typing import Optional

import modal


app = modal.App("sports-soccer-edit")


image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg", "git")
    .pip_install(
        "boto3",
        "ultralytics>=8.0.0",
        "supervision>=0.21.0",
        "opencv-python-headless",
        "numpy",
        # CPU by default; switch to CUDA wheel by editing below or passing device=cuda
        "torch",
        "torchvision",
        "torchaudio",
    )
    .copy_local_dir(".", "/workspace")
)


@app.function(image=image, gpu="T4", secrets=[modal.Secret.from_name("aws-creds")])
def run_remote(
    input_s3: str,
    output_s3: str,
    *,
    device: str = "cuda",
    transition_s: float = 0.5,
    stride: int = 1,
    crop_width: int = 1080,
    margin: int = 32,
    smoothing_alpha: float = 0.25,
    max_speed: float = 480.0,
    epsilon: float = 12.0,
    disable_ball: bool = False,
    player_model_s3: str | None = None,
    ball_model_s3: str | None = None,
) -> str:
    """
    Remote GPU entrypoint to generate keyframes and render the final video.
    - Expects AWS credentials via a Modal Secret named `aws-creds`.
    - input_s3/output_s3 like s3://bucket/key.mp4
    """
    import boto3
    import subprocess
    import sys
    from urllib.parse import urlparse

    def s3_download(s3_uri: str, dst_path: str) -> None:
        u = urlparse(s3_uri)
        s3 = boto3.client("s3")
        s3.download_file(u.netloc, u.path.lstrip("/"), dst_path)

    def s3_upload(src_path: str, s3_uri: str) -> None:
        u = urlparse(s3_uri)
        s3 = boto3.client("s3")
        s3.upload_file(src_path, u.netloc, u.path.lstrip("/"))

    repo_root = "/workspace"
    # Mount local repo at runtime for freshest code; expects caller runs `modal run` from repo root
    # If running via schedule/deploy, consider packaging code into the image instead.

    here = os.path.join(repo_root, "examples/soccer")
    gen_py = os.path.join(here, "generate_keyframes.py")
    rend_py = os.path.join(here, "render_from_keyframes.py")

    with tempfile.TemporaryDirectory() as td:
        inp = os.path.join(td, "input.mp4")
        outp = os.path.join(td, "output.mp4")
        keyf = os.path.join(td, "keyframes.json")

        s3_download(input_s3, inp)

        # Optional: download models into expected paths
        data_dir = os.path.join(here, "data")
        os.makedirs(data_dir, exist_ok=True)
        player_dst = os.path.join(data_dir, "football-player-detection.pt")
        ball_dst = os.path.join(data_dir, "football-ball-detection.pt")
        if player_model_s3:
            s3_download(player_model_s3, player_dst)
        if ball_model_s3:
            s3_download(ball_model_s3, ball_dst)

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
            str(transition_s),
        ]
        subprocess.check_call(rend_cmd)

        s3_upload(outp, output_s3)

    return output_s3


@app.local_entrypoint()
def main():
    """
    Example local launch:
    modal run examples/soccer/remote/modal_app.py \
      --input s3://my-bucket/in/clip.mp4 --output s3://my-bucket/out/clip_edit.mp4
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--transition", type=float, default=0.5)
    args = parser.parse_args()

    # Attach the local repo at /workspace so we use your current code.
    # You must run `modal run` from the repository root.
    with modal.Mount.from_local_dir(os.getcwd(), remote_path="/workspace"):
        print(run_remote.call(args.input, args.output, device=args.device, transition_s=args.transition))


