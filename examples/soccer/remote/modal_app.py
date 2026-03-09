"""Modal HTTP endpoint — auto-crop keyframe generation using SAM 3.

POST /crop
  Body: { videoUrl, format, videoWidth, videoHeight, sport? }
  Returns: [{ t: float, o: int }, ...]

Set window.__AUTO_CROP_URL__ = "<deployed url>/crop" in the browser to wire
the LIGR VideoFormatEditor AUTO CROP button to this endpoint.
"""
import re

import modal

app = modal.App("auto-crop")

# Volume caches SAM 3 weights across cold starts (~2 GB download avoided).
model_volume = modal.Volume.from_name("sam3-weights", create_if_missing=True)
MODEL_DIR = "/models"

image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "ultralytics>=8.3.237",
        "supervision>=0.21.0",
        "opencv-python-headless",
        "numpy",
        "requests",
        "fastapi[standard]",
        "openai-clip",
    )
    .pip_install(
        "torch==2.4.1+cu121",
        "torchvision==0.19.1+cu121",
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )
    .add_local_python_source("sports")
)

SPORTS_MAP = {
    "football": "football",
    "soccer": "football",
    "tennis": "tennis",
}


_FORMAT_RATIOS = {
    # LIGR FormatLabels enum values
    "vertical": (9, 16),
    "square": (1, 1),
    "standard": (4, 3),
    "portrait": (4, 5),
}


def _parse_crop_width(format_str: str, video_height: int) -> int:
    """Compute crop width in pixels from format string and source video height.

    Accepts either LIGR FormatLabels enum values ("vertical", "square", …) or
    explicit W:H strings ("9:16", "1:1", …).
    """
    if format_str in _FORMAT_RATIOS:
        w, h = _FORMAT_RATIOS[format_str]
        return max(1, round(video_height * w / h))
    m = re.search(r"(\d+)\s*:\s*(\d+)", format_str)
    if m:
        w, h = int(m.group(1)), int(m.group(2))
        return max(1, round(video_height * w / h))
    return video_height  # fallback: 1:1


@app.function(
    image=image,
    gpu="T4",
    volumes={MODEL_DIR: model_volume},
    timeout=600,
    # Keep one warm container so the team doesn't wait for cold starts.
    min_containers=1,
)
@modal.asgi_app()
def endpoint():
    import os
    import tempfile

    import requests as http
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware

    from sports.pipelines import FOOTBALL, TENNIS, KeyframeGenerator

    api = FastAPI(title="Auto-Crop")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["POST", "GET", "OPTIONS"],
        allow_headers=["*"],
    )

    SPORTS = {"football": FOOTBALL, "tennis": TENNIS}

    @api.get("/health")
    def health():
        return {"status": "ok"}

    @api.post("/crop")
    async def crop(
        videoFile: UploadFile = File(...),
        format: str = Form("9:16"),
        videoHeight: int = Form(1080),
        sport: str = Form("football"),
    ):
        try:
            import shutil
            from ultralytics.utils.downloads import attempt_download_asset

            sport_key = SPORTS_MAP.get(sport.lower(), "football")
            sport_config = SPORTS[sport_key]
            crop_width_px = _parse_crop_width(format, videoHeight)

            print(f"[auto-crop] sport={sport_key} format={format!r} "
                  f"crop_width={crop_width_px}px filename={videoFile.filename}")

            model_path = os.path.join(MODEL_DIR, "sam3_b.pt")

            if not os.path.exists(model_path):
                print("[auto-crop] model not found, downloading...")
                tmp = attempt_download_asset("sam3_b.pt")
                shutil.copy(tmp, model_path)
                model_volume.commit()
                print(f"[auto-crop] model saved to {model_path}")

            with tempfile.TemporaryDirectory() as td:
                video_path = os.path.join(td, "input.mp4")
                with open(video_path, "wb") as f:
                    while chunk := await videoFile.read(1 << 20):
                        f.write(chunk)

                gen = KeyframeGenerator(
                    sport=sport_config,
                    model_path=model_path,
                    device="cuda",
                    crop_width_px=crop_width_px,
                )
                keyframes = gen.generate(video_path)

            print(f"[auto-crop] generated {len(keyframes)} keyframes")
            model_volume.commit()
            return [kf.as_dict() for kf in keyframes]
        except Exception as exc:
            raise HTTPException(500, str(exc)) from exc

    return api
