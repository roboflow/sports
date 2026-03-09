# Remote GPU execution (Modal and RunPod)

## Modal quickstart

Prereqs: `pip install modal` and an AWS secret named `aws-creds` in Modal (access key + secret) if you use S3.

Run from repo root so your code is mounted at `/workspace`:

```bash
modal run examples/soccer/remote/modal_app.py \
  --input s3://your-bucket/in/clip.mp4 \
  --output s3://your-bucket/out/clip_slides.mp4
```

Optional flags (pass via the local entrypoint):

- `--device cuda|cpu` (default cuda)
- `--transition 0.5` (seconds)

To specify models from S3, call the function directly in Python:

```python
import modal
from examples.soccer.remote.modal_app import run_remote

with modal.Mount.from_local_dir('.', remote_path='/workspace'):
    print(run_remote.call(
        input_s3='s3://your-bucket/in/clip.mp4',
        output_s3='s3://your-bucket/out/clip_slides.mp4',
        player_model_s3='s3://your-bucket/models/football-player-detection.pt',
        ball_model_s3='s3://your-bucket/models/football-ball-detection.pt',
        device='cuda', transition_s=0.5,
    ))
```

## RunPod serverless

Package a container with Python, ffmpeg, torch, ultralytics, supervision, awscli, curl. Set `entrypoint` to `python -m runpod` and provide this handler:

- File: `examples/soccer/remote/runpod_handler.py`
- Payload:

```json
{
  "input": {
    "input_path": "s3://your-bucket/in/clip.mp4",
    "output_path": "s3://your-bucket/out/clip_slides.mp4",
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
```

The handler will download the input (HTTP/S or S3), run keyframe generation + rendering, and upload the result.




