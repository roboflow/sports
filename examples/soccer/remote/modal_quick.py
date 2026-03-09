from __future__ import annotations

import os
import tempfile

import modal


app = modal.App("sports-soccer-quick")

image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg", "curl", "git")
    .pip_install(
        # Core CV stack (from PyPI)
        "ultralytics>=8.0.0",
        "supervision>=0.21.0",
        "opencv-python-headless",
        "numpy",
        "transformers",
        "umap-learn",
        "scikit-learn",
        "tqdm",
        "sentencepiece",
        "protobuf",
    )
    .pip_install(
        # Torch GPU wheels for CUDA 12.1 (works on T4)
        "torch==2.4.1+cu121",
        "torchvision==0.19.1+cu121",
        "torchaudio==2.4.1",
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )
    # Include the local `sports` package in the container's PYTHONPATH (/root)
    .add_local_python_source("sports")
)


@app.function(
    image=image,
    gpu="T4",
)
def quick_run(input_bytes: bytes, device: str = "cuda", transition_s: float = 0.5) -> bytes:
    """Return the edited mp4 bytes for a small clip.

    This avoids any external storage; just uploads from local and downloads result.
    """
    import subprocess
    import sys

    with tempfile.TemporaryDirectory() as td:
        inp = os.path.join(td, "input.mp4")
        outp = os.path.join(td, "output.mp4")

        with open(inp, "wb") as f:
            f.write(input_bytes)

        # Import after ensuring local package is available in the image
        from sports.pipelines import KeyframeGenerator
        import supervision as sv

        # Fallback to CPU if CUDA is requested but unavailable
        try:
            import torch  # noqa: F401
            if device == "cuda":
                import torch as _torch
                if not _torch.cuda.is_available():
                    device = "cpu"
        except Exception:
            pass

        # Determine video dimensions and pick square crop width targeting 1080x1080 output
        video_info = sv.VideoInfo.from_video_path(inp)
        crop_w = min(1080, video_info.width)  # horizontal crop size in source pixels
        crop_h = min(1080, video_info.height)  # vertical crop size in source pixels (square when height>=1080)
        y_off = max(0, int(round((video_info.height - crop_h) / 2)))  # center vertically if needed

        generator = KeyframeGenerator(
            player_model_path="yolov8n.pt",  # Ultralytics auto-downloads
            device=device,
            stride=1,
            crop_width_px=int(crop_w),
            margin_px=48,
            smoothing_alpha=0.12,
            max_speed_px_per_s=240.0,
            compression_epsilon_px=36.0,
            ball_model_path=None,
        )
        keyframes = generator.generate(inp)

        # Build timestamps list and aligned offsets for square crop
        timestamps = []
        offsets = []
        for kf in keyframes:
            t = getattr(kf, "timestamp_s", None)
            o = getattr(kf, "offset_px", None)
            if t is None and hasattr(kf, "as_pair"):
                t, o = kf.as_pair()
            if t is not None and o is not None:
                timestamps.append(float(t))
                offsets.append(int(o))
        # Deduplicate while preserving order
        uniq_ts = []
        uniq_of = []
        for i, t in enumerate(timestamps):
            if not uniq_ts or t != uniq_ts[-1]:
                uniq_ts.append(t)
                uniq_of.append(offsets[i])
        timestamps, offsets = uniq_ts, uniq_of

        # Additional stabilization: deadband + minimum segment duration
        deadband_px = 24  # ignore small changes under this threshold
        min_segment_s = 0.50  # avoid quick back-and-forth shorter than this

        if len(timestamps) >= 2:
            t_s, o_s = [timestamps[0]], [offsets[0]]
            for i in range(1, len(timestamps)):
                if abs(offsets[i] - o_s[-1]) < deadband_px:
                    continue
                t_s.append(timestamps[i])
                o_s.append(offsets[i])

            # Enforce minimum duration between changes by removing short blips
            j = 1
            while j < len(t_s):
                if t_s[j] - t_s[j - 1] < min_segment_s:
                    # Drop this change; extend previous offset
                    t_s.pop(j)
                    o_s.pop(j)
                    continue
                j += 1

            timestamps, offsets = t_s, o_s
        if not timestamps:
            # Fallback: single-pass copy if no keyframes found
            subprocess.check_call(["ffmpeg", "-hide_banner", "-y", "-i", inp, "-c", "copy", outp])
            with open(outp, "rb") as f:
                return f.read()

        # Probe duration
        dur = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", inp
        ], text=True).strip()
        try:
            total_duration = float(dur)
        except Exception:
            total_duration = timestamps[-1] if timestamps else 0.0
        if not timestamps or timestamps[-1] < total_duration - 1e-3:
            timestamps.append(total_duration)
            # Repeat last offset for terminal segment
            if offsets:
                offsets.append(offsets[-1])

        # Build a single dynamic-crop expression: x(t) interpolates between keyframes
        # x(t) = o_i + (o_{i+1}-o_i) * (t - t_i) / (t_{i+1}-t_i) for t in [t_i, t_{i+1}]
        seg_exprs = []
        for i in range(len(timestamps) - 1):
            s = float(timestamps[i])
            e = float(timestamps[i + 1])
            dt = e - s
            if dt <= 0.02:
                continue
            o0 = int(offsets[i]) if i < len(offsets) else 0
            o1 = int(offsets[i + 1]) if (i + 1) < len(offsets) else o0
            slope = (o1 - o0) / dt if dt > 0 else 0.0
            val = f"{o0}+{slope:.10f}*(t-{s:.10f})"
            cond = f"between(t,{s:.10f},{e:.10f})"
            seg_exprs.append((cond, val))

        # Default to last offset when outside ranges
        default_val = str(int(offsets[-1] if offsets else 0))
        piecewise = default_val
        for cond, val in reversed(seg_exprs):
            piecewise = f"if({cond},{val},{piecewise})"
        max_off = max(0, int(video_info.width - crop_w))
        x_expr = f"max(0,min({max_off},{piecewise}))"

        # Single-pass filter: dynamic crop + scale
        import subprocess as _sp
        # Quote x expression to avoid commas being parsed as filter separators
        vf = f"crop=w={int(crop_w)}:h={int(crop_h)}:x='{x_expr}':y={int(y_off)},scale=1080:1080"
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", inp,
            "-vf", vf,
            "-map", "0:v:0",
            "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-movflags", "+faststart",
            outp,
        ]
        try:
            res = _sp.run(cmd, check=True, stdout=_sp.PIPE, stderr=_sp.PIPE, text=True)
            if res.stdout:
                print(res.stdout)
            if res.stderr:
                print(res.stderr)
        except _sp.CalledProcessError as e:
            print("ffmpeg command:", " ".join(cmd))
            print("ffmpeg stderr:\n", e.stderr)
            raise

        with open(outp, "rb") as f:
            return f.read()


@app.local_entrypoint()
def main(path: str, out: str = "./clip_slides.mp4", device: str = "cuda"):
    with open(path, "rb") as f:
        data = f.read()
    result = quick_run.remote(data, device=device)
    with open(out, "wb") as f:
        f.write(result)
    print(f"Wrote {out}")
