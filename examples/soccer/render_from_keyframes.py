#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import tempfile
from typing import List, Tuple


def load_keyframes_json(path: str) -> List[Tuple[float, int]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Accept either list of {t, o} dicts or list of [t, o]
    keyframes: List[Tuple[float, int]] = []
    for item in data:
        if isinstance(item, dict) and "t" in item and "o" in item:
            keyframes.append((float(item["t"]), int(item["o"])) )
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            keyframes.append((float(item[0]), int(item[1])))
    # sort by timestamp ascending
    keyframes.sort(key=lambda x: x[0])
    return keyframes


def probe_duration(path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nw=1:nk=1",
        path,
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    try:
        return float(out)
    except Exception:
        raise RuntimeError(f"Could not probe duration for {path!r}, got: {out!r}")


def build_filter_complex(timestamps: List[float], total_duration: float, base_d: float) -> str:
    # Build segments [ti, ti+1], with slide transitions between consecutive segments.
    # Use xfade/acrossfade with clamped durations.
    parts: List[str] = []

    # ensure final boundary reaches end of video
    if not timestamps:
        raise ValueError("No timestamps provided")
    if timestamps[-1] < total_duration - 1e-3:
        timestamps = [*timestamps, total_duration]

    # de-dup and strictly ascending
    asc: List[float] = []
    for t in timestamps:
        if not asc or t > asc[-1] + 1e-6:
            asc.append(t)

    # trims for first pass
    for i in range(len(asc) - 1):
        s = asc[i]
        e = asc[i + 1]
        # skip empty/too short segments
        if e - s <= 0.02:
            continue
        parts.append(f"[0:v]trim=start={s}:end={e},setpts=PTS-STARTPTS[v{i}];")
        parts.append(f"[0:a]atrim=start={s}:end={e},asetpts=PTS-STARTPTS[a{i}];")

    # chain transitions
    prev_v = None
    prev_a = None
    prev_len = 0.0
    cum_out = 0.0

    seg_index = 0
    for i in range(len(asc) - 1):
        s = asc[i]
        e = asc[i + 1]
        seg_d = e - s
        if seg_d <= 0.02:
            continue

        if prev_v is None:
            prev_v = f"v{i}"
            prev_a = f"a{i}"
            prev_len = seg_d
            cum_out = seg_d
            continue

        # clamp transition length to fit within both sides (<= 45% of each)
        d_i = max(0.05, min(base_d, 0.45 * prev_len, 0.45 * seg_d))
        offset = max(0.0, cum_out - d_i)

        parts.append(
            f"[{prev_v}][v{i}]xfade=transition=slideleft:duration={d_i}:offset={offset}[vx{i}];"
        )
        parts.append(f"[{prev_a}][a{i}]acrossfade=d={d_i}:curve1=tri:curve2=tri[ax{i}];")

        prev_v = f"vx{i}"
        prev_a = f"ax{i}"
        cum_out = cum_out + seg_d - d_i
        prev_len = seg_d
        seg_index += 1

    if prev_v is None or prev_a is None:
        raise ValueError("Not enough non-empty segments after timestamps")

    parts.append(f"[{prev_v}][{prev_a}]")  # sentinel to simplify return
    return "".join(parts)


def render(
    input_video: str,
    keyframes_json: str,
    output_path: str,
    base_transition: float = 0.5,
    vcodec: str = "libx264",
    acodec: str = "aac",
    crf: int = 20,
    preset: str = "veryfast",
) -> None:
    keyframes = load_keyframes_json(keyframes_json)
    if not keyframes:
        raise SystemExit("No keyframes found in JSON")
    timestamps = [t for t, _ in keyframes]
    duration = probe_duration(input_video)
    filter_complex = build_filter_complex(timestamps, duration, base_transition)

    # Extract final map labels from the sentinel
    # The last appended part is like: "[vxN][axN]"; we can compute them too, but parse here for simplicity
    last = filter_complex.rsplit("[", 1)[-1]
    if "]" in last and "][" in filter_complex:
        # safer: recompute last labels directly
        # Find last vx/ax index by scanning backwards
        # But we already have prev_v/prev_a naming; rebuild quickly instead
        pass

    # Build command and run
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        input_video,
        "-filter_complex",
        filter_complex,
        "-map",
        "[vx{}]".format(len(timestamps)),  # placeholder, will be replaced below
        "-map",
        "[ax{}]".format(len(timestamps)),
        "-c:v",
        vcodec,
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-c:a",
        acodec,
        "-movflags",
        "+faststart",
        output_path,
    ]

    # Determine final labels by scanning for the last transition we created
    # The last transition indices correspond to the last valid i used in xfade
    # We approximate by searching highest vx/ax occurrence
    import re

    vx_matches = [int(m) for m in re.findall(r"\[vx(\d+)\]", filter_complex)]
    ax_matches = [int(m) for m in re.findall(r"\[ax(\d+)\]", filter_complex)]
    if vx_matches:
        vx_last = max(vx_matches)
        ax_last = max(ax_matches) if ax_matches else vx_last
        # replace placeholder maps
        cmd[cmd.index("-map") + 1] = f"[vx{vx_last}]"
        idx2 = cmd.index("-map", cmd.index("-map") + 1)
        cmd[idx2 + 1] = f"[ax{ax_last}]"
    else:
        # No transitions: only a single segment v0/a0
        cmd[cmd.index("-map") + 1] = "[v0]"
        idx2 = cmd.index("-map", cmd.index("-map") + 1)
        cmd[idx2 + 1] = "[a0]"

    subprocess.check_call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render slide-transitions from keyframes JSON")
    parser.add_argument("--input_video", required=True, help="Source video path (e.g. clip.mp4)")
    parser.add_argument("--keyframes_json", required=True, help="JSON from generate_keyframes.py")
    parser.add_argument("--output", required=True, help="Output mp4 path")
    parser.add_argument("--transition", type=float, default=0.5, help="Base transition duration (s)")
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--preset", default="veryfast")
    args = parser.parse_args()

    render(
        input_video=args.input_video,
        keyframes_json=args.keyframes_json,
        output_path=args.output,
        base_transition=args.transition,
        crf=args.crf,
        preset=args.preset,
    )


if __name__ == "__main__":
    main()




