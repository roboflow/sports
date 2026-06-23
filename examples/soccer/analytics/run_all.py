"""analytics/run_all.py — in-process orchestrator for all five analytics renders.

Computes the shared :class:`~analytics.clip_pipeline.ClipAnalysis` ONCE (one BoTSORT pass,
one team-classifier fit, one set of homography maps, one kinematics integration) and then
drives each of the five mode renderers in-process, handing them that single analysis. Each
mode still writes its own output video; only the expensive shared groundwork is reused.

The five renders are DIRECTION, SPEED, DISTANCE, PLAYER_FOCUS (follow-all) and PLAYER_FOCUS
(single-player spotlight). BoTSORT runs once for the whole run-all; each renderer reuses
that shared analysis.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

from analytics.clip_pipeline import ClipAnalysis, compute_clip_analysis

# (output suffix, human label) for the five renders, in run order.
_RENDER_PLAN = (
    ("direction", "DIRECTION"),
    ("speed", "SPEED"),
    ("distance", "DISTANCE"),
    ("focus-all", "PLAYER_FOCUS (follow-all)"),
    ("focus-single", "PLAYER_FOCUS (spotlight)"),
)


def _derive_target(base_path: str, suffix: str) -> str:
    """``data/renders/08fd33_0.mp4`` + ``direction`` → ``…/08fd33_0-direction.mp4``."""
    p = Path(base_path)
    return str(p.with_name(f"{p.stem}-{suffix}{p.suffix or '.mp4'}"))


def _mode_args(args, *, target: str, track_id: int | None = None):
    """Shallow copy of ``args`` with a per-mode target path and focus track id."""
    new = copy.copy(args)
    new.target_video_path = target
    new.track_id = track_id
    return new


def _pick_focus_track_id(analysis: ClipAnalysis) -> int | None:
    """Pick the most-active tracked player (max cumulative distance) for the spotlight.

    Deterministic from the shared kinematics; an explicit ``--track-id`` overrides it.
    """
    best_tid: int | None = None
    best_distance = -1.0
    for tid, track in analysis.tracks.items():
        distance = float(getattr(track, "distance_m", 0.0) or 0.0)
        if distance > best_distance:
            best_distance = distance
            best_tid = int(tid)
    return best_tid


def run_all(args) -> list[str]:
    """Compute the shared analysis once and render all five analytics videos in-process."""
    from analytics.direction import run_direction
    from analytics.distance import run_distance
    from analytics.player_focus import run_player_focus
    from analytics.speed import run_speed

    t_start = time.time()

    print("── run-all: computing shared ClipAnalysis ─────────")
    analysis = compute_clip_analysis(args, need_homography=True)
    print(f"Shared analysis ready in {time.time() - t_start:.1f}s.")

    base = args.target_video_path
    explicit_focus = getattr(args, "track_id", None)
    focus_id = explicit_focus if explicit_focus is not None else _pick_focus_track_id(analysis)
    print(f"Spotlight (PLAYER_FOCUS single) track id: {focus_id}")

    outputs: list[str] = []
    timings: list[tuple[str, float]] = []
    for suffix, label in _RENDER_PLAN:
        target = _derive_target(base, suffix)
        track_id = focus_id if suffix == "focus-single" else None
        t_mode = time.time()
        print(f"── run-all: rendering {label} → {target} ─────────")
        if suffix == "direction":
            run_direction(_mode_args(args, target=target), analysis)
        elif suffix == "speed":
            run_speed(_mode_args(args, target=target), analysis)
        elif suffix == "distance":
            run_distance(_mode_args(args, target=target), analysis)
        else:  # focus-all / focus-single
            run_player_focus(_mode_args(args, target=target, track_id=track_id), analysis)
        outputs.append(target)
        timings.append((label, time.time() - t_mode))

    total_time = time.time() - t_start
    print("── run-all: done ─────────────────────────────────────────────────────")
    for label, dt in timings:
        print(f"  {label:<28} {dt:6.1f}s")
    print(f"  {'TOTAL':<28} {total_time:6.1f}s")
    print(f"  All {len(outputs)} renders used a single shared tracking pass.")
    print("Outputs:")
    for path in outputs:
        print(f"  {path}")
    return outputs
