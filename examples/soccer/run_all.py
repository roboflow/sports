"""In-process orchestrator for all five analytics renders.

Builds a shared :class:`~sports.common.video_tracking.VideoTrackingSession` once
(one BoTSORT pass, one team-classifier fit, homography maps, kinematics) and drives
each mode renderer in-process. Each mode writes its own output video.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

from sports.common.video_tracking import build_video_tracking_session
from direction import run_direction
from distance import run_distance
from speed import run_speed
from speed_and_distance import run_speed_and_distance

_RENDER_PLAN = (
    ("direction", "DIRECTION"),
    ("speed", "SPEED"),
    ("distance", "DISTANCE"),
    ("speed-distance-all", "SPEED_AND_DISTANCE (all players)"),
    ("speed-distance-single", "SPEED_AND_DISTANCE (spotlight)"),
)


def _derive_target(base_path: str, suffix: str) -> str:
    """``data/renders/08fd33_0.mp4`` + ``direction`` → ``…/08fd33_0-direction.mp4``."""
    p = Path(base_path)
    return str(p.with_name(f"{p.stem}-{suffix}{p.suffix or '.mp4'}"))


def _mode_args(args, *, target: str, track_id: int | None = None):
    """Shallow copy of ``args`` with a per-mode target path and spotlight track id."""
    new = copy.copy(args)
    new.target_video_path = target
    new.track_id = track_id
    return new


def _pick_spotlight_track_id(tracks) -> int | None:
    """Pick the most-active tracked player (max cumulative distance) for the spotlight."""
    best_tid: int | None = None
    best_distance = -1.0
    for tid, track in tracks.items():
        distance = float(getattr(track, "distance_m", 0.0) or 0.0)
        if distance > best_distance:
            best_distance = distance
            best_tid = int(tid)
    return best_tid


def run_all(args) -> list[str]:
    """Build shared session once and render all five analytics videos in-process."""
    t_start = time.time()

    print("── run-all: building shared VideoTrackingSession ─────────")
    session = build_video_tracking_session(args, need_homography=True)
    print(f"Shared session ready in {time.time() - t_start:.1f}s.")

    base = args.target_video_path
    explicit_spotlight = getattr(args, "track_id", None)
    spotlight_id = (
        explicit_spotlight
        if explicit_spotlight is not None
        else _pick_spotlight_track_id(session.tracks)
    )
    print(f"Spotlight (SPEED_AND_DISTANCE) track id: {spotlight_id}")

    outputs: list[str] = []
    timings: list[tuple[str, float]] = []
    for suffix, label in _RENDER_PLAN:
        target = _derive_target(base, suffix)
        track_id = spotlight_id if suffix == "speed-distance-single" else None
        t_mode = time.time()
        print(f"── run-all: rendering {label} → {target} ─────────")
        if suffix == "direction":
            run_direction(_mode_args(args, target=target), session)
        elif suffix == "speed":
            run_speed(_mode_args(args, target=target), session)
        elif suffix == "distance":
            run_distance(_mode_args(args, target=target), session)
        else:
            run_speed_and_distance(
                _mode_args(args, target=target, track_id=track_id), session,
            )
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
