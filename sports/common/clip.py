from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import supervision as sv

from sports.common.cache import FrameCache, build_or_load_detections
from sports.common.tracking import (
    collect_referee_tracker_ids,
    collect_team_frames,
    create_player_detector,
    create_player_tracker,
    drop_blocked_tracker_ids,
    fit_team_classifier,
    open_video,
    resolve_goalkeepers_team_id,
)
from sports.common.team import (
    apply_team_lock,
    lock_teams_by_tracklet_majority,
    relock_detection_teams,
)
from sports.configs.soccer import GOALKEEPER_CLASS_ID, PLAYER_CLASS_ID, TEAM_NONE

DEFAULT_PLAYER_MODEL_ID = "football-players-detection-3zvbc/11"


@dataclass
class ClipLocks:
    """Clip-level team lock for direction mode."""

    team_lock: dict
    gk_lock: dict = field(default_factory=dict)
    locked_goal_defenders: tuple = None


@dataclass
class ClipAnalysis:
    """Shared groundwork for direction-mode renders."""

    args: Any
    source_video_path: str
    fps: float
    width: int
    height: int
    max_frames: Any
    tracker_kind: str
    needs_frame: bool

    det_by_frame: dict
    team_classifier: Any

    frames: list
    referee_frames: list
    blocked_ids: frozenset

    _locks: dict = field(default_factory=dict, repr=False)

    def locks(self) -> ClipLocks:
        """Return clip-level team locks, cached."""
        if "centroid" not in self._locks:
            cloned = _clone_team_frames(self.frames)
            team_lock = derive_team_lock(cloned)
            self._locks["centroid"] = ClipLocks(team_lock=team_lock)
        return self._locks["centroid"]

    def iter_tracked(self):
        """Yield frame_idx and tracked detections with blocked ids removed."""
        for frame_idx, dets in self.frames:
            yield frame_idx, drop_blocked_tracker_ids(dets, self.blocked_ids)

    def tracked_by_frame(self) -> dict:
        """Return frame-indexed tracked detections."""
        return {fi: dets for fi, dets in self.iter_tracked()}

    def decorate_replay_frame(
        self,
        frame_idx: int,
        tracked: sv.Detections,
        locks: ClipLocks,
        vel_smoother,
    ) -> sv.Detections:
        """Apply team lock and centroid goalkeeper assignment to a replay frame."""
        team_arr = (
            np.array(tracked.data.get("team"), dtype=int)
            if tracked.data and tracked.data.get("team") is not None
            else np.full(len(tracked), TEAM_NONE, dtype=int)
        )
        team_arr = apply_team_lock(team_arr, tracked.tracker_id, locks.team_lock)
        if len(tracked):
            gk_mask = tracked.class_id == GOALKEEPER_CLASS_ID
            pl_mask = tracked.class_id == PLAYER_CLASS_ID
            if gk_mask.any() and (team_arr == 0).any() and (team_arr == 1).any():
                gk_teams = resolve_goalkeepers_team_id(
                    tracked[pl_mask], team_arr[pl_mask], tracked[gk_mask]
                )
                team_arr[gk_mask] = gk_teams
        data = dict(tracked.data) if tracked.data else {}
        data["team"] = team_arr
        decorated = sv.Detections(
            xyxy=tracked.xyxy,
            class_id=tracked.class_id,
            tracker_id=tracked.tracker_id,
            confidence=tracked.confidence,
            data=data,
        )
        decorated = vel_smoother.smooth_detections(decorated)
        decorated = relock_detection_teams(decorated, locks.team_lock)
        return decorated


def _fill_goalkeeper_teams_centroid(frames) -> None:
    """Fill goalkeeper team ids per frame using the centroid rule."""
    for _, dets in frames:
        if dets.data is None or len(dets) == 0 or dets.tracker_id is None:
            continue
        team = np.asarray(dets.data.get("team", np.full(len(dets), TEAM_NONE)), dtype=int)
        gk_mask = dets.class_id == GOALKEEPER_CLASS_ID
        pl_mask = dets.class_id == PLAYER_CLASS_ID
        if not gk_mask.any():
            continue
        if not ((team[pl_mask] == 0).any() and (team[pl_mask] == 1).any()):
            continue
        gk_teams = resolve_goalkeepers_team_id(dets[pl_mask], team[pl_mask], dets[gk_mask])
        team[gk_mask] = gk_teams
        dets.data["team"] = team


def derive_team_lock(frames) -> dict:
    """Derive clip-level team lock with centroid goalkeeper fill."""
    _fill_goalkeeper_teams_centroid(frames)
    return lock_teams_by_tracklet_majority(frames)


def _clone_team_frames(frames):
    cloned = []
    for frame_idx, dets in frames:
        data = dict(dets.data) if dets.data else {}
        if "team" in data:
            data["team"] = np.array(data["team"], dtype=int)
        cloned.append(
            (
                frame_idx,
                sv.Detections(
                    xyxy=dets.xyxy,
                    class_id=dets.class_id,
                    tracker_id=dets.tracker_id,
                    confidence=dets.confidence,
                    data=data,
                ),
            )
        )
    return cloned


def _make_player_detector_factory(args) -> Callable:
    player_model_id = getattr(args, "player_model_id", DEFAULT_PLAYER_MODEL_ID)

    def _factory():
        return create_player_detector(
            backend=args.player_detector,
            model_path=getattr(args, "player_model_path", None),
            model_id=player_model_id,
            device=args.device,
            api_key=getattr(args, "api_key", None),
        )

    return _factory


def compute_clip_analysis(args) -> ClipAnalysis:
    """Run shared direction groundwork once and return ClipAnalysis."""
    _, fps, width, height = open_video(args.source_video_path)
    max_frames = getattr(args, "max_frames", None)
    tracker_kind = getattr(args, "tracker", "botsort")
    needs_frame = tracker_kind == "botsort"

    player_model_id = getattr(args, "player_model_id", DEFAULT_PLAYER_MODEL_ID)
    cache = FrameCache(
        args.source_video_path,
        cache_dir=getattr(args, "cache_dir", None),
        enabled=getattr(args, "cache", True),
        player_backend=args.player_detector,
        player_model_id=player_model_id,
    )
    det_by_frame = build_or_load_detections(
        args.source_video_path,
        _make_player_detector_factory(args),
        cache,
        max_frames=max_frames,
    )

    print("Fitting team classifier...")
    cap, _, _, _ = open_video(args.source_video_path)
    try:
        team_classifier = fit_team_classifier(
            cap,
            device=args.device,
            max_frames=max_frames,
            det_by_frame=det_by_frame,
        )
    finally:
        cap.release()

    print("Tracking once (shared BoTSORT pass)...")
    frames, referee_frames = collect_team_frames(
        args.source_video_path,
        team_classifier=team_classifier,
        tracker=create_player_tracker(fps, kind=tracker_kind),
        needs_frame=needs_frame,
        max_frames=max_frames,
        detections_by_frame=det_by_frame,
        capture_velocity=True,
    )
    blocked_ids = collect_referee_tracker_ids(referee_frames)

    return ClipAnalysis(
        args=args,
        source_video_path=args.source_video_path,
        fps=fps,
        width=width,
        height=height,
        max_frames=max_frames,
        tracker_kind=tracker_kind,
        needs_frame=needs_frame,
        det_by_frame=det_by_frame,
        team_classifier=team_classifier,
        frames=frames,
        referee_frames=referee_frames,
        blocked_ids=blocked_ids,
    )
