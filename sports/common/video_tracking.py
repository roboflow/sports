from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import supervision as sv

from sports.common.cache import (
    FrameCache,
    build_or_load_detections,
    build_or_load_keypoints,
)
from sports.common.detection import (
    DEFAULT_PITCH_MODEL_ID,
    create_pitch_keypoint_detector,
)
from sports.common.homography import (
    build_minimap_transform_map,
    gap_fill_speed_transforms,
    replay_gated_transforms,
)
from sports.common.kinematics import (
    PlayerTrack,
    collect_tracks,
    compute_kinematics,
)
from sports.common.goalkeeper import apply_goalkeeper_teams, derive_clip_locks
from sports.common.team import (
    TeamLocks,
    apply_team_lock,
    clone_team_frames,
    relock_detection_teams,
)
from sports.common.tracking import (
    DEFAULT_PLAYER_MODEL_ID,
    collect_referee_tracker_ids,
    collect_team_frames,
    create_player_detector,
    create_player_tracker,
    drop_blocked_tracker_ids,
    fit_team_classifier,
    open_video,
)
from sports.configs.soccer import TEAM_NONE


@dataclass
class VideoTrackingSession:
    """Shared tracking groundwork for analytics-mode renders."""

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

    kp_by_frame: dict | None = None

    _team_lock: TeamLocks | None = field(default=None, repr=False)
    _speed_transforms: dict | None = field(default=None, repr=False)
    _gap_filled_transforms: dict | None = field(default=None, repr=False)
    _minimap_transforms: dict | None = field(default=None, repr=False)
    _tracks: dict[int, PlayerTrack] | None = field(default=None, repr=False)

    def team_locks(self) -> TeamLocks:
        """Return clip-level team locks, cached."""
        if self._team_lock is None:
            cloned = clone_team_frames(self.frames)
            minimap = (
                self.minimap_transforms_by_frame
                if self.kp_by_frame is not None
                else None
            )
            self._team_lock = derive_clip_locks(
                cloned, minimap_transforms=minimap,
            )
        return self._team_lock

    def _gated_speed_transforms(self) -> dict:
        if self._speed_transforms is None:
            if not self.kp_by_frame:
                raise RuntimeError("pitch keypoints were not computed for this session")
            self._speed_transforms = replay_gated_transforms(self.kp_by_frame)
        return self._speed_transforms

    @property
    def gap_filled_transforms_by_frame(self) -> dict:
        """Speed homography with ungated fallback for display badges."""
        if self._gap_filled_transforms is None:
            if not self.kp_by_frame:
                raise RuntimeError("pitch keypoints were not computed for this session")
            self._gap_filled_transforms = gap_fill_speed_transforms(
                self._gated_speed_transforms(),
                self.minimap_transforms_by_frame,
            )
        return self._gap_filled_transforms

    @property
    def minimap_transforms_by_frame(self) -> dict:
        """Ungated homography per frame for the visible minimap."""
        if self._minimap_transforms is None:
            if not self.kp_by_frame:
                raise RuntimeError("pitch keypoints were not computed for this session")
            self._minimap_transforms = build_minimap_transform_map(self.kp_by_frame)
        return self._minimap_transforms

    @property
    def tracks(self) -> dict[int, PlayerTrack]:
        """Per-track cumulative distance from gated homography (not gap-filled)."""
        if self._tracks is None:
            raw = collect_tracks(self.iter_tracked())
            self._tracks = compute_kinematics(
                raw,
                self.fps,
                mode="homography",
                frame_transforms=self._gated_speed_transforms(),
            )
        return self._tracks

    def iter_tracked(self):
        """Yield frame_idx and tracked detections with blocked ids removed."""
        for frame_idx, dets in self.frames:
            yield frame_idx, drop_blocked_tracker_ids(dets, self.blocked_ids)

    def tracked_by_frame(self) -> dict:
        """Return frame-indexed tracked detections."""
        return {fi: dets for fi, dets in self.iter_tracked()}

    def apply_replay_teams(
        self,
        tracked: sv.Detections,
        *,
        locks: TeamLocks,
        vel_smoother,
        frame_idx: int,
    ) -> sv.Detections:
        """Apply team lock and goalkeeper assignment to a replay frame."""
        team_arr = (
            np.array(tracked.data.get("team"), dtype=int)
            if tracked.data and tracked.data.get("team") is not None
            else np.full(len(tracked), TEAM_NONE, dtype=int)
        )
        team_arr = apply_team_lock(team_arr, tracked.tracker_id, locks.team_lock)
        data = dict(tracked.data) if tracked.data else {}
        data["team"] = team_arr
        decorated = sv.Detections(
            xyxy=tracked.xyxy,
            class_id=tracked.class_id,
            tracker_id=tracked.tracker_id,
            confidence=tracked.confidence,
            data=data,
        )
        radar_h = (
            self.minimap_transforms_by_frame.get(frame_idx)
            if self.kp_by_frame is not None
            else None
        )
        decorated = apply_goalkeeper_teams(
            decorated, transformer=radar_h, locks=locks,
        )
        decorated = vel_smoother.smooth_detections(decorated)
        decorated = relock_detection_teams(decorated, locks.team_lock)
        return decorated


def _create_player_detector_factory(args) -> Callable:
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


def _create_pitch_detector_factory(args) -> Callable:
    pitch_model_id = getattr(args, "pitch_model_id", DEFAULT_PITCH_MODEL_ID)

    def _factory():
        return create_pitch_keypoint_detector(
            backend=args.pitch_detector,
            model_path=getattr(args, "pitch_model_path", None),
            model_id=pitch_model_id,
            device=args.device,
            api_key=getattr(args, "api_key", None),
        )

    return _factory


def build_video_tracking_session(
    args, *, need_homography: bool = False
) -> VideoTrackingSession:
    """Run shared analytics groundwork once and return a VideoTrackingSession."""
    _, fps, width, height = open_video(args.source_video_path)
    max_frames = getattr(args, "max_frames", None)
    tracker_kind = getattr(args, "tracker", "botsort")
    needs_frame = tracker_kind == "botsort"

    player_model_id = getattr(args, "player_model_id", DEFAULT_PLAYER_MODEL_ID)
    pitch_model_id = getattr(args, "pitch_model_id", DEFAULT_PITCH_MODEL_ID)
    cache = FrameCache(
        args.source_video_path,
        cache_dir=getattr(args, "cache_dir", None),
        enabled=getattr(args, "cache", True),
        player_backend=args.player_detector,
        player_model_id=player_model_id,
        pitch_backend=getattr(args, "pitch_detector", "yolo"),
        pitch_model_id=pitch_model_id,
    )
    det_by_frame = build_or_load_detections(
        args.source_video_path,
        _create_player_detector_factory(args),
        cache,
        max_frames=max_frames,
    )

    kp_by_frame = None
    if need_homography:
        kp_by_frame = build_or_load_keypoints(
            args.source_video_path,
            _create_pitch_detector_factory(args),
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

    return VideoTrackingSession(
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
        kp_by_frame=kp_by_frame,
    )
