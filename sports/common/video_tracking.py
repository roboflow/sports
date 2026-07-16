from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import supervision as sv

from sports.common.cache import (
    FrameCache,
    build_or_load_ball_detections,
    build_or_load_detections,
    build_or_load_keypoints,
)
from sports.common.ball import attach_ball, create_ball_detector
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
    KalmanVelocitySmoother,
    PlayerTrack,
    collect_tracks,
    compute_kinematics,
)
from sports.common.passes import (
    PassDetectionConfig,
    PossessionScanResult,
    scan_possession_events,
)
from sports.common.pass_alternatives import PassEvent, plan_pass_events
from sports.common.pass_options import PassQualityScorer, PassWeights
from sports.common.goalkeeper import apply_goalkeeper_teams, derive_gk_locks
from sports.common.team import (
    TeamLocks,
    apply_team_lock,
    clone_team_frames,
    lock_teams_by_tracklet_majority,
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

    device: str | None = field(default=None, repr=False)
    ball_model_path: str | None = field(default=None, repr=False)
    _cache: FrameCache | None = field(default=None, repr=False)
    _ball_by_frame: dict | None = field(default=None, repr=False)
    _pass_frames: list | None = field(default=None, repr=False)
    _pass_scan: PossessionScanResult | None = field(default=None, repr=False)
    _pass_scorer: PassQualityScorer | None = field(default=None, repr=False)
    _pass_alternative_events: list[PassEvent] | None = field(default=None, repr=False)
    _team_lock: TeamLocks | None = field(default=None, repr=False)
    _speed_transforms: dict | None = field(default=None, repr=False)
    _gap_filled_transforms: dict | None = field(default=None, repr=False)
    _minimap_transforms: dict | None = field(default=None, repr=False)
    _tracks: dict[int, PlayerTrack] | None = field(default=None, repr=False)

    def team_locks(self) -> TeamLocks:
        """Return clip-level team locks, cached."""
        if self._team_lock is None:
            outfield = clone_team_frames(self.frames)
            team_lock = lock_teams_by_tracklet_majority(outfield)
            gk_lock: dict[int, int] = {}
            locked_goal_defenders: tuple[int, int] | None = None
            if self.kp_by_frame is not None:
                gk_lock, locked_goal_defenders = derive_gk_locks(
                    self.frames,
                    minimap_transforms=self.minimap_transforms_by_frame,
                )
            self._team_lock = TeamLocks(
                team_lock=team_lock,
                gk_lock=gk_lock,
                locked_goal_defenders=locked_goal_defenders,
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
        decorated = relock_detection_teams(
            decorated, locks.team_lock, gk_lock=locks.gk_lock or None,
        )
        return decorated

    def ball_detections_by_frame(self) -> dict:
        """Lazy load/compute dedicated ball YOLO detections."""
        if self._ball_by_frame is None:
            if self._cache is None:
                raise RuntimeError("session cache was not initialized")
            if self.device is None:
                raise RuntimeError("session device was not initialized")

            def factory():
                return create_ball_detector(
                    device=self.device,
                    model_path=self.ball_model_path,
                )

            self._ball_by_frame = build_or_load_ball_detections(
                self.source_video_path,
                factory,
                self._cache,
                max_frames=self.max_frames,
            )
        return self._ball_by_frame

    def pass_frames(self) -> list[tuple[int, sv.Detections]]:
        """Yield replay-decorated detections with ball attached, in video order."""
        if self._pass_frames is None:
            locks = self.team_locks()
            vel_smoother = KalmanVelocitySmoother(alpha=0.3)
            tracked_by = self.tracked_by_frame()
            ball_by = self.ball_detections_by_frame()
            out = []
            cap, _, _, _ = open_video(self.source_video_path)
            try:
                frame_idx = 0
                while True:
                    ret, _ = cap.read()
                    if not ret:
                        break
                    frame_idx += 1
                    if self.max_frames is not None and frame_idx > self.max_frames:
                        break
                    tracked = tracked_by.get(frame_idx, sv.Detections.empty())
                    dets = self.apply_replay_teams(
                        tracked,
                        locks=locks,
                        vel_smoother=vel_smoother,
                        frame_idx=frame_idx,
                    )
                    dets = attach_ball(
                        dets,
                        ball_by.get(frame_idx, sv.Detections.empty()),
                    )
                    out.append((frame_idx, dets))
            finally:
                cap.release()
            self._pass_frames = out
        return self._pass_frames

    @property
    def pass_by_frame(self) -> dict[int, sv.Detections]:
        return dict(self.pass_frames())

    @property
    def pass_scorer(self) -> PassQualityScorer:
        if self._pass_scorer is None:
            transformers = (
                self.gap_filled_transforms_by_frame if self.kp_by_frame else None
            )
            self._pass_scorer = PassQualityScorer(
                transformers=transformers,
                keypoints_by_frame=self.kp_by_frame,
                pitch_confidence=0.9,
            )
        return self._pass_scorer

    def pass_scan(self) -> PossessionScanResult:
        if self._pass_scan is None:
            config = PassDetectionConfig().for_frame_rate(self.fps)
            transformers = (
                self.gap_filled_transforms_by_frame if self.kp_by_frame else None
            )
            self._pass_scan = scan_possession_events(
                iter(self.pass_frames()),
                config=config,
                metric=True,
                transformers=transformers,
                fps=float(self.fps),
            )
        return self._pass_scan

    def pass_alternative_events(
        self,
        *,
        max_events: int | None = None,
        min_gap_frames: int = 90,
        weights: PassWeights | None = None,
    ) -> list[PassEvent]:
        """Cinematic freeze moments with top teammate pass lanes."""
        if self._pass_alternative_events is None:
            transformers = (
                self.gap_filled_transforms_by_frame if self.kp_by_frame else {}
            )
            self._pass_alternative_events = plan_pass_events(
                self.pass_frames(),
                fps=float(self.fps),
                frame_transforms=transformers,
                keypoints_by_frame=self.kp_by_frame or {},
                scorer=self.pass_scorer,
                weights=weights,
                max_events=max_events,
                min_gap_frames=min_gap_frames,
            )
        return self._pass_alternative_events


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
        ball_model_path=getattr(args, "ball_model_path", None),
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

    session = VideoTrackingSession(
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
    session._cache = cache
    session.device = args.device
    session.ball_model_path = getattr(args, "ball_model_path", None)
    return session
