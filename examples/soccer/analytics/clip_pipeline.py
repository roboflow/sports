"""Per-clip analysis pipeline shared by all analytics modes.

Loads cached detections and pitch keypoints, runs a single BoTSORT pass with team and
goalkeeper resolution, builds optional homography maps and kinematics, and exposes the
result as :class:`ClipAnalysis`. Used by standalone mode entry points and by ``run-all``
so tracking, homography, and kinematics are computed once per clip. Does not own
low-level trackers, smoothers, or draw helpers (see ``player_motion``) or mode renders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import supervision as sv

from analytics.cache import (
    FrameCache,
    build_or_load_detections,
    build_or_load_keypoints,
)
from analytics.goalkeepers import (
    apply_goalkeeper_frame,
    collect_team_frames,
    derive_clip_locks,
)
from analytics.homography import (
    MetricContext,
    build_metric_from_maps,
    build_radar_homography_map,
)
from analytics.player_motion import (
    GOALKEEPER_CLASS_ID,
    PLAYER_CLASS_ID,
    TEAM_NONE,
    collect_referee_tracker_ids,
    collect_tracks,
    compute_kinematics,
    create_pitch_keypoint_detector,
    create_player_detector,
    create_player_tracker,
    drop_blocked_tracker_ids,
    fit_team_classifier,
    open_video,
    resolve_goalkeepers_team_id,
)
from analytics.teams import apply_team_lock, relock_detection_teams

DEFAULT_PLAYER_MODEL_ID = "football-players-detection-3zvbc/11"
DEFAULT_PITCH_MODEL_ID = "football-field-detection-f07vi/15"


@dataclass
class ClipLocks:
    """Clip-level stabilization locks for one goalkeeper-assignment strategy."""

    team_lock: dict[int, int]
    gk_lock: dict[int, int]
    locked_goal_defenders: tuple[int, int] | None


@dataclass
class ClipAnalysis:
    """Shared, compute-once groundwork for the player-motion analytics renders.

    Built by :func:`compute_clip_analysis`. Holds the cached detections / keypoints, the
    fitted team classifier, the single BoTSORT pass output (``frames`` with stable ids,
    outfield teams and captured per-frame Kalman velocity), the referee-blocked tracklet
    ids, and lazily-built homography maps, locks (per goalkeeper-assignment) and
    kinematics. Renderers consume this instead of rebuilding the pipeline.
    """

    args: Any
    source_video_path: str
    fps: float
    width: int
    height: int
    max_frames: int | None
    tracker_kind: str
    needs_frame: bool

    det_by_frame: dict[int, sv.Detections]
    kp_by_frame: dict[int, sv.KeyPoints] | None
    team_classifier: Any

    # Single shared BoTSORT pass (referee rows still present in ``referee_frames``).
    frames: list[tuple[int, sv.Detections]]
    referee_frames: list[tuple[int, sv.Detections]]
    blocked_ids: frozenset[int]

    # Lazily-populated caches.
    _metric: MetricContext | None = field(default=None, repr=False)
    _speed_transforms: dict[int, Any] | None = field(default=None, repr=False)
    _gap_filled: dict[int, Any] | None = field(default=None, repr=False)
    _radar_h_by_frame: dict[int, Any] | None = field(default=None, repr=False)
    _locks: dict[str, ClipLocks] = field(default_factory=dict, repr=False)
    _tracks: dict[int, Any] | None = field(default=None, repr=False)

    # -- homography ----------------------------------------------------------
    @property
    def metric(self) -> MetricContext:
        """Gated speed H + no-mirror radar H + keypoints, built once from the cache."""
        if self._metric is None:
            if not self.kp_by_frame:
                raise RuntimeError(
                    "ClipAnalysis.metric requested but pitch keypoints were not computed."
                )
            self._metric = build_metric_from_maps(
                self.kp_by_frame,
                detections_by_frame=self.det_by_frame,
                pitch_confidence=0.9,
            )
        return self._metric

    @property
    def speed_transforms(self) -> dict[int, Any]:
        """Gated speed H per frame (distance integration)."""
        if self._speed_transforms is None:
            self._speed_transforms = self.metric.speed_transforms
        return self._speed_transforms

    @property
    def gap_filled(self) -> dict[int, Any]:
        """Speed H per frame, gated where available else ungated keypoint H (displayed speed)."""
        if self._gap_filled is None:
            self._gap_filled = self.metric.speed_transforms_gap_filled(0.9)
        return self._gap_filled

    @property
    def radar_h_by_frame(self) -> dict[int, Any]:
        """No-mirror keypoint-radar H per frame for the visible minimap / traces / dots."""
        if self._radar_h_by_frame is None:
            self._radar_h_by_frame = build_radar_homography_map(self.metric, confidence=0.9)
        return self._radar_h_by_frame

    # -- locks ---------------------------------------------------------------
    def locks(self, gk_assignment: str) -> ClipLocks:
        """Clip locks for ``gk_assignment`` ('goal_distance' or 'centroid'), cached.

        Derived from the single shared tracking pass without re-tracking. The frames are
        deep-copied per assignment so the goalkeeper team-fill of one strategy cannot leak
        into another (or into the pristine ``frames`` the replay renders read).
        """
        if gk_assignment not in self._locks:
            metric = self.metric if gk_assignment == "goal_distance" else None
            team_lock, gk_lock, locked_goal_defenders = derive_clip_locks(
                _clone_team_frames(self.frames),
                gk_assignment=gk_assignment,
                metric=metric,
                pitch_confidence=0.9,
            )
            self._locks[gk_assignment] = ClipLocks(
                team_lock=team_lock,
                gk_lock=gk_lock,
                locked_goal_defenders=locked_goal_defenders,
            )
        return self._locks[gk_assignment]

    # -- kinematics ----------------------------------------------------------
    @property
    def tracks(self) -> dict[int, Any]:
        """Per-track cumulative-distance kinematics, computed once from the shared pass.

        Tracks are collected from the shared pass's tracked detections (referee-blocked
        tracklets dropped first, exactly as the standalone DISTANCE / SPEED_AND_DISTANCE first
        pass does) and integrated with the gated speed homographies.
        """
        if self._tracks is None:
            tracks = collect_tracks(self.iter_tracked())
            compute_kinematics(
                tracks,
                self.fps,
                mode="homography",
                frame_transforms=self.speed_transforms,
                min_frames=2,
            )
            self._tracks = tracks
        return self._tracks

    # -- replay helpers ------------------------------------------------------
    def iter_tracked(self):
        """Yield ``(frame_idx, tracked)`` with referee-blocked tracklets dropped.

        ``tracked`` carries the shared pass's stable ids, outfield-team predictions and the
        captured per-frame Kalman velocity (``data['kf_vx']`` / ``data['kf_vy']``). This is
        the byte-for-byte equivalent of what a standalone render pass produces for that
        frame, minus the per-frame goalkeeper / team-lock decoration the renderer applies.
        """
        for frame_idx, dets in self.frames:
            yield frame_idx, drop_blocked_tracker_ids(dets, self.blocked_ids)

    def tracked_by_frame(self) -> dict[int, sv.Detections]:
        """``{frame_idx: tracked}`` (referee-blocked dropped) for O(1) replay lookup."""
        return {fi: dets for fi, dets in self.iter_tracked()}

    def decorate_replay_frame(
        self,
        frame_idx: int,
        tracked: sv.Detections,
        *,
        gk_assignment: str,
        locks: ClipLocks,
        vel_smoother,
    ) -> sv.Detections:
        """Apply the per-frame team / goalkeeper decoration to a replayed tracked frame.

        Reproduces the standalone render loops' decoration starting from the shared pass's
        tracked detections (stable ids + outfield-team predictions + captured Kalman
        velocity): apply the clip team lock, resolve goalkeepers (centroid inline or
        goal-distance via the gated radar H), smooth the captured velocity, then let the
        clip team lock have the final say. The captured single-update velocity matches what
        standalone and run-all renders read for every mode.
        """
        team_arr = (
            np.array(tracked.data.get("team"), dtype=int)
            if tracked.data and tracked.data.get("team") is not None
            else np.full(len(tracked), TEAM_NONE, dtype=int)
        )
        team_arr = apply_team_lock(
            team_arr, tracked.class_id, tracked.tracker_id, locks.team_lock
        )
        if gk_assignment == "centroid" and len(tracked):
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
        if gk_assignment == "goal_distance":
            decorated = apply_goalkeeper_frame(
                decorated, self.metric.radar_transforms.get(frame_idx), locks.gk_lock
            )
        decorated = vel_smoother.smooth_detections(decorated)
        decorated = relock_detection_teams(decorated, locks.team_lock)
        return decorated


def _clone_team_frames(
    frames: list[tuple[int, sv.Detections]],
) -> list[tuple[int, sv.Detections]]:
    """Copy ``frames`` with fresh ``data`` dicts + copied ``team`` arrays.

    Lock derivation mutates the goalkeeper ``data['team']`` entries in place; cloning keeps
    the shared pass pristine so it can back more than one goalkeeper-assignment strategy and
    the replay renders.
    """
    cloned: list[tuple[int, sv.Detections]] = []
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


def _make_player_detector_factory(args) -> Callable[[], Callable[[np.ndarray], sv.Detections]]:
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


def _make_pitch_detector_factory(args) -> Callable[[], Callable[[np.ndarray], sv.KeyPoints]]:
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


def compute_clip_analysis(args, *, need_homography: bool = True) -> ClipAnalysis:
    """Run the shared analytics groundwork once and return a :class:`ClipAnalysis`.

    Performs the cached detection / keypoint load, fits the team classifier, and runs the
    single referee-filtered BoTSORT pass (capturing per-frame Kalman velocity). Homography
    maps, locks and kinematics are built lazily on first access. When ``need_homography``
    is False the pitch keypoints are still loaded from cache only if requested by a later
    consumer — DIRECTION, which has no pitch homography, passes ``need_homography=False``
    and never triggers the keypoint load or homography build.
    """
    _, fps, width, height = open_video(args.source_video_path)
    max_frames = getattr(args, "max_frames", None)
    tracker_kind = getattr(args, "tracker", "botsort")
    # CMC needs the frame only for the full BoTSORT tracker; botsort_nocmc / bytetrack
    # ignore it, so the frame is passed only for "botsort".
    needs_frame = tracker_kind == "botsort"

    player_model_id = getattr(args, "player_model_id", DEFAULT_PLAYER_MODEL_ID)
    pitch_model_id = getattr(args, "pitch_model_id", DEFAULT_PITCH_MODEL_ID)
    cache = FrameCache(
        args.source_video_path,
        cache_dir=getattr(args, "cache_dir", None),
        enabled=getattr(args, "cache", True),
        player_backend=args.player_detector,
        player_model_id=player_model_id,
        pitch_backend=args.pitch_detector,
        pitch_model_id=pitch_model_id,
    )
    det_by_frame = build_or_load_detections(
        args.source_video_path,
        _make_player_detector_factory(args),
        cache,
        max_frames=max_frames,
    )
    kp_by_frame: dict[int, sv.KeyPoints] | None = None
    if need_homography:
        kp_by_frame = build_or_load_keypoints(
            args.source_video_path,
            _make_pitch_detector_factory(args),
            cache,
            max_frames=max_frames,
        )

    print("Fitting team classifier…")
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

    print("Tracking once (shared BoTSORT pass)…")
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
        kp_by_frame=kp_by_frame,
        team_classifier=team_classifier,
        frames=frames,
        referee_frames=referee_frames,
        blocked_ids=blocked_ids,
    )
