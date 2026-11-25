"""analytics/goalkeepers.py — optional goal-distance goalkeeper team assignment.

This is ADDITIVE and OPTIONAL. It does not touch the sports library or the existing
``resolve_goalkeepers_team_id`` centroid function in ``analytics/player_motion.py`` (kept as the
``centroid`` option / fallback). The better path here assigns each goalkeeper to the team
defending the *nearer goal mouth*, decides goal sides from a defensive block (the N most
defensive outfield players per team — not a whole-team centroid), and locks each goalkeeper
tracklet's team over the whole clip.

Class-id note: this analytics package uses PLAYER_CLASS_ID=2 / GOALKEEPER_CLASS_ID=1
(see ``analytics/class_ids.py``). Goal-side ids follow the sports pitch template: TEAM_LEFT=0
defends x≈0, TEAM_RIGHT=1 defends x≈length.
"""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

from sports.configs.soccer import SoccerPitchConfiguration

from analytics.class_ids import GOALKEEPER_CLASS_ID, PLAYER_CLASS_ID, TEAM_NONE
from analytics.player_motion import (
    build_trackable_detections,
    collect_referee_tracker_ids,
    combine_for_referee_check,
    feet_xy,
    get_crops,
    kalman_velocity_arrays,
    player_mask,
    resolve_goalkeepers_team_id,
    split_detection_roles,
)
from analytics.teams import lock_teams_by_tracklet_majority

# Goal-side team ids (match the sports pitch template orientation).
TEAM_LEFT = 0
TEAM_RIGHT = 1

_PITCH = SoccerPitchConfiguration()
PITCH_LENGTH_CM = float(_PITCH.length)  # 12000
PITCH_WIDTH_CM = float(_PITCH.width)    # 7000


# ---------------------------------------------------------------------------
# Pitch coordinate helper (cm). analytics/homography exposes
# RansacViewTransformer.transform_points directly, so this is a thin None/empty-safe
# wrapper rather than a re-implementation.
# ---------------------------------------------------------------------------

def image_to_pitch_cm(points_xy: np.ndarray, transformer) -> np.ndarray | None:
    if transformer is None or points_xy is None or points_xy.size == 0:
        return None
    return transformer.transform_points(points_xy.astype(np.float32))


# ---------------------------------------------------------------------------
# infer_goal_defenders — defensive-block goal-side mapping (n_defenders=3)
# ---------------------------------------------------------------------------

def infer_goal_defenders(
    pitch_xy_cm: np.ndarray, teams: np.ndarray, n_defenders: int = 3
) -> tuple[int, int]:
    """Return ``(left_goal_team, right_goal_team)`` using defensive blocks.

    Instead of team-wide averages, look at the 'defensive block' (mean X of the N most
    defensive players) for each team at both ends, then assign teams to the side where their
    defensive advantage margin is strongest.
    """
    x_by_team: dict[int, np.ndarray] = {}
    for tid in (0, 1):
        mask = teams == tid
        x_by_team[tid] = np.sort(pitch_xy_cm[mask, 0]) if mask.any() else np.array([])

    if x_by_team[0].size == 0 or x_by_team[1].size == 0:
        return TEAM_LEFT, TEAM_RIGHT

    def _block_avg(x_sorted: np.ndarray, side: str) -> float:
        n = min(len(x_sorted), n_defenders)
        return float(x_sorted[:n].mean()) if side == "left" else float(x_sorted[-n:].mean())

    l0, r0 = _block_avg(x_by_team[0], "left"), _block_avg(x_by_team[0], "right")
    l1, r1 = _block_avg(x_by_team[1], "left"), _block_avg(x_by_team[1], "right")

    left_margin = l1 - l0   # >0 → team 0 is further left than team 1
    right_margin = r1 - r0  # >0 → team 1 is further right than team 0
    return (0, 1) if left_margin >= right_margin else (1, 0)


# ---------------------------------------------------------------------------
# resolve_goalkeepers_team_by_goal — nearer goal mouth → defending team
# ---------------------------------------------------------------------------

def resolve_goalkeepers_team_by_goal(
    goalkeepers_pitch_cm: np.ndarray,
    outfield_pitch_cm: np.ndarray,
    outfield_team_id: np.ndarray,
    *,
    pitch_length_cm: float = PITCH_LENGTH_CM,
    pitch_width_cm: float = PITCH_WIDTH_CM,
) -> np.ndarray:
    """Assign each GK to the team defending the nearer goal mouth."""
    if goalkeepers_pitch_cm is None or len(goalkeepers_pitch_cm) == 0:
        return np.array([], dtype=int)

    if (
        outfield_pitch_cm is not None
        and len(outfield_pitch_cm) >= 4
        and np.any(outfield_team_id == 0)
        and np.any(outfield_team_id == 1)
    ):
        left_def, right_def = infer_goal_defenders(outfield_pitch_cm, outfield_team_id)
    else:
        left_def, right_def = TEAM_LEFT, TEAM_RIGHT

    left_goal = np.array([0.0, pitch_width_cm / 2.0], dtype=np.float32)
    right_goal = np.array([pitch_length_cm, pitch_width_cm / 2.0], dtype=np.float32)
    ids: list[int] = []
    for xy in goalkeepers_pitch_cm:
        d_left = float(np.linalg.norm(xy - left_goal))
        d_right = float(np.linalg.norm(xy - right_goal))
        ids.append(left_def if d_left < d_right else right_def)
    return np.array(ids, dtype=int)


# ---------------------------------------------------------------------------
# apply_goalkeeper_teams_by_goal — per-frame GK team from pitch distance to goals
# ---------------------------------------------------------------------------

def apply_goalkeeper_teams_by_goal(dets: sv.Detections, transformer) -> sv.Detections:
    """Set ``data['team']`` on goalkeeper rows from pitch distance to each goal (one frame).

    Returns ``dets`` unchanged when there are no goalkeepers or the homography is unusable.
    """
    if len(dets) == 0:
        return dets
    gk_mask = dets.class_id == GOALKEEPER_CLASS_ID
    out_mask = dets.class_id == PLAYER_CLASS_ID
    if not gk_mask.any():
        return dets

    feet = feet_xy(dets)
    gk_cm = image_to_pitch_cm(feet[gk_mask], transformer)
    if gk_cm is None or not np.isfinite(gk_cm).all():
        return dets

    out_cm = None
    out_teams = None
    if out_mask.any():
        out_cm = image_to_pitch_cm(feet[out_mask], transformer)
        base_team = dets.data.get("team") if dets.data else None
        if base_team is not None:
            out_teams = np.asarray(base_team, dtype=int)[out_mask]

    gk_teams = resolve_goalkeepers_team_by_goal(
        gk_cm,
        out_cm if out_cm is not None else np.empty((0, 2), dtype=np.float32),
        out_teams if out_teams is not None else np.array([], dtype=int),
    )

    team = np.array(
        dets.data.get("team", np.full(len(dets), TEAM_NONE)) if dets.data else np.full(len(dets), TEAM_NONE),
        dtype=int,
    )
    team[gk_mask] = gk_teams
    data = dict(dets.data) if dets.data else {}
    data["team"] = team
    return sv.Detections(
        xyxy=dets.xyxy,
        class_id=dets.class_id,
        tracker_id=dets.tracker_id,
        confidence=dets.confidence,
        data=data,
    )


# ---------------------------------------------------------------------------
# warmup_goal_defenders_radar — clip-level lock of which team defends each goal
# ---------------------------------------------------------------------------

def _goal_warmup_ready(
    pitch_cm: np.ndarray,
    teams: np.ndarray,
    *,
    min_players: int = 8,
    min_x_spread_cm: float = 1400.0,
) -> bool:
    """Enough players on a plausible pitch layout to vote defending teams."""
    if pitch_cm is None or len(pitch_cm) < min_players:
        return False
    if not np.isfinite(pitch_cm).all():
        return False
    if not (np.any(teams == 0) and np.any(teams == 1)):
        return False
    x_spread = float(np.percentile(pitch_cm[:, 0], 90) - np.percentile(pitch_cm[:, 0], 10))
    return x_spread >= min_x_spread_cm


def warmup_goal_defenders_radar(
    frames_with_dets,
    transforms: dict[int, object] | None,
    *,
    sample_step: int = 8,
) -> tuple[int, int] | None:
    """Lock left/right defending teams using the same sports-radar H as the minimap.

    Votes the defensive-block goal-side mapping on sampled frames and returns the majority.
    ``transforms`` should be the memoized ``MetricContext.keypoint_radar_transforms`` map.
    """
    if not transforms:
        return None
    sampled_frames = [
        (frame_idx, dets)
        for frame_idx, dets in frames_with_dets
        if int(frame_idx) % sample_step == 0
    ]
    votes: dict[tuple[int, int], int] = {}
    for frame_idx, dets in sampled_frames:
        transformer = transforms.get(int(frame_idx))
        if transformer is None:
            continue
        pmask = player_mask(dets)
        if not pmask.any():
            continue
        pitch_cm = image_to_pitch_cm(feet_xy(dets)[pmask], transformer)
        if pitch_cm is None:
            continue
        base_team = dets.data.get("team") if dets.data else None
        teams = np.asarray(base_team, dtype=int)[pmask] if base_team is not None else np.full(int(pmask.sum()), TEAM_NONE)
        if not _goal_warmup_ready(pitch_cm, teams):
            continue
        pair = infer_goal_defenders(pitch_cm, teams)
        votes[pair] = votes.get(pair, 0) + 1
    if not votes:
        return None
    return max(votes, key=votes.get)


# ---------------------------------------------------------------------------
# stabilize_goalkeeper_teams — clip-level GK team lock per tracklet
# ---------------------------------------------------------------------------

def _sample_stabilize_frames(
    frames: list[tuple[int, sv.Detections]],
    *,
    max_warmup_frames: int,
    max_sample_frames: int,
) -> list[tuple[int, sv.Detections]]:
    """First ``max_warmup_frames`` clip entries, evenly subsampled to at most ``max_sample_frames``."""
    warmup = frames[:max_warmup_frames]
    if not warmup:
        return []
    if len(warmup) <= max_sample_frames:
        return warmup
    sample_idxs = np.linspace(0, len(warmup) - 1, max_sample_frames, dtype=int)
    return [warmup[i] for i in np.unique(sample_idxs)]


def stabilize_goalkeeper_teams(
    frames: list[tuple[int, sv.Detections]],
    *,
    transforms: dict[int, object] | None = None,
    locked_goal_defenders: tuple[int, int] | None = None,
    max_warmup_frames: int = 30,
    max_sample_frames: int = 10,
    min_gk_frames: int = 3,
    mutate: bool = True,
) -> dict[int, int]:
    """Lock each goalkeeper tracklet to one defending team over the whole clip.

    1. Identify tracklets ever detected as a goalkeeper in a small early-clip sample.
    2. Average pitch X from precomputed per-frame radar H (``transforms``).
    3. Assign the team defending the nearer goal for the whole tracklet.

    Only the first ``max_warmup_frames`` entries of ``frames`` are considered, subsampled
    evenly to at most ``max_sample_frames`` indices (linspace over that prefix). This is
    enough for clip-level GK side lock via avg pitch X vs midline.

    ``transforms`` should be ``MetricContext.keypoint_radar_transforms(confidence)`` — a
    memoized map of ``ViewTransformer`` per frame. No homography is fit inside this loop.

    ``min_gk_frames`` counts goalkeeper-class appearances **within the sample only** (default
    3 ≈ 30% of 10 sampled frames; raise toward ``max_sample_frames`` for stricter locks).

    Returns ``{tracker_id: team}`` for every stabilized goalkeeper tracklet. When
    ``mutate`` is True, also patches ``data['team']`` on all supplied frames in place.
    """
    if locked_goal_defenders:
        left_def, right_def = locked_goal_defenders
    else:
        left_def, right_def = TEAM_LEFT, TEAM_RIGHT

    sampled_frames = _sample_stabilize_frames(
        frames,
        max_warmup_frames=max_warmup_frames,
        max_sample_frames=max_sample_frames,
    )
    if not sampled_frames or not transforms:
        return {}

    track_positions: dict[int, list[float]] = {}
    gk_frame_counts: dict[int, int] = {}

    for frame_idx, dets in sampled_frames:
        if dets.tracker_id is None:
            continue
        t = transforms.get(int(frame_idx))
        if t is None:
            continue

        feet_cm = image_to_pitch_cm(feet_xy(dets), t)
        if feet_cm is None:
            continue

        for i, tid in enumerate(dets.tracker_id):
            tid = int(tid)
            if tid < 0:
                continue
            if int(dets.class_id[i]) == GOALKEEPER_CLASS_ID:
                gk_frame_counts[tid] = gk_frame_counts.get(tid, 0) + 1
            if np.isfinite(feet_cm[i]).all():
                track_positions.setdefault(tid, []).append(float(feet_cm[i, 0]))

    stable: dict[int, int] = {}
    pitch_mid_cm = PITCH_LENGTH_CM / 2.0
    for tid, count in gk_frame_counts.items():
        if count < min_gk_frames:
            continue
        positions = track_positions.get(tid)
        if not positions:
            continue
        avg_x = sum(positions) / len(positions)
        stable[tid] = left_def if avg_x < pitch_mid_cm else right_def

    if mutate and stable:
        for _, dets in frames:
            if dets.tracker_id is None or dets.data is None:
                continue
            team_array = dets.data.get("team")
            if team_array is None:
                continue
            team_array = np.asarray(team_array, dtype=int)
            for i, tid in enumerate(dets.tracker_id):
                tid = int(tid)
                if tid in stable:
                    team_array[i] = stable[tid]
            dets.data["team"] = team_array

    return stable


# ---------------------------------------------------------------------------
# compute_goalkeeper_lock — one detect→track→classify clip pass → GK team lock
# ---------------------------------------------------------------------------

def apply_goalkeeper_frame(
    dets: sv.Detections, transformer, gk_lock: dict[int, int] | None
) -> sv.Detections:
    """Render-loop GK assignment for goal-distance mode.

    Applies the live per-frame goal-distance assignment, then overrides with the clip-stable
    per-tracklet lock where available (the lock wins for goalkeeper tracklets seen enough).
    """
    dets = apply_goalkeeper_teams_by_goal(dets, transformer)
    if gk_lock and dets.tracker_id is not None and dets.data is not None:
        team = np.asarray(
            dets.data.get("team", np.full(len(dets), TEAM_NONE)), dtype=int
        )
        for i, tid in enumerate(dets.tracker_id):
            tid = int(tid)
            if tid in gk_lock:
                team[i] = gk_lock[tid]
        dets.data["team"] = team
    return dets


def collect_team_frames(
    source_video_path: str,
    *,
    player_detector_fn=None,
    team_classifier,
    tracker,
    needs_frame: bool,
    max_frames: int | None = None,
    detections_by_frame: dict[int, sv.Detections] | None = None,
    capture_velocity: bool = False,
) -> tuple[list[tuple[int, sv.Detections]], list[tuple[int, sv.Detections]]]:
    """One detect→track→classify pass over the clip.

    Returns ``(team_frames, referee_frames)``:
      - ``team_frames``: ``[(frame_idx, detections)]`` of tracked players + goalkeepers
        with outfield players team-classified; goalkeepers stay TEAM_NONE here (resolved
        later by the goal-distance / centroid logic).
      - ``referee_frames``: ``[(frame_idx, detections)]`` stacking the tracked rows with
        the raw referee rows, used to flag player tracklets that coincide with a referee.

    The trackable set is built via :func:`build_trackable_detections` (the same builder
    the render passes use) so referees are excluded and tracker ids align across passes.
    ``tracker`` must be a fresh tracker of the same kind used in the render pass. When
    ``detections_by_frame`` is supplied the raw detections come from it (cache) instead
    of the detector.

    When ``capture_velocity`` is True the per-frame feet-referenced Kalman velocity is
    read straight off the tracker after its single update and attached to each frame's
    detections as ``data['kf_vx']`` / ``data['kf_vy']``. This lets a downstream consumer
    (the shared :class:`~analytics.clip_pipeline.ClipAnalysis`) reuse this single tracking
    pass for the render velocity instead of advancing a second tracker; it matches the
    single-update velocity the DISTANCE / SPEED_AND_DISTANCE render passes read today. The
    default (False) leaves the returned detections untouched.
    """
    cap = cv2.VideoCapture(source_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {source_video_path}")

    frames: list[tuple[int, sv.Detections]] = []
    referee_frames: list[tuple[int, sv.Detections]] = []
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if max_frames is not None and frame_idx > max_frames:
                break
            if detections_by_frame is not None:
                raw = detections_by_frame.get(frame_idx)
                if raw is None:
                    raw = sv.Detections.empty()
            else:
                raw = player_detector_fn(frame)
            trackable = build_trackable_detections(raw, frame_width=float(frame.shape[1]))
            _, _, refs = split_detection_roles(raw)
            tracked = (
                tracker.update(trackable, frame=frame if needs_frame else None)
                if len(trackable)
                else sv.Detections.empty()
            )
            team_arr = np.full(len(tracked), TEAM_NONE, dtype=int)
            if len(tracked):
                t_players = tracked[tracked.class_id == PLAYER_CLASS_ID]
                if len(t_players):
                    team_arr[tracked.class_id == PLAYER_CLASS_ID] = team_classifier.predict(
                        get_crops(frame, t_players)
                    )
            data = {"team": team_arr}
            if capture_velocity:
                kf_vx, kf_vy = kalman_velocity_arrays(tracked, tracker)
                data["kf_vx"] = kf_vx
                data["kf_vy"] = kf_vy
            dets = sv.Detections(
                xyxy=tracked.xyxy,
                class_id=tracked.class_id,
                tracker_id=tracked.tracker_id,
                confidence=tracked.confidence,
                data=data,
            )
            frames.append((frame_idx, dets))
            referee_frames.append((frame_idx, combine_for_referee_check(tracked, refs)))
    finally:
        cap.release()
    return frames, referee_frames


def derive_clip_locks(
    frames: list[tuple[int, sv.Detections]],
    *,
    gk_assignment: str = "goal_distance",
    metric=None,
    pitch_confidence: float = 0.9,
) -> tuple[dict[int, int], dict[int, int], tuple[int, int] | None]:
    """Derive the team / goalkeeper / goal-defender locks from a collected clip pass.

    A single ``collect_team_frames`` tracking pass can be shared (e.g. by
    :class:`~analytics.clip_pipeline.ClipAnalysis`) and the locks derived for more than
    one ``gk_assignment`` without re-tracking. ``frames`` is the ``team_frames`` list
    from :func:`collect_team_frames`. NOTE: this mutates the goalkeeper ``data['team']``
    entries on ``frames`` (the goal-distance / centroid fill), so callers that need
    pristine frames for more than one assignment must pass a copy.

    Returns ``(team_lock, gk_lock, locked_goal_defenders)``.
    """
    # Resolve goalkeeper-row teams on the collected frames BEFORE the majority vote so
    # that frames detected as a goalkeeper also contribute a team vote for their track.
    # This is what lets a class-flipping keeper (player on some frames, goalkeeper on
    # others) end up with one combined per-tracker_id team instead of flickering between
    # the player-classifier team and the goalkeeper team.
    gk_lock: dict[int, int] = {}
    locked_goal_defenders: tuple[int, int] | None = None
    if gk_assignment == "goal_distance" and metric is not None:
        radar_transforms = metric.keypoint_radar_transforms(pitch_confidence)
        locked_goal_defenders = warmup_goal_defenders_radar(
            frames, radar_transforms
        )
        gk_lock = stabilize_goalkeeper_teams(
            frames,
            transforms=radar_transforms,
            locked_goal_defenders=locked_goal_defenders,
            mutate=True,
        )
    else:
        _fill_goalkeeper_teams_centroid(frames)

    # One clip-level lock per tracker_id, combining player-classifier votes and the
    # goalkeeper votes filled in above. Applied to every row of a track regardless of
    # its per-frame class.
    team_lock = lock_teams_by_tracklet_majority(frames)

    # Keep the per-frame goalkeeper path in lock-step with the unified lock so the two
    # never disagree for the same track.
    for tid in list(gk_lock):
        if tid in team_lock:
            gk_lock[tid] = team_lock[tid]

    return team_lock, gk_lock, locked_goal_defenders


def _fill_goalkeeper_teams_centroid(frames: list[tuple[int, sv.Detections]]) -> None:
    """Fill goalkeeper-row ``data['team']`` per frame via the team-centroid rule (in place).

    Used for the ``centroid`` assignment so that goalkeeper-detected frames still vote in
    the per-tracker_id team lock; without this a keeper whose class flips would only be
    voted on its player-detected frames.
    """
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
