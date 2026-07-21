"""Goalkeeper team assignment: goal-distance when homography exists, centroid fallback."""

from __future__ import annotations

import numpy as np
import supervision as sv

from sports.common.kinematics import feet_xy, player_mask
from sports.common.team import (
    TeamLocks,
    _fill_goalkeeper_teams_by_centroid,
    clone_team_frames,
    lock_teams_by_tracklet_majority,
)
from sports.common.tracking import resolve_goalkeepers_team_id
from sports.configs.soccer import (
    GOALKEEPER_CLASS_ID,
    PLAYER_CLASS_ID,
    SoccerPitchConfiguration,
    TEAM_NONE,
)

TEAM_LEFT = 0
TEAM_RIGHT = 1

_PITCH = SoccerPitchConfiguration()
PITCH_LENGTH_CM = float(_PITCH.length)
PITCH_WIDTH_CM = float(_PITCH.width)


def image_to_pitch_cm(points_xy: np.ndarray, transformer) -> np.ndarray | None:
    if transformer is None or points_xy is None or points_xy.size == 0:
        return None
    return transformer.transform_points(points_xy.astype(np.float32))


def infer_goal_defenders(
    pitch_xy_cm: np.ndarray, teams: np.ndarray, n_defenders: int = 3
) -> tuple[int, int]:
    """Return ``(left_goal_team, right_goal_team)`` using defensive blocks.

    For each team, take the mean pitch-X of the ``n_defenders`` most defensive
    players at each end (lowest X on the left, highest X on the right). Assign
    each goal to the team with the stronger defensive-block margin there
    (handshake: prefer the side with the larger margin so both goals stay opposite).
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

    left_margin = l1 - l0
    right_margin = r1 - r0
    return (0, 1) if left_margin >= right_margin else (1, 0)


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


def apply_goalkeeper_teams_by_goal(dets: sv.Detections, transformer) -> sv.Detections:
    """Set goalkeeper team ids from pitch distance to each goal (one frame)."""
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
        dets.data.get("team", np.full(len(dets), TEAM_NONE))
        if dets.data
        else np.full(len(dets), TEAM_NONE),
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


def _goal_warmup_ready(
    pitch_cm: np.ndarray,
    teams: np.ndarray,
    *,
    min_players: int = 8,
    min_x_spread_cm: float = 1400.0,
) -> bool:
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
    """Clip-level vote for which team defends left vs right goal."""
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
        teams = (
            np.asarray(base_team, dtype=int)[pmask]
            if base_team is not None
            else np.full(int(pmask.sum()), TEAM_NONE)
        )
        if not _goal_warmup_ready(pitch_cm, teams):
            continue
        pair = infer_goal_defenders(pitch_cm, teams)
        votes[pair] = votes.get(pair, 0) + 1
    if not votes:
        return None
    return max(votes, key=votes.get)


def _sample_stabilize_frames(
    frames: list[tuple[int, sv.Detections]],
    *,
    max_warmup_frames: int,
    max_sample_frames: int,
) -> list[tuple[int, sv.Detections]]:
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
    """Lock each goalkeeper tracklet to one defending team over the clip."""
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


def apply_goalkeeper_frame(
    dets: sv.Detections, transformer, gk_lock: dict[int, int] | None
) -> sv.Detections:
    """Per-frame goal-distance assignment with clip-stable gk_lock override."""
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


def _apply_goalkeeper_centroid(dets: sv.Detections) -> sv.Detections:
    """Centroid fallback when homography is unavailable for this frame."""
    if len(dets) == 0 or dets.data is None:
        return dets
    team = np.asarray(
        dets.data.get("team", np.full(len(dets), TEAM_NONE)), dtype=int
    )
    gk_mask = dets.class_id == GOALKEEPER_CLASS_ID
    pl_mask = dets.class_id == PLAYER_CLASS_ID
    if not gk_mask.any():
        return dets
    if not ((team[pl_mask] == 0).any() and (team[pl_mask] == 1).any()):
        return dets
    gk_teams = resolve_goalkeepers_team_id(dets[pl_mask], team[pl_mask], dets[gk_mask])
    team[gk_mask] = gk_teams
    data = dict(dets.data)
    data["team"] = team
    return sv.Detections(
        xyxy=dets.xyxy,
        class_id=dets.class_id,
        tracker_id=dets.tracker_id,
        confidence=dets.confidence,
        data=data,
    )


def apply_goalkeeper_teams(
    dets: sv.Detections,
    *,
    transformer,
    locks: TeamLocks,
) -> sv.Detections:
    """Route to goal-distance when H exists, else nearest-team centroid."""
    if transformer is not None:
        return apply_goalkeeper_frame(dets, transformer, locks.gk_lock or None)
    return _apply_goalkeeper_centroid(dets)


def derive_gk_locks(
    frames: list[tuple[int, sv.Detections]],
    *,
    minimap_transforms: dict[int, object] | None = None,
) -> tuple[dict[int, int], tuple[int, int] | None]:
    """Derive goalkeeper and goal-defender locks without mutating outfield teams."""
    if minimap_transforms:
        gk_frames = clone_team_frames(frames)
        locked_goal_defenders = warmup_goal_defenders_radar(
            gk_frames, minimap_transforms,
        )
        gk_lock = stabilize_goalkeeper_teams(
            gk_frames,
            transforms=minimap_transforms,
            locked_goal_defenders=locked_goal_defenders,
            mutate=False,
        )
        return gk_lock, locked_goal_defenders
    return {}, None


def derive_clip_locks(
    frames: list[tuple[int, sv.Detections]],
    *,
    minimap_transforms: dict[int, object] | None = None,
) -> TeamLocks:
    """Derive team / goalkeeper / goal-defender locks from a tracking pass."""
    outfield = clone_team_frames(frames)
    team_lock = lock_teams_by_tracklet_majority(outfield)

    gk_lock: dict[int, int] = {}
    locked_goal_defenders: tuple[int, int] | None = None
    if minimap_transforms:
        gk_lock, locked_goal_defenders = derive_gk_locks(
            frames, minimap_transforms=minimap_transforms,
        )
    else:
        _fill_goalkeeper_teams_by_centroid(outfield)

    return TeamLocks(
        team_lock=team_lock,
        gk_lock=gk_lock,
        locked_goal_defenders=locked_goal_defenders,
    )
