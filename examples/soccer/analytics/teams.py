"""analytics/teams.py — clip-level team-id stabilization for analytics features.

Ported and adapted from world_cup_projects/common/teams.py
(``lock_teams_by_tracklet_majority``, lines 147-185). Locks one team per
``tracker_id`` for the whole clip using a majority shirt-colour vote, so brief
per-frame team-classifier noise cannot flip an outfield player mid-tracklet.

Adapted to this analytics package's role ids (PLAYER_CLASS_ID=2) and to return a
``{tracker_id: team}`` lock that render loops apply, rather than mutating cached
detections in place.
"""

from __future__ import annotations

import numpy as np
import supervision as sv

from analytics.support import PLAYER_CLASS_ID, TEAM_NONE


def lock_teams_by_tracklet_majority(
    frames: list[tuple[int, sv.Detections]],
) -> dict[int, int]:
    """Lock one team per outfield ``tracker_id`` (majority shirt-colour vote).

    Unlike hysteresis, the locked team never changes mid-tracklet. ``frames`` is a list
    of ``(frame_idx, detections)`` whose ``data['team']`` holds the per-frame classifier
    team for outfield players. Returns ``{tracker_id: team}`` for every voted player.
    """
    votes: dict[int, list[int]] = {}
    for _, dets in frames:
        if dets.tracker_id is None or dets.data is None:
            continue
        team = np.asarray(dets.data.get("team", np.full(len(dets), TEAM_NONE)), dtype=int)
        for i, tid in enumerate(dets.tracker_id):
            tid = int(tid)
            if tid < 0 or int(dets.class_id[i]) != PLAYER_CLASS_ID:
                continue
            raw = int(team[i])
            if raw in (0, 1):
                votes.setdefault(tid, []).append(raw)

    locked: dict[int, int] = {}
    for tid, vals in votes.items():
        counts = np.bincount(np.asarray(vals, dtype=int), minlength=2)
        locked[tid] = int(np.argmax(counts))
    return locked


def apply_team_lock(
    team_arr: np.ndarray,
    class_id: np.ndarray,
    tracker_id: np.ndarray | None,
    team_lock: dict[int, int],
) -> np.ndarray:
    """Override outfield-player team ids with the clip-level majority lock (in place)."""
    if tracker_id is None or not team_lock:
        return team_arr
    for i, tid in enumerate(tracker_id):
        tid = int(tid)
        if int(class_id[i]) == PLAYER_CLASS_ID and tid in team_lock:
            team_arr[i] = team_lock[tid]
    return team_arr
