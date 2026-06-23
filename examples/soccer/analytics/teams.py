"""analytics/teams.py — clip-level team-id stabilization for analytics features.

Locks one team per ``tracker_id`` for the whole clip using a majority shirt-colour
vote, so brief per-frame team-classifier noise cannot flip an outfield player
mid-tracklet.

Adapted to this analytics package's role ids (PLAYER_CLASS_ID=2) and to return a
``{tracker_id: team}`` lock that render loops apply, rather than mutating cached
detections in place.
"""

from __future__ import annotations

import numpy as np
import supervision as sv

from analytics.player_motion import TEAM_NONE


def lock_teams_by_tracklet_majority(
    frames: list[tuple[int, sv.Detections]],
) -> dict[int, int]:
    """Lock one team per ``tracker_id`` (majority shirt-colour vote across the clip).

    The vote is keyed on ``tracker_id`` regardless of the per-frame detected class, so a
    track whose class flips between player and goalkeeper still gets a single team: every
    frame on which ``data['team']`` holds a valid team (0/1) contributes a vote, whether
    that frame was detected as a player or as a goalkeeper. Unlike hysteresis, the locked
    team never changes mid-tracklet. ``frames`` is a list of ``(frame_idx, detections)``.
    Returns ``{tracker_id: team}`` for every voted track.
    """
    votes: dict[int, list[int]] = {}
    for _, dets in frames:
        if dets.tracker_id is None or dets.data is None:
            continue
        team = np.asarray(dets.data.get("team", np.full(len(dets), TEAM_NONE)), dtype=int)
        for i, tid in enumerate(dets.tracker_id):
            tid = int(tid)
            if tid < 0:
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
    """Override team ids with the clip-level majority lock, keyed on ``tracker_id``.

    Applies to every row whose ``tracker_id`` is locked regardless of the per-frame
    detected class, so a class-flipping track (e.g. a keeper detected as a player on some
    frames) keeps one stable team on every frame. ``class_id`` is accepted for signature
    stability but no longer gates the override.
    """
    if tracker_id is None or not team_lock:
        return team_arr
    for i, tid in enumerate(tracker_id):
        tid = int(tid)
        if tid in team_lock:
            team_arr[i] = team_lock[tid]
    return team_arr


def relock_detection_teams(
    dets: sv.Detections, team_lock: dict[int, int]
) -> sv.Detections:
    """Return ``dets`` with ``data['team']`` re-locked to the clip-level team.

    Apply this as the final team step in a render loop, after any per-frame goalkeeper
    assignment: it makes the clip-level per-tracklet lock authoritative for every locked
    track regardless of class, so a keeper whose detected class flips frame-to-frame never
    changes colour. Tracks that were never locked keep their per-frame team.
    """
    if not team_lock or dets.tracker_id is None or len(dets) == 0:
        return dets
    team = np.asarray(
        dets.data.get("team", np.full(len(dets), TEAM_NONE)) if dets.data else np.full(len(dets), TEAM_NONE),
        dtype=int,
    )
    team = apply_team_lock(team, dets.class_id, dets.tracker_id, team_lock)
    data = dict(dets.data) if dets.data else {}
    data["team"] = team
    return sv.Detections(
        xyxy=dets.xyxy,
        class_id=dets.class_id,
        tracker_id=dets.tracker_id,
        confidence=dets.confidence,
        data=data,
    )
