"""Small vectorized geometry helpers shared by the demos."""

from __future__ import annotations

import numpy as np


def point_to_segment_distance(
    points: np.ndarray, a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Shortest distance from each point to the segment ``a -> b``.

    Args:
        points: ``(N, 2)`` array of points.
        a: ``(2,)`` segment start.
        b: ``(2,)`` segment end.

    Returns:
        ``(N,)`` array of distances.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-9:
        return np.linalg.norm(points - a, axis=1)
    t = ((points - a) @ ab) / denom
    t = np.clip(t, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(points - proj, axis=1)


def point_to_segment_distance_and_t(
    points: np.ndarray, a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Distance to segment ``a -> b`` and projection parameter ``t`` (0=start, 1=end)."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-9:
        t = np.zeros(len(points), dtype=np.float64)
        return np.linalg.norm(points - a, axis=1), t
    t = ((points - a) @ ab) / denom
    t_clip = np.clip(t, 0.0, 1.0)
    proj = a + t_clip[:, None] * ab
    return np.linalg.norm(points - proj, axis=1), t


def _pass_corridor_mask(
    dists: np.ndarray,
    t: np.ndarray,
    *,
    t_min: float,
    t_max: float,
    lane_width: float | None,
) -> np.ndarray:
    """Points that can intercept: on the segment and within ``lane_width`` (full width)."""
    on_segment = (t >= t_min) & (t <= t_max)
    if lane_width is None or lane_width <= 0:
        return on_segment
    return on_segment & (dists <= lane_width / 2.0)


def lane_segment_clearance(
    points: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    t_min: float = 0.0,
    t_max: float = 1.0,
    lane_width: float | None = None,
) -> tuple[float, float]:
    """Min perpendicular distance among intercept threats vs min over all points.

    *Lane* = projection ``t`` in ``[t_min, t_max]``. When ``lane_width`` is set (e.g.
    1 m in metric scoring), only rivals within half that distance of the pass line
    count as able to intercept.
    """
    if len(points) == 0:
        return float("inf"), float("inf")
    dists, t = point_to_segment_distance_and_t(points, a, b)
    segment_min = float(dists.min())
    in_corridor = _pass_corridor_mask(
        dists, t, t_min=t_min, t_max=t_max, lane_width=lane_width
    )
    if not in_corridor.any():
        return float("inf"), segment_min
    return float(dists[in_corridor].min()), segment_min


def count_lane_blockers(
    points: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    t_min: float = 0.0,
    t_max: float = 1.0,
    lane_width: float | None = None,
) -> int:
    """Count points in the pass corridor (segment projection + optional width)."""
    if len(points) == 0:
        return 0
    dists, t = point_to_segment_distance_and_t(points, a, b)
    return int(
        _pass_corridor_mask(
            dists, t, t_min=t_min, t_max=t_max, lane_width=lane_width
        ).sum()
    )


def lane_segment_clearance_body(
    feet: np.ndarray,
    body: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    t_min: float = 0.0,
    t_max: float = 1.0,
    lane_width: float | None = None,
    player_radius: np.ndarray | None = None,
) -> tuple[float, float]:
    """Lane clearance using min(feet, body) distance per player on the segment.

    *player_radius* shrinks effective distance (bbox half-width) so a player whose
    body crosses the pass line counts even when feet are slightly off the segment.
    """
    if len(feet) == 0:
        return float("inf"), float("inf")
    feet = np.asarray(feet, dtype=np.float64).reshape(-1, 2)
    body = np.asarray(body, dtype=np.float64).reshape(-1, 2)
    d_f, t_f = point_to_segment_distance_and_t(feet, a, b)
    d_b, t_b = point_to_segment_distance_and_t(body, a, b)
    on_seg = ((t_f >= t_min) & (t_f <= t_max)) | ((t_b >= t_min) & (t_b <= t_max))
    segment_min = float(min(d_f.min(), d_b.min()))
    if not on_seg.any():
        return segment_min, segment_min
    per_player = np.where(on_seg, np.minimum(d_f, d_b), np.inf)
    if player_radius is not None:
        per_player = np.maximum(0.0, per_player - np.asarray(player_radius, dtype=np.float64))
    if lane_width is None or lane_width <= 0:
        threats = per_player[on_seg]
    else:
        threats = per_player[on_seg & (per_player <= lane_width / 2.0)]
    if len(threats) == 0:
        return float("inf"), segment_min
    return float(np.min(threats)), segment_min


def count_lane_blockers_body(
    feet: np.ndarray,
    body: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    t_min: float = 0.0,
    t_max: float = 1.0,
    lane_width: float | None = None,
    player_radius: np.ndarray | None = None,
) -> int:
    if len(feet) == 0:
        return 0
    feet = np.asarray(feet, dtype=np.float64).reshape(-1, 2)
    body = np.asarray(body, dtype=np.float64).reshape(-1, 2)
    d_f, t_f = point_to_segment_distance_and_t(feet, a, b)
    d_b, t_b = point_to_segment_distance_and_t(body, a, b)
    on_seg = ((t_f >= t_min) & (t_f <= t_max)) | ((t_b >= t_min) & (t_b <= t_max))
    per_player = np.where(on_seg, np.minimum(d_f, d_b), np.inf)
    if player_radius is not None:
        per_player = np.maximum(0.0, per_player - np.asarray(player_radius, dtype=np.float64))
    if lane_width is None or lane_width <= 0:
        return int(on_seg.sum())
    return int((on_seg & (per_player <= lane_width / 2.0)).sum())


def pass_corridor_polygon(
    a: np.ndarray,
    b: np.ndarray,
    half_width: float,
    *,
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> np.ndarray:
    """Four corners of the pass corridor quad (same units as ``a`` / ``b``)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ab = b - a
    length = float(np.linalg.norm(ab))
    if length < 1e-9:
        return np.repeat(a.reshape(1, 2), 4, axis=0)
    u = ab / length
    perp = np.array([-u[1], u[0]], dtype=np.float64)
    start = a + u * (t_min * length)
    end = a + u * (t_max * length)
    hw = float(half_width)
    return np.array(
        [start + perp * hw, end + perp * hw, end - perp * hw, start - perp * hw],
        dtype=np.float64,
    )


def lane_blocking_mask_body(
    feet: np.ndarray,
    body: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    t_min: float = 0.0,
    t_max: float = 1.0,
    lane_width: float | None = None,
    player_radius: np.ndarray | None = None,
) -> np.ndarray:
    """Boolean mask (len ``feet``) of players blocking the corridor."""
    if len(feet) == 0:
        return np.zeros(0, dtype=bool)
    feet = np.asarray(feet, dtype=np.float64).reshape(-1, 2)
    body = np.asarray(body, dtype=np.float64).reshape(-1, 2)
    d_f, t_f = point_to_segment_distance_and_t(feet, a, b)
    d_b, t_b = point_to_segment_distance_and_t(body, a, b)
    on_seg = ((t_f >= t_min) & (t_f <= t_max)) | ((t_b >= t_min) & (t_b <= t_max))
    per_player = np.where(on_seg, np.minimum(d_f, d_b), np.inf)
    if player_radius is not None:
        per_player = np.maximum(0.0, per_player - np.asarray(player_radius, dtype=np.float64))
    if lane_width is None or lane_width <= 0:
        return on_seg
    return on_seg & (per_player <= lane_width / 2.0)


def unit(vector: np.ndarray) -> np.ndarray:
    """Return the unit vector; zero vector maps to zero."""
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return np.zeros_like(vector)
    return vector / norm
