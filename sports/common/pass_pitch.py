"""Pass analytics pitch-space helpers."""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

from sports.common.geometry import unit
from sports.common.goalkeeper import image_to_pitch_cm
from sports.common.homography import fit_pitch_homography
from sports.common.kinematics import feet_xy, player_mask
from sports.common.view import ViewTransformer
from sports.configs.soccer import TEAM_LEFT

__all__ = [
    "attack_direction",
    "image_displacement_to_pitch_m",
    "image_to_pitch_cm",
    "image_to_pitch_m",
    "lane_scoring_transformer_for_frame",
    "pitch_attack_direction",
    "pitch_cm_to_image",
    "pitch_circle_to_image",
]


def image_to_pitch_m(
    points_xy: np.ndarray, transformer: ViewTransformer | None
) -> np.ndarray | None:
    cm = image_to_pitch_cm(points_xy, transformer)
    if cm is None:
        return None
    return cm / 100.0


def image_displacement_to_pitch_m(
    feet_px: np.ndarray,
    displacement_px: np.ndarray,
    transformer: ViewTransformer | None,
) -> np.ndarray | None:
    """Map an image-space displacement at ``feet_px`` into pitch meters."""
    if transformer is None:
        return None
    feet = np.asarray(feet_px, dtype=np.float64).reshape(2)
    disp = np.asarray(displacement_px, dtype=np.float64).reshape(2)
    p0 = image_to_pitch_m(feet.reshape(1, 2), transformer)
    p1 = image_to_pitch_m((feet + disp).reshape(1, 2), transformer)
    if p0 is None or p1 is None:
        return None
    return p1[0] - p0[0]


def lane_scoring_transformer_for_frame(
    speed_transforms: dict[int, ViewTransformer | None] | None,
    frame_idx: int,
    keypoints: sv.KeyPoints | None,
    *,
    pitch_confidence: float = 0.9,
) -> ViewTransformer | None:
    """Prefer gated speed H; fall back to per-frame radar fit for lane scoring only."""
    if speed_transforms is not None:
        speed_t = speed_transforms.get(int(frame_idx))
        if speed_t is not None:
            return speed_t
    return fit_pitch_homography(keypoints, confidence=pitch_confidence)


def _attack_from_team_centroids(
    positions: np.ndarray,
    teams: np.ndarray,
    carrier_team: int,
    *,
    side_fallback: bool,
) -> np.ndarray:
    """Unit vector from own-team centroid toward opponent centroid."""
    own = positions[teams == carrier_team]
    opp = positions[teams == (1 - carrier_team)]
    if len(own) == 0 or len(opp) == 0:
        if side_fallback:
            return (
                np.array([1.0, 0.0])
                if carrier_team == TEAM_LEFT
                else np.array([-1.0, 0.0])
            )
        return np.array([1.0, 0.0])
    return unit(opp.mean(axis=0) - own.mean(axis=0))


def attack_direction(
    detections: sv.Detections,
    carrier_team: int,
    *,
    transformer: ViewTransformer | None = None,
    player_mask_fn=player_mask,
    feet_fn=feet_xy,
) -> np.ndarray:
    """Attack direction from own→opp team centroids (image px or pitch meters).

    With ``transformer``, positions are pitch meters and empty-team fallbacks use
    left/right pitch orientation. Without it, image feet are used.
    """
    pmask = player_mask_fn(detections)
    if not pmask.any():
        return np.array([1.0, 0.0])

    feet = feet_fn(detections)[pmask]
    teams = detections.data["team"][pmask]
    if transformer is None:
        return _attack_from_team_centroids(
            feet, teams, carrier_team, side_fallback=False,
        )

    pitch_xy = image_to_pitch_m(feet, transformer)
    if pitch_xy is None:
        return np.array([1.0, 0.0])
    return _attack_from_team_centroids(
        pitch_xy, teams, carrier_team, side_fallback=True,
    )


def pitch_attack_direction(
    detections: sv.Detections,
    carrier_team: int,
    transformer: ViewTransformer,
    *,
    player_mask_fn=player_mask,
    feet_fn=feet_xy,
) -> np.ndarray:
    """Pitch-meter attack direction (compat wrapper around :func:`attack_direction`)."""
    return attack_direction(
        detections,
        carrier_team,
        transformer=transformer,
        player_mask_fn=player_mask_fn,
        feet_fn=feet_fn,
    )


def pitch_cm_to_image(
    points_cm: np.ndarray, transformer: ViewTransformer | None
) -> np.ndarray | None:
    """Map pitch points (cm) back to image pixels via H^{-1}."""
    if transformer is None or points_cm.size == 0:
        return None
    pts = points_cm.reshape(-1, 1, 2).astype(np.float32)
    try:
        inv = np.linalg.inv(transformer.m)
    except np.linalg.LinAlgError:
        return None
    return cv2.perspectiveTransform(pts, inv).reshape(-1, 2)


def pitch_circle_to_image(
    center_m: np.ndarray,
    radius_m: float,
    transformer: ViewTransformer | None,
    *,
    n_points: int = 48,
) -> np.ndarray | None:
    """Project a pitch-space ground circle onto image pixels."""
    center = np.asarray(center_m, dtype=np.float64).reshape(2) * 100.0
    r_cm = float(radius_m) * 100.0
    angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    poly_cm = np.column_stack(
        [center[0] + r_cm * np.cos(angles), center[1] + r_cm * np.sin(angles)]
    )
    img = pitch_cm_to_image(poly_cm, transformer)
    if img is None:
        return None
    return np.round(img).astype(np.int32)
