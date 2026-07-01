import numpy as np
import supervision as sv

from sports.configs.soccer import REFEREE_CLASS_ID, TEAM_NONE

# Same palette as examples/soccer/main.py legacy demo modes.
PLAYER_COLORS = ["#FF1493", "#00BFFF", "#FF6347", "#FFD700"]

ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(PLAYER_COLORS),
    thickness=2,
)


def team_ellipse_color_lookup(detections: sv.Detections) -> np.ndarray:
    """Map each detection row to a PLAYER_COLORS palette index for EllipseAnnotator."""
    n = len(detections)
    lookup = np.full(n, 2, dtype=int)
    if n == 0 or detections.data is None:
        return lookup
    teams = detections.data.get("team", np.full(n, TEAM_NONE))
    for i in range(n):
        if int(detections.class_id[i]) == REFEREE_CLASS_ID:
            lookup[i] = REFEREE_CLASS_ID
        elif int(teams[i]) in (0, 1):
            lookup[i] = int(teams[i])
    return lookup


def annotate_team_ellipses(
    frame: np.ndarray,
    detections: sv.Detections,
) -> np.ndarray:
    """Draw team-colored ground ellipses via the shared supervision annotator."""
    if len(detections) == 0:
        return frame
    return ELLIPSE_ANNOTATOR.annotate(
        scene=frame,
        detections=detections,
        custom_color_lookup=team_ellipse_color_lookup(detections),
    )
