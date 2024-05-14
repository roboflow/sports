from typing import Optional

import cv2
import supervision as sv
import numpy as np

from sports.configs.soccer import SoccerFieldConfiguration


def draw_soccer_field(
    config: SoccerFieldConfiguration,
    background_color: sv.Color = sv.Color(34, 139, 34),
    line_color: sv.Color = sv.Color.WHITE,
    padding: int = 50,
    line_thickness: int = 4,
    point_radius: int = 8,
    scale: float = 0.1
) -> np.ndarray:
    """
    Draws a soccer field based on the given configuration.

    Args:
        config (SoccerFieldConfiguration): Configuration of the soccer field.
        background_color (sv.Color, optional): Background color of the field.
            Defaults to sv.Color(34, 139, 34).
        line_color (sv.Color, optional): Color of the field lines.
            Defaults to sv.Color.WHITE.
        padding (int, optional): Padding around the field. Defaults to 50.
        line_thickness (int, optional): Thickness of the field lines.
            Defaults to 4.
        point_radius (int, optional): Radius of the points. Defaults to 8.
        scale (float, optional): Scale factor for the field dimensions.
            Defaults to 0.1.

    Returns:
        np.ndarray: Image of the soccer field.
    """
    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)
    scaled_padding = padding
    scaled_circle_radius = int(config.centre_circle_radius * scale)
    scaled_penalty_spot_distance = int(config.penalty_spot_distance * scale)

    field_image = np.ones(
        (scaled_width + 2 * scaled_padding,
         scaled_length + 2 * scaled_padding, 3),
        dtype=np.uint8
    ) * np.array(background_color.as_bgr(), dtype=np.uint8)

    for start, end in config.edges:
        point1 = (int(config.vertices[start - 1][0] * scale) + scaled_padding,
                  int(config.vertices[start - 1][1] * scale) + scaled_padding)
        point2 = (int(config.vertices[end - 1][0] * scale) + scaled_padding,
                  int(config.vertices[end - 1][1] * scale) + scaled_padding)
        cv2.line(
            img=field_image,
            pt1=point1,
            pt2=point2,
            color=line_color.as_bgr(),
            thickness=line_thickness
        )

    centre_circle_center = (
        scaled_length // 2 + scaled_padding,
        scaled_width // 2 + scaled_padding
    )
    cv2.circle(
        img=field_image,
        center=centre_circle_center,
        radius=scaled_circle_radius,
        color=line_color.as_bgr(),
        thickness=line_thickness
    )

    penalty_spots = [
        (
            scaled_penalty_spot_distance + scaled_padding,
            scaled_width // 2 + scaled_padding
        ),
        (
            scaled_length - scaled_penalty_spot_distance + scaled_padding,
            scaled_width // 2 + scaled_padding
        )
    ]
    for spot in penalty_spots:
        cv2.circle(
            img=field_image,
            center=spot,
            radius=point_radius,
            color=line_color.as_bgr(),
            thickness=-1
        )

    return field_image


def draw_players(
    config: SoccerFieldConfiguration,
    xy: np.ndarray,
    face_color: sv.Color = sv.Color.RED,
    edge_color: sv.Color = sv.Color.BLACK,
    radius: int = 10,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    soccer_field: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draws players on the soccer field.

    Args:
        config (SoccerFieldConfiguration): Configuration of the soccer field.
        xy (np.ndarray): Array of player positions with shape (N, 2).
        face_color (sv.Color, optional): Fill color for the players.
            Defaults to sv.Color.RED.
        edge_color (sv.Color, optional): Edge color for the players.
            Defaults to sv.Color.BLACK.
        radius (int, optional): Radius of the player circles. Defaults to 10.
        thickness (int, optional): Thickness of the edge lines. Defaults to 2.
        padding (int, optional): Padding around the field. Defaults to 50.
        scale (float, optional): Scale factor for the field dimensions.
            Defaults to 0.1.
        soccer_field (Optional[np.ndarray], optional): Pre-drawn soccer field
            map. If None, a new field is drawn. Defaults to None.

    Returns:
        np.ndarray: Image of the soccer field with players.
    """
    if soccer_field is None:
        soccer_field = draw_soccer_field(
            config=config,
            padding=padding,
            scale=scale
        )

    scaled_padding = padding
    for position in xy:
        point = (
            int(position[0] * scale) + scaled_padding,
            int(position[1] * scale) + scaled_padding
        )
        cv2.circle(
            img=soccer_field,
            center=point,
            radius=radius,
            color=face_color.as_bgr(),
            thickness=-1
        )
        cv2.circle(
            img=soccer_field,
            center=point,
            radius=radius,
            color=edge_color.as_bgr(),
            thickness=thickness
        )

    return soccer_field


def draw_field_control(
    config: SoccerFieldConfiguration,
    team_1_xy: np.ndarray,
    team_2_xy: np.ndarray,
    team_1_color: sv.Color = sv.Color.RED,
    team_2_color: sv.Color = sv.Color.WHITE,
    opacity: float = 0.5,
    padding: int = 50,
    scale: float = 0.1,
    soccer_field: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draws a field control heatmap based on player positions.

    Args:
        config (SoccerFieldConfiguration): Configuration of the soccer field.
        team_1_xy (np.ndarray): Array of player positions for team 1.
        team_2_xy (np.ndarray): Array of player positions for team 2.
        team_1_color (sv.Color, optional): Color for team 1 areas.
            Defaults to sv.Color.RED.
        team_2_color (sv.Color, optional): Color for team 2 areas.
            Defaults to sv.Color.WHITE.
        opacity (float, optional): Opacity of the heatmap. Defaults to 0.5.
        padding (int, optional): Padding around the field. Defaults to 50.
        scale (float, optional): Scale factor for the field dimensions.
            Defaults to 0.1.
        soccer_field (Optional[np.ndarray], optional): Pre-drawn soccer field
            map. If None, a new field is drawn. Defaults to None.

    Returns:
        np.ndarray: Image of the soccer field with the control heatmap.
    """
    if soccer_field is None:
        soccer_field = draw_soccer_field(
            config=config,
            padding=padding,
            scale=scale
        )

    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)
    scaled_padding = padding

    heatmap = np.zeros_like(soccer_field, dtype=np.uint8)

    team_1_color_bgr = np.array(team_1_color.as_bgr(), dtype=np.uint8)
    team_2_color_bgr = np.array(team_2_color.as_bgr(), dtype=np.uint8)

    y_coords, x_coords = np.indices((scaled_width + 2 * scaled_padding,
                                     scaled_length + 2 * scaled_padding))
    y_coords -= scaled_padding
    x_coords -= scaled_padding

    def calculate_distances(xy, x_coords, y_coords):
        return np.sqrt((xy[:, 0][:, None, None] * scale - x_coords) ** 2 +
                       (xy[:, 1][:, None, None] * scale - y_coords) ** 2)

    distances_team_1 = calculate_distances(team_1_xy, x_coords, y_coords)
    distances_team_2 = calculate_distances(team_2_xy, x_coords, y_coords)

    min_distances_team_1 = np.min(distances_team_1, axis=0)
    min_distances_team_2 = np.min(distances_team_2, axis=0)

    control_mask = min_distances_team_1 < min_distances_team_2

    heatmap[control_mask] = team_1_color_bgr
    heatmap[~control_mask] = team_2_color_bgr

    overlay = cv2.addWeighted(heatmap, opacity, soccer_field, 1 - opacity, 0)

    return overlay
