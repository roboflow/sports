from math import asin, degrees
from typing import List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv

from sports.handball.config import HandballCourtConfiguration


def _to_pixel(
    point: Tuple[float, float],
    scale: float,
    padding: int,
) -> Tuple[int, int]:
    return (
        int(round(point[0] * scale + padding)),
        int(round(point[1] * scale + padding)),
    )


def _arc_point(
    center: Tuple[float, float],
    radius: float,
    angle_degrees: float,
) -> Tuple[float, float]:
    angle = np.deg2rad(angle_degrees)
    return (
        center[0] + radius * np.cos(angle),
        center[1] + radius * np.sin(angle),
    )


def _draw_arc(
    image: np.ndarray,
    center: Tuple[float, float],
    radius: float,
    start_degrees: float,
    end_degrees: float,
    color: Tuple[int, int, int],
    thickness: int,
    scale: float,
    padding: int,
) -> None:
    center_px = _to_pixel(center, scale, padding)
    radius_px = int(round(radius * scale))
    cv2.ellipse(
        img=image,
        center=center_px,
        axes=(radius_px, radius_px),
        angle=0,
        startAngle=int(round(start_degrees)),
        endAngle=int(round(end_degrees)),
        color=color,
        thickness=thickness,
    )


def _draw_dashed_line(
    image: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int,
    dash_length: float,
    gap_length: float,
) -> None:
    start_np = np.array(start, dtype=float)
    end_np = np.array(end, dtype=float)
    delta = end_np - start_np
    length = float(np.linalg.norm(delta))
    if length == 0:
        return

    direction = delta / length
    distance = 0.0
    while distance < length:
        dash_start = start_np + direction * distance
        dash_end = start_np + direction * min(distance + dash_length, length)
        cv2.line(
            image,
            tuple(np.round(dash_start).astype(int)),
            tuple(np.round(dash_end).astype(int)),
            color,
            thickness,
        )
        distance += dash_length + gap_length


def _draw_dashed_arc(
    image: np.ndarray,
    center: Tuple[float, float],
    radius: float,
    start_degrees: float,
    end_degrees: float,
    color: Tuple[int, int, int],
    thickness: int,
    scale: float,
    padding: int,
    dash_length: float,
    gap_length: float,
    detail_degrees: float = 1,
) -> None:
    dash_length_degrees = degrees(dash_length / radius)
    gap_length_degrees = degrees(gap_length / radius)
    angle = start_degrees
    while angle < end_degrees:
        dash_end = min(angle + dash_length_degrees, end_degrees)
        samples = np.arange(angle, dash_end + detail_degrees, detail_degrees)
        points = np.array(
            [
                _to_pixel(_arc_point(center, radius, sample), scale, padding)
                for sample in samples
            ],
            dtype=np.int32,
        )
        if len(points) >= 2:
            cv2.polylines(
                image,
                [points],
                isClosed=False,
                color=color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )
        angle = dash_end + gap_length_degrees


def _draw_goal_frame(
    image: np.ndarray,
    config: HandballCourtConfiguration,
    side: str,
    color: Tuple[int, int, int],
    thickness: int,
    scale: float,
    padding: int,
) -> None:
    if side == "left":
        points = [
            (-config.goal_depth, config.goal_top_y),
            (0, config.goal_top_y),
            (-config.goal_depth, config.goal_top_y),
            (-config.goal_depth, config.goal_bottom_y),
            (-config.goal_depth, config.goal_bottom_y),
            (0, config.goal_bottom_y),
        ]
    else:
        points = [
            (config.length, config.goal_top_y),
            (config.length + config.goal_depth, config.goal_top_y),
            (config.length + config.goal_depth, config.goal_top_y),
            (config.length + config.goal_depth, config.goal_bottom_y),
            (config.length + config.goal_depth, config.goal_bottom_y),
            (config.length, config.goal_bottom_y),
        ]

    for i in range(0, len(points), 2):
        cv2.line(
            image,
            _to_pixel(points[i], scale, padding),
            _to_pixel(points[i + 1], scale, padding),
            color,
            thickness,
        )


def _draw_substitution_marks(
    image: np.ndarray,
    config: HandballCourtConfiguration,
    color: Tuple[int, int, int],
    thickness: int,
    scale: float,
    padding: int,
) -> None:
    mark_half_length = config.substitution_line_length / 2
    for x in (
        config.center_x - config.substitution_line_distance,
        config.center_x + config.substitution_line_distance,
    ):
        top_start = _to_pixel((x, -mark_half_length), scale, padding)
        top_end = _to_pixel((x, mark_half_length), scale, padding)
        bottom_start = _to_pixel((x, config.width - mark_half_length), scale, padding)
        bottom_end = _to_pixel((x, config.width + mark_half_length), scale, padding)
        cv2.line(image, top_start, top_end, color, thickness)
        cv2.line(image, bottom_start, bottom_end, color, thickness)


def draw_court(
    config: HandballCourtConfiguration,
    background_color: sv.Color = sv.Color(38, 132, 170),
    line_color: sv.Color = sv.Color.WHITE,
    goal_color: sv.Color = sv.Color.RED,
    throw_off_area_color: Optional[sv.Color] = None,
    padding: int = 50,
    line_thickness: int = 4,
    scale: float = 0.1,
) -> np.ndarray:
    """
    Draws a handball court with IHF-standard markings.

    Args:
        config (HandballCourtConfiguration): Configuration object containing the
            dimensions and layout of the court.
        background_color (sv.Color, optional): Color of the court background.
            Defaults to sv.Color(38, 132, 170).
        line_color (sv.Color, optional): Color of the court lines.
            Defaults to sv.Color.WHITE.
        goal_color (sv.Color, optional): Color of the goal frame.
            Defaults to sv.Color.RED.
        throw_off_area_color (Optional[sv.Color], optional): Fill color for
            the throw-off area. If None, only the circle line is drawn.
        padding (int, optional): Padding around the court in pixels.
            Defaults to 50.
        line_thickness (int, optional): Thickness of the court lines in pixels.
            Defaults to 4.
        scale (float, optional): Scaling factor for the court dimensions.
            Defaults to 0.1.

    Returns:
        np.ndarray: Image of the handball court.
    """
    scaled_width = int(round(config.width * scale))
    scaled_length = int(round(config.length * scale))
    court = np.ones(
        (scaled_width + 2 * padding, scaled_length + 2 * padding, 3),
        dtype=np.uint8,
    ) * np.array(background_color.as_bgr(), dtype=np.uint8)

    bgr_line = line_color.as_bgr()
    bgr_goal = goal_color.as_bgr()
    center = (config.center_x, config.center_y)

    if throw_off_area_color is not None:
        cv2.circle(
            court,
            _to_pixel(center, scale, padding),
            radius=int(round(config.throw_off_area_radius * scale)),
            color=throw_off_area_color.as_bgr(),
            thickness=-1,
        )

    for start, end in config.edges:
        point1 = _to_pixel(config.vertices[start - 1], scale, padding)
        point2 = _to_pixel(config.vertices[end - 1], scale, padding)
        cv2.line(court, point1, point2, bgr_line, line_thickness)

    _draw_goal_frame(
        court, config, "left", bgr_goal, line_thickness, scale, padding
    )
    _draw_goal_frame(
        court, config, "right", bgr_goal, line_thickness, scale, padding
    )

    left_top_goal_center = (0, config.goal_top_y)
    left_bottom_goal_center = (0, config.goal_bottom_y)
    right_top_goal_center = (config.length, config.goal_top_y)
    right_bottom_goal_center = (config.length, config.goal_bottom_y)

    _draw_arc(
        court,
        left_top_goal_center,
        config.goal_area_radius,
        270,
        360,
        bgr_line,
        line_thickness,
        scale,
        padding,
    )
    _draw_arc(
        court,
        left_bottom_goal_center,
        config.goal_area_radius,
        0,
        90,
        bgr_line,
        line_thickness,
        scale,
        padding,
    )
    _draw_arc(
        court,
        right_top_goal_center,
        config.goal_area_radius,
        180,
        270,
        bgr_line,
        line_thickness,
        scale,
        padding,
    )
    _draw_arc(
        court,
        right_bottom_goal_center,
        config.goal_area_radius,
        90,
        180,
        bgr_line,
        line_thickness,
        scale,
        padding,
    )

    sideline_angle = degrees(asin(config.goal_top_y / config.free_throw_radius))
    dash_length_px = max(1, config.free_throw_line_segment_length * scale)
    gap_length_px = max(1, config.free_throw_line_gap_length * scale)
    arc_dash_kwargs = {
        "dash_length": config.free_throw_line_segment_length,
        "gap_length": config.free_throw_line_gap_length,
    }

    _draw_dashed_arc(
        court,
        left_top_goal_center,
        config.free_throw_radius,
        360 - sideline_angle,
        360,
        bgr_line,
        line_thickness,
        scale,
        padding,
        **arc_dash_kwargs,
    )
    _draw_dashed_line(
        court,
        _to_pixel((config.free_throw_radius, config.goal_top_y), scale, padding),
        _to_pixel((config.free_throw_radius, config.goal_bottom_y), scale, padding),
        bgr_line,
        line_thickness,
        dash_length_px,
        gap_length_px,
    )
    _draw_dashed_arc(
        court,
        left_bottom_goal_center,
        config.free_throw_radius,
        0,
        sideline_angle,
        bgr_line,
        line_thickness,
        scale,
        padding,
        **arc_dash_kwargs,
    )
    _draw_dashed_arc(
        court,
        right_top_goal_center,
        config.free_throw_radius,
        180,
        180 + sideline_angle,
        bgr_line,
        line_thickness,
        scale,
        padding,
        **arc_dash_kwargs,
    )
    _draw_dashed_line(
        court,
        _to_pixel(
            (config.length - config.free_throw_radius, config.goal_top_y),
            scale,
            padding,
        ),
        _to_pixel(
            (config.length - config.free_throw_radius, config.goal_bottom_y),
            scale,
            padding,
        ),
        bgr_line,
        line_thickness,
        dash_length_px,
        gap_length_px,
    )
    _draw_dashed_arc(
        court,
        right_bottom_goal_center,
        config.free_throw_radius,
        180 - sideline_angle,
        180,
        bgr_line,
        line_thickness,
        scale,
        padding,
        **arc_dash_kwargs,
    )

    _draw_substitution_marks(
        court, config, bgr_line, line_thickness, scale, padding
    )

    cv2.circle(
        court,
        _to_pixel(center, scale, padding),
        radius=int(round(config.throw_off_area_radius * scale)),
        color=bgr_line,
        thickness=line_thickness,
    )

    return court


def draw_points_on_court(
    config: HandballCourtConfiguration,
    xy: np.ndarray,
    face_color: sv.Color = sv.Color.RED,
    edge_color: sv.Color = sv.Color.BLACK,
    radius: int = 10,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    court: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Draws points on a handball court.

    Args:
        config (HandballCourtConfiguration): Configuration object containing the
            dimensions and layout of the court.
        xy (np.ndarray): Array of points to draw as (x, y) coordinates.
        face_color (sv.Color, optional): Color of the point faces.
            Defaults to sv.Color.RED.
        edge_color (sv.Color, optional): Color of the point edges.
            Defaults to sv.Color.BLACK.
        radius (int, optional): Radius of the points in pixels.
            Defaults to 10.
        thickness (int, optional): Thickness of the point edges in pixels.
            Defaults to 2.
        padding (int, optional): Padding around the court in pixels.
            Defaults to 50.
        scale (float, optional): Scaling factor for court coordinates.
            Defaults to 0.1.
        court (Optional[np.ndarray], optional): Existing court image to draw on.

    Returns:
        np.ndarray: Image of the handball court with points drawn on it.
    """
    if court is None:
        court = draw_court(config=config, padding=padding, scale=scale)

    if xy is None or np.size(xy) == 0:
        return court

    for point in np.atleast_2d(xy):
        scaled_point = _to_pixel(tuple(point), scale, padding)
        cv2.circle(
            img=court,
            center=scaled_point,
            radius=radius,
            color=face_color.as_bgr(),
            thickness=-1,
        )
        cv2.circle(
            img=court,
            center=scaled_point,
            radius=radius,
            color=edge_color.as_bgr(),
            thickness=thickness,
        )

    return court


def draw_paths_on_court(
    config: HandballCourtConfiguration,
    paths: List[np.ndarray],
    color: sv.Color = sv.Color.WHITE,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    court: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Draws paths on a handball court.

    Args:
        config (HandballCourtConfiguration): Configuration object containing the
            dimensions and layout of the court.
        paths (List[np.ndarray]): List of paths with (x, y) coordinates.
        color (sv.Color, optional): Color of the paths.
            Defaults to sv.Color.WHITE.
        thickness (int, optional): Thickness of the paths in pixels.
            Defaults to 2.
        padding (int, optional): Padding around the court in pixels.
            Defaults to 50.
        scale (float, optional): Scaling factor for court coordinates.
            Defaults to 0.1.
        court (Optional[np.ndarray], optional): Existing court image to draw on.

    Returns:
        np.ndarray: Image of the handball court with paths drawn on it.
    """
    if court is None:
        court = draw_court(config=config, padding=padding, scale=scale)

    if not paths:
        return court

    for path in paths:
        if path is None or np.size(path) == 0:
            continue

        scaled_path = [
            _to_pixel(tuple(point), scale, padding)
            for point in np.atleast_2d(path)
            if point.size > 0 and not np.isnan(point).any()
        ]

        if len(scaled_path) < 2:
            continue

        for i in range(len(scaled_path) - 1):
            cv2.line(
                img=court,
                pt1=scaled_path[i],
                pt2=scaled_path[i + 1],
                color=color.as_bgr(),
                thickness=thickness,
            )

    return court
