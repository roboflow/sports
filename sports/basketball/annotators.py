import cv2
import numpy as np
import supervision as sv
from typing import Tuple, Optional, List
from sports.basketball.config import CourtConfiguration
from sports.common.core import MeasurementUnit


BACKBOARD_TO_RIM_CM = 63.5
BACKBOARD_SPAN_CM = 183.0


def _to_pixel(
    point: Tuple[float, float],
    scale: float,
    padding: int,
) -> Tuple[int, int]:
    """Scale court point to pixel space and apply padding."""
    return (
        int(round(point[0] * scale + padding)),
        int(round(point[1] * scale + padding)),
    )


def _draw_circular_arc_from_three_points(
    image: np.ndarray,
    first_point: Tuple[float, float],
    middle_point: Tuple[float, float],
    last_point: Tuple[float, float],
    bgr_color: Tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw an arc defined by three points."""
    def circle_from_three_points(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
    ):
        sum_sq_p2 = p2[0] ** 2 + p2[1] ** 2
        term1 = (p1[0] ** 2 + p1[1] ** 2 - sum_sq_p2) / 2.0
        term2 = (sum_sq_p2 - p3[0] ** 2 - p3[1] ** 2) / 2.0
        det = (
            (p1[0] - p2[0]) * (p2[1] - p3[1])
            - (p2[0] - p3[0]) * (p1[1] - p2[1])
        )
        if abs(det) < 1e-10:
            return (p2, 0.0)
        center_x = (term1 * (p2[1] - p3[1]) - term2 * (p1[1] - p2[1])) / det
        center_y = (
            (p1[0] - p2[0]) * term2 - (p2[0] - p3[0]) * term1
        ) / det
        radius = float(np.hypot(center_x - p1[0], center_y - p1[1]))
        return (center_x, center_y), radius

    def angle_deg(center_xy, point_xy) -> float:
        return float(
            np.degrees(
                np.arctan2(point_xy[1] - center_xy[1], point_xy[0] - center_xy[0])
            )
        )

    center, radius = circle_from_three_points(
        first_point, middle_point, last_point
    )
    center_px = (int(round(center[0])), int(round(center[1])))
    radius_px = int(round(radius))

    start_angle = angle_deg(center, first_point)
    end_angle = angle_deg(center, last_point)
    if end_angle < start_angle:
        end_angle += 360.0

    cv2.ellipse(
        image,
        center=center_px,
        axes=(radius_px, radius_px),
        angle=0,
        startAngle=int(round(start_angle)),
        endAngle=int(round(end_angle)),
        color=bgr_color,
        thickness=thickness,
    )


def _ellipse_point(
    center: Tuple[int, int],
    axes: Tuple[int, int],
    rotation_degrees: float,
    theta_degrees: float,
) -> Tuple[int, int]:
    cx, cy = center
    a, b = axes
    phi = np.deg2rad(rotation_degrees)
    t = np.deg2rad(theta_degrees)
    x = cx + a * np.cos(t) * np.cos(phi) - b * np.sin(t) * np.sin(phi)
    y = cy + a * np.cos(t) * np.sin(phi) + b * np.sin(t) * np.cos(phi)
    return int(round(x)), int(round(y))


def _draw_dashed_ellipse(
    image: np.ndarray,
    center: Tuple[int, int],
    axes: Tuple[int, int],
    rotation_degrees: float,
    start_degrees: float,
    end_degrees: float,
    bgr_color: Tuple[int, int, int],
    thickness: int,
    dash_length_degrees: float = 12.0,
    gap_length_degrees: float = 8.0,
    detail_degrees: float = 2.0,
) -> None:
    """Draw a dashed elliptical arc."""
    angle = start_degrees
    while angle < end_degrees:
        dash_start = angle
        dash_end = min(angle + dash_length_degrees, end_degrees)
        t = dash_start
        prev_point = _ellipse_point(center, axes, rotation_degrees, t)
        t += detail_degrees
        while t <= dash_end:
            cur_point = _ellipse_point(center, axes, rotation_degrees, t)
            cv2.line(image, prev_point, cur_point, bgr_color, thickness)
            prev_point = cur_point
            t += detail_degrees
        angle = dash_end + gap_length_degrees


def draw_court(
    config: CourtConfiguration,
    scale: float = 20,
    padding: int = 50,
    line_thickness: int = 4,
    line_color: sv.Color = sv.Color.WHITE,
    background_color: sv.Color = sv.Color(224, 190, 139),
    paint_color: Optional[sv.Color] = None,
) -> np.ndarray:
    """Render a basketball court to an image."""
    court_height_px = int(round(config.court_width * scale))
    court_length_px = int(round(config.court_length * scale))
    center_circle_radius_px = int(round(config.center_circle_radius * scale))

    image = np.zeros(
        (court_height_px + 2 * padding, court_length_px + 2 * padding, 3),
        dtype=np.uint8,
    )
    image[:, :] = background_color.as_bgr()

    # Paint fill beneath lines
    if paint_color is not None:
        left_poly = np.array(
            [_to_pixel(config.vertices[i], scale, padding)
             for i in config.left_paint_indexes],
            dtype=np.int32,
        )
        right_poly = np.array(
            [_to_pixel(config.vertices[i], scale, padding)
             for i in config.right_paint_indexes],
            dtype=np.int32,
        )
        cv2.fillPoly(image, [left_poly], color=paint_color.as_bgr())
        cv2.fillPoly(image, [right_poly], color=paint_color.as_bgr())

    # Court edges
    for start_idx, end_idx in config.edges:
        start_px = _to_pixel(config.vertices[start_idx], scale, padding)
        end_px = _to_pixel(config.vertices[end_idx], scale, padding)
        cv2.line(image, start_px, end_px, line_color.as_bgr(), line_thickness)

    # Straight three-point segments
    for start_idx, end_idx in [(1, 7), (4, 8), (28, 24), (31, 25)]:
        start_px = _to_pixel(config.vertices[start_idx], scale, padding)
        end_px = _to_pixel(config.vertices[end_idx], scale, padding)
        cv2.line(image, start_px, end_px, line_color.as_bgr(), line_thickness)

    # Right-side border joins
    for start_idx, end_idx in [(18, 27), (20, 32)]:
        start_px = _to_pixel(config.vertices[start_idx], scale, padding)
        end_px = _to_pixel(config.vertices[end_idx], scale, padding)
        cv2.line(image, start_px, end_px, line_color.as_bgr(), line_thickness)

    # Stitch right sideline segment
    right_top_px = _to_pixel(config.vertices[28], scale, padding)
    right_bottom_px = _to_pixel(config.vertices[31], scale, padding)
    cv2.line(image, right_top_px, right_bottom_px,
             line_color.as_bgr(), line_thickness)

    # Throw-in short lines
    throw_in_specs = [
        (config.vertices[12],
         (config.vertices[12][0],
          config.vertices[12][1] + config.baseline_to_throw_line_length)),
        (config.vertices[14],
         (config.vertices[14][0],
          config.vertices[14][1] - config.baseline_to_throw_line_length)),
        (config.vertices[18],
         (config.vertices[18][0],
          config.vertices[18][1] + config.baseline_to_throw_line_length)),
        (config.vertices[20],
         (config.vertices[20][0],
          config.vertices[20][1] - config.baseline_to_throw_line_length)),
    ]
    for start_pt, end_pt in throw_in_specs:
        start_px = _to_pixel(start_pt, scale, padding)
        end_px = _to_pixel(end_pt, scale, padding)
        cv2.line(image, start_px, end_px, line_color.as_bgr(), line_thickness)

    # Center circle
    center_px = _to_pixel(config.vertices[16], scale, padding)
    cv2.circle(
        image,
        center_px,
        center_circle_radius_px,
        line_color.as_bgr(),
        line_thickness,
    )

    # Baseline x in pixel space
    left_baseline_x = _to_pixel((0.0, 0.0), scale, padding)[0]
    right_baseline_x = _to_pixel((config.court_length, 0.0), scale, padding)[0]

    # Backboard real-world constants
    if config.measurement_unit == MeasurementUnit.FEET:
        backboard_to_rim = BACKBOARD_TO_RIM_CM / 30.48
        backboard_span = BACKBOARD_SPAN_CM / 30.48
    else:
        backboard_to_rim = BACKBOARD_TO_RIM_CM
        backboard_span = BACKBOARD_SPAN_CM

    # Hoops, restricted areas, free-throw circles, ticks, backboards
    for side in ["left", "right"]:
        basket_idx = (
            config.left_basket_index if side == "left"
            else config.right_basket_index
        )
        basket_x, basket_y = config.vertices[basket_idx]
        basket_px = _to_pixel((basket_x, basket_y), scale, padding)

        rim_radius_px = int(round((config.rim_diameter / 2.0) * scale))
        restricted_radius_px = int(
            round(config.restricted_area_radius * scale)
        )

        # Rim
        cv2.circle(
            image, basket_px, rim_radius_px,
            line_color.as_bgr(), line_thickness
        )

        # Restricted area semicircle
        start_ang, end_ang = ((180, 360) if side == "left" else (0, 180))
        cv2.ellipse(
            image,
            center=basket_px,
            angle=90,
            axes=(restricted_radius_px, restricted_radius_px),
            startAngle=start_ang,
            endAngle=end_ang,
            color=line_color.as_bgr(),
            thickness=line_thickness,
        )

        # Free-throw circle center
        free_idx = 10 if side == "left" else 22
        free_x, free_y = config.vertices[free_idx]
        free_px = _to_pixel((free_x, free_y), scale, padding)

        # Free-throw semicircle, solid inside paint
        cv2.ellipse(
            image,
            center=free_px,
            angle=90,
            axes=(center_circle_radius_px, center_circle_radius_px),
            startAngle=start_ang,
            endAngle=end_ang,
            color=line_color.as_bgr(),
            thickness=line_thickness,
        )

        # Free-throw semicircle, dashed outside paint
        dashed_start, dashed_end = (
            (0, 180) if side == "left" else (180, 360)
        )
        _draw_dashed_ellipse(
            image=image,
            center=free_px,
            axes=(center_circle_radius_px, center_circle_radius_px),
            rotation_degrees=90,
            start_degrees=dashed_start,
            end_degrees=dashed_end,
            bgr_color=line_color.as_bgr(),
            thickness=line_thickness,
            dash_length_degrees=12.0,
            gap_length_degrees=8.0,
            detail_degrees=2.0,
        )

        # Three-point arc from three points
        arc_indices = [7, 13, 8] if side == "left" else [25, 19, 24]
        a_px, b_px, c_px = [
            _to_pixel(config.vertices[i], scale, padding) for i in arc_indices
        ]
        _draw_circular_arc_from_three_points(
            image, a_px, b_px, c_px, line_color.as_bgr(), line_thickness
        )

        # Tick lines from free-throw circle intersections to baselines
        x_free = free_px[0]
        y_top = free_px[1] - center_circle_radius_px
        y_bottom = free_px[1] + center_circle_radius_px
        if side == "left":
            cv2.line(
                image, (left_baseline_x, y_top), (x_free, y_top),
                line_color.as_bgr(), line_thickness
            )
            cv2.line(
                image, (left_baseline_x, y_bottom), (x_free, y_bottom),
                line_color.as_bgr(), line_thickness
            )
        else:
            cv2.line(
                image, (x_free, y_top), (right_baseline_x, y_top),
                line_color.as_bgr(), line_thickness
            )
            cv2.line(
                image, (x_free, y_bottom), (right_baseline_x, y_bottom),
                line_color.as_bgr(), line_thickness
            )

        # Backboard, 2x thickness, touching the rim
        backboard_x = (
            basket_x - backboard_to_rim if side == "left"
            else basket_x + backboard_to_rim
        )
        half_span = backboard_span / 2.0
        backboard_top_px = _to_pixel(
            (backboard_x, basket_y - half_span), scale, padding
        )
        backboard_bottom_px = _to_pixel(
            (backboard_x, basket_y + half_span), scale, padding
        )
        cv2.line(
            image, backboard_top_px, backboard_bottom_px,
            line_color.as_bgr(), max(2, 2 * line_thickness)
        )

    return image


def draw_made_and_miss_on_court(
    config: CourtConfiguration,
    made_xy: Optional[np.ndarray] = None,
    miss_xy: Optional[np.ndarray] = None,
    made_thickness: Optional[int] = None,
    miss_thickness: Optional[int] = None,
    made_color: sv.Color = sv.Color.from_hex("#007A33"),
    miss_color: sv.Color = sv.Color.from_hex("#850101"),
    made_size: int = 20,
    miss_size: int = 20,
    scale: float = 20,
    padding: int = 50,
    line_thickness: int = 6,
    court: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Draw made shots as circle outlines and missed shots as crosses."""
    if court is None:
        court = draw_court(
            config=config,
            scale=scale,
            padding=padding,
            line_thickness=line_thickness,
        )

    made_stroke = (
        made_thickness if made_thickness is not None else line_thickness
    )
    missed_stroke = (
        miss_thickness if miss_thickness is not None else line_thickness
    )

    def point_to_pixel(point: Tuple[float, float]) -> Tuple[int, int]:
        return _to_pixel(point, scale=scale, padding=padding)

    # Normalize inputs to iterable collections
    made_iter = (
        np.atleast_2d(made_xy) if made_xy is not None and made_xy.size > 0 else ()
    )
    miss_iter = (
        np.atleast_2d(miss_xy) if miss_xy is not None and miss_xy.size > 0 else ()
    )

    # Made shots: circle border
    for point in made_iter:
        center_x, center_y = point_to_pixel(tuple(point))
        cv2.circle(
            img=court,
            center=(center_x, center_y),
            radius=made_size,
            color=made_color.as_bgr(),
            thickness=made_stroke,
        )

    # Missed shots: cross
    for point in miss_iter:
        center_x, center_y = point_to_pixel(tuple(point))
        x0, y0 = center_x - miss_size, center_y - miss_size
        x1, y1 = center_x + miss_size, center_y + miss_size
        cv2.line(court, (x0, y0), (x1, y1), miss_color.as_bgr(), missed_stroke)
        cv2.line(court, (x0, y1), (x1, y0), miss_color.as_bgr(), missed_stroke)

    return court


def draw_points_on_court(
    config: CourtConfiguration,
    xy: Optional[np.ndarray] = None,
    labels: Optional[list[str]] = None,
    fill_color: Optional[sv.Color] = sv.Color.BLACK,
    text_color: sv.Color = sv.Color.WHITE,
    edge_color: Optional[sv.Color] = sv.Color.WHITE,
    size: int = 30,
    edge_thickness: Optional[int] = None,
    scale: float = 20,
    padding: int = 50,
    line_thickness: int = 6,
    court: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Draw points on the court.
    Points render as circles with optional fill, edge, and center labels.
    """
    if court is None:
        court = draw_court(
            config=config,
            scale=scale,
            padding=padding,
            line_thickness=line_thickness,
        )

    if xy is None or np.size(xy) == 0:
        return court

    pts = np.atleast_2d(xy)
    n = pts.shape[0]

    labels = labels if labels is not None else [None] * n
    if len(labels) < n:
        labels = list(labels) + [None] * (n - len(labels))

    stroke = edge_thickness if edge_thickness is not None else max(2, line_thickness // 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, size / 28.0)
    font_thickness = max(1, size // 8)

    for i in range(n):
        cx, cy = _to_pixel(tuple(pts[i]), scale=scale, padding=padding)

        # Face (fill)
        if fill_color is not None:
            cv2.circle(
                img=court,
                center=(cx, cy),
                radius=size,
                color=fill_color.as_bgr(),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

        # Edge (outline)
        if edge_color is not None and stroke > 0:
            cv2.circle(
                img=court,
                center=(cx, cy),
                radius=size,
                color=edge_color.as_bgr(),
                thickness=stroke,
                lineType=cv2.LINE_AA,
            )

        # Label
        label = labels[i]
        if label is not None and str(label) != "":
            text = str(label)
            (tw, th), base = cv2.getTextSize(text, font, font_scale, font_thickness)
            tx = int(cx - tw / 2)
            ty = int(cy + th / 2)
            cv2.putText(
                img=court,
                text=text,
                org=(tx, ty),
                fontFace=font,
                fontScale=font_scale,
                color=text_color.as_bgr(),
                thickness=font_thickness,
                lineType=cv2.LINE_AA,
            )

    return court


def draw_paths_on_court(
    config: CourtConfiguration,
    paths: List[np.ndarray],
    color: Optional[sv.Color] = sv.Color.BLACK,
    thickness: Optional[int] = None,
    scale: float = 20,
    padding: int = 50,
    line_thickness: int = 6,
    court: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Draw time-ordered paths as polylines in court coordinates.
    Each path is an array of shape (T, 2) with x, y in court units.
    NaN rows split a path into multiple segments.
    """
    if court is None:
        court = draw_court(
            config=config,
            scale=scale,
            padding=padding,
            line_thickness=line_thickness,
        )

    if not paths or color is None:
        return court

    stroke = thickness if thickness is not None else line_thickness
    bgr = color.as_bgr()

    def to_segments(pts: np.ndarray) -> list[np.ndarray]:
        pts = np.atleast_2d(pts).astype(float)
        segments = []
        cur = []
        for p in pts:
            if np.isnan(p).any():
                if len(cur) > 0:
                    segments.append(np.asarray(cur, dtype=float))
                    cur = []
            else:
                cur.append(p)
        if len(cur) > 0:
            segments.append(np.asarray(cur, dtype=float))
        return segments

    for path in paths:
        if path is None or np.size(path) == 0:
            continue

        for seg in to_segments(path):
            if seg.shape[0] >= 2:
                poly = np.array(
                    [[_to_pixel((float(x), float(y)), scale, padding) for x, y in seg]],
                    dtype=np.int32,
                )
                cv2.polylines(
                    img=court,
                    pts=poly,
                    isClosed=False,
                    color=bgr,
                    thickness=stroke,
                    lineType=cv2.LINE_AA,
                )
            elif seg.shape[0] == 1:
                cx, cy = _to_pixel((float(seg[0, 0]), float(seg[0, 1])), scale, padding)
                cv2.circle(
                    img=court,
                    center=(cx, cy),
                    radius=max(1, stroke // 2),
                    color=bgr,
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )

    return court
