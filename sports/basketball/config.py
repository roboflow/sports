from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_CEILING
from enum import Enum
from typing import Dict, List, Tuple

from sports.common.core import MeasurementUnit

CENTIMETERS_PER_FOOT = Decimal("30.48")


class League(Enum):
    NBA = "nba"
    FIBA = "fiba"


# presets stored in centimeters
PRESETS_CENTIMETERS: Dict[League, Dict[str, int]] = {
    League.NBA: dict(
        court_width=1524,
        court_length=2865,
        three_point_arc_radius=724,
        straight_section_three_point_line=424,
        sideline_to_three_point_line=91,
        paint_width=488,
        paint_length=579,
        free_throw_line_distance=457,
        center_circle_radius=183,
        restricted_area_radius=122,
        rim_diameter=46,
        baseline_to_rim_center=160,
        baseline_to_throw_line=835,
        baseline_to_throw_line_length=50,
    ),
    League.FIBA: dict(
        court_width=1500,
        court_length=2800,
        three_point_arc_radius=675,
        straight_section_three_point_line=330,
        sideline_to_three_point_line=90,
        paint_width=490,
        paint_length=580,
        free_throw_line_distance=460,
        center_circle_radius=180,
        restricted_area_radius=125,
        rim_diameter=45,
        baseline_to_rim_center=157,
        baseline_to_throw_line=830,
        baseline_to_throw_line_length=50,
    ),
}


@dataclass
class CourtConfiguration:
    """Configure basketball court dimensions for NBA or FIBA leagues.
    Provides court measurements in centimeters or feet with proper unit
    conversion and vertex/edge data for court visualization.

    Args:
        league: The basketball league standard to use (`League.NBA` or
            `League.FIBA`).
        measurement_unit: Output unit for measurements
            (`MeasurementUnit.CENTIMETERS` or `MeasurementUnit.FEET`).

    Examples:
        ```
        from sports import MeasurementUnit
        from sports.basketball import CourtConfiguration, League

        # Create NBA court configuration in centimeters
        nba_config = BasketballCourtConfiguration(
            league=League.NBA,
            measurement_unit=MeasurementUnit.CENTIMETERS
        )
        print(f"Court width: {nba_config.court_width} cm")
        # Court width: 1524.0 cm
        print(f"Paint width: {nba_config.paint_width} cm")
        # Paint width: 488.0 cm
        ```
    """
    league: League
    measurement_unit: MeasurementUnit = MeasurementUnit.CENTIMETERS

    # internal values in centimeters
    _court_width_in_centimeters: int = field(init=False)
    _court_length_in_centimeters: int = field(init=False)
    _three_point_arc_radius_in_centimeters: int = field(init=False)
    _straight_section_three_point_line_in_centimeters: int = field(init=False)
    _sideline_to_three_point_line_in_centimeters: int = field(init=False)
    _paint_width_in_centimeters: int = field(init=False)
    _paint_length_in_centimeters: int = field(init=False)
    _free_throw_line_distance_in_centimeters: int = field(init=False)
    _center_circle_radius_in_centimeters: int = field(init=False)
    _restricted_area_radius_in_centimeters: int = field(init=False)
    _rim_diameter_in_centimeters: int = field(init=False)
    _baseline_to_rim_center_in_centimeters: int = field(init=False)
    _baseline_to_throw_line_in_centimeters: int = field(init=False)
    _baseline_to_throw_line_length_in_centimeters: int = field(init=False)

    def __post_init__(self) -> None:
        preset = PRESETS_CENTIMETERS[self.league]
        self._court_width_in_centimeters = preset["court_width"]
        self._court_length_in_centimeters = preset["court_length"]
        self._three_point_arc_radius_in_centimeters = preset["three_point_arc_radius"]
        self._straight_section_three_point_line_in_centimeters = preset[
            "straight_section_three_point_line"
        ]
        self._sideline_to_three_point_line_in_centimeters = preset[
            "sideline_to_three_point_line"
        ]
        self._paint_width_in_centimeters = preset["paint_width"]
        self._paint_length_in_centimeters = preset["paint_length"]
        self._free_throw_line_distance_in_centimeters = preset[
            "free_throw_line_distance"
        ]
        self._center_circle_radius_in_centimeters = preset["center_circle_radius"]
        self._restricted_area_radius_in_centimeters = preset[
            "restricted_area_radius"
        ]
        self._rim_diameter_in_centimeters = preset["rim_diameter"]
        self._baseline_to_rim_center_in_centimeters = preset["baseline_to_rim_center"]
        self._baseline_to_throw_line_in_centimeters = preset["baseline_to_throw_line"]
        self._baseline_to_throw_line_length_in_centimeters = preset[
            "baseline_to_throw_line_length"
        ]

    # conversion helpers
    def _to_output_unit_rounded_up(self, value_in_centimeters: float) -> float:
        value = Decimal(str(value_in_centimeters))
        if self.measurement_unit == MeasurementUnit.FEET:
            value = value / CENTIMETERS_PER_FOOT
        return float(value.quantize(Decimal("0.01"), rounding=ROUND_CEILING))

    # public properties in the selected unit
    @property
    def court_width(self) -> float:
        """Get the court width in the configured measurement unit.

        Returns:
            `float`: Court width in centimeters or feet based on
                `measurement_unit` setting.
        """
        return self._to_output_unit_rounded_up(self._court_width_in_centimeters)

    @property
    def court_length(self) -> float:
        """Get the court length in the configured measurement unit.

            Returns:
                `float`: Court length in centimeters or feet based on
                    `measurement_unit` setting.
            """
        return self._to_output_unit_rounded_up(self._court_length_in_centimeters)

    @property
    def three_point_arc_radius(self) -> float:
        """Get the three-point arc radius in the configured measurement unit.

            Returns:
                `float`: Three-point arc radius in centimeters or feet based on
                    `measurement_unit` setting.
            """
        return self._to_output_unit_rounded_up(
            self._three_point_arc_radius_in_centimeters
        )

    @property
    def straight_section_three_point_line(self) -> float:
        """Get the straight section length of three-point line.

            Returns:
                `float`: Straight section length in centimeters or feet based on
                    `measurement_unit` setting.
            """
        return self._to_output_unit_rounded_up(
            self._straight_section_three_point_line_in_centimeters
        )

    @property
    def sideline_to_three_point_line(self) -> float:
        """Get the distance from sideline to three-point line corner.

            Returns:
                `float`: Sideline to three-point distance in centimeters or feet
                    based on `measurement_unit` setting.
            """
        return self._to_output_unit_rounded_up(
            self._sideline_to_three_point_line_in_centimeters
        )

    @property
    def paint_width(self) -> float:
        """Get the paint area width in the configured measurement unit.

            Returns:
                `float`: Paint width in centimeters or feet based on
                    `measurement_unit` setting.
            """
        return self._to_output_unit_rounded_up(self._paint_width_in_centimeters)

    @property
    def paint_length(self) -> float:
        """Get the paint area length in the configured measurement unit.

            Returns:
                `float`: Paint length in centimeters or feet based on
                    `measurement_unit` setting.
            """
        return self._to_output_unit_rounded_up(self._paint_length_in_centimeters)

    @property
    def free_throw_line_distance(self) -> float:
        """Get the free throw line distance from baseline.

            Returns:
                `float`: Free throw line distance in centimeters or feet based
                    on `measurement_unit` setting.
            """
        return self._to_output_unit_rounded_up(
            self._free_throw_line_distance_in_centimeters
        )

    @property
    def center_circle_radius(self) -> float:
        """Get the center circle radius in the configured measurement unit.

            Returns:
                `float`: Center circle radius in centimeters or feet based on
                    `measurement_unit` setting.
            """
        return self._to_output_unit_rounded_up(
            self._center_circle_radius_in_centimeters
        )

    @property
    def restricted_area_radius(self) -> float:
        """Get the restricted area radius in the configured measurement unit.

            Returns:
                `float`: Restricted area radius in centimeters or feet based on
                    `measurement_unit` setting.
            """
        return self._to_output_unit_rounded_up(
            self._restricted_area_radius_in_centimeters
        )

    @property
    def rim_diameter(self) -> float:
        """Get the rim diameter in the configured measurement unit.

            Returns:
                `float`: Rim diameter in centimeters or feet based on
                    `measurement_unit` setting.
            """
        return self._to_output_unit_rounded_up(self._rim_diameter_in_centimeters)

    @property
    def baseline_to_rim_center(self) -> float:
        """Get the distance from baseline to rim center.

            Returns:
                `float`: Baseline to rim distance in centimeters or feet based
                    on `measurement_unit` setting.
            """
        return self._to_output_unit_rounded_up(
            self._baseline_to_rim_center_in_centimeters
        )

    @property
    def baseline_to_throw_line(self) -> float:
        """Get the distance from baseline to throw-in line position.

            Returns:
                `float`: Baseline to throw-in line distance in centimeters or
                    feet based on `measurement_unit` setting.
            """

        return self._to_output_unit_rounded_up(
            self._baseline_to_throw_line_in_centimeters
        )

    @property
    def baseline_to_throw_line_length(self) -> float:
        """Get the length of throw-in line markers.

            Returns:
                `float`: Throw-in line marker length in centimeters or feet
                    based on `measurement_unit` setting.
            """
        return self._to_output_unit_rounded_up(
            self._baseline_to_throw_line_length_in_centimeters
        )

    # internals for geometry in centimeters
    @property
    def _paint_start_in_centimeters(self) -> int:
        return (
            self._court_width_in_centimeters - self._paint_width_in_centimeters
        ) // 2

    def _raw_vertices_centimeters(self) -> List[Tuple[int, int]]:
        paint_start = self._paint_start_in_centimeters
        middle_court_width = self._court_width_in_centimeters // 2
        court_width = self._court_width_in_centimeters
        court_length = self._court_length_in_centimeters

        return [
            (0, 0),  # 00
            (0, self._sideline_to_three_point_line_in_centimeters),  # 01
            (0, paint_start),  # 02
            (0, paint_start + self._paint_width_in_centimeters),  # 03
            (0, court_width - self._sideline_to_three_point_line_in_centimeters),  # 04
            (0, court_width),  # 05
            (self._baseline_to_rim_center_in_centimeters, middle_court_width),  # 06
            (
                self._straight_section_three_point_line_in_centimeters,
                self._sideline_to_three_point_line_in_centimeters,
            ),  # 07
            (
                self._straight_section_three_point_line_in_centimeters,
                court_width - self._sideline_to_three_point_line_in_centimeters,
            ),  # 08
            (self._paint_length_in_centimeters, paint_start),  # 09
            (
                self._paint_length_in_centimeters,
                paint_start + self._paint_width_in_centimeters // 2,
            ),  # 10
            (
                self._paint_length_in_centimeters,
                paint_start + self._paint_width_in_centimeters,
            ),  # 11
            (self._baseline_to_throw_line_in_centimeters, 0),  # 12
            (
                self._baseline_to_rim_center_in_centimeters
                + self._three_point_arc_radius_in_centimeters,
                middle_court_width,
            ),  # 13
            (self._baseline_to_throw_line_in_centimeters, court_width),  # 14
            (court_length // 2, 0),  # 15
            (court_length // 2, middle_court_width),  # 16
            (court_length // 2, court_width),  # 17
            (court_length - self._baseline_to_throw_line_in_centimeters, 0),  # 18
            (
                court_length
                - self._baseline_to_rim_center_in_centimeters
                - self._three_point_arc_radius_in_centimeters,
                middle_court_width,
            ),  # 19
            (court_length - self._baseline_to_throw_line_in_centimeters, court_width),  # 20
            (court_length - self._paint_length_in_centimeters, paint_start),  # 21
            (
                court_length - self._paint_length_in_centimeters,
                paint_start + self._paint_width_in_centimeters // 2,
            ),  # 22
            (
                court_length - self._paint_length_in_centimeters,
                paint_start + self._paint_width_in_centimeters,
            ),  # 23
            (
                court_length - self._straight_section_three_point_line_in_centimeters,
                self._sideline_to_three_point_line_in_centimeters,
            ),  # 24
            (
                court_length - self._straight_section_three_point_line_in_centimeters,
                court_width - self._sideline_to_three_point_line_in_centimeters,
            ),  # 25
            (court_length - self._baseline_to_rim_center_in_centimeters, middle_court_width),  # 26
            (court_length, 0),  # 27
            (court_length, self._sideline_to_three_point_line_in_centimeters),  # 28
            (court_length, paint_start),  # 29
            (court_length, paint_start + self._paint_width_in_centimeters),  # 30
            (court_length, court_width - self._sideline_to_three_point_line_in_centimeters),  # 31
            (court_length, court_width),  # 32
        ]

    def _vertices_in_unit(self) -> List[Tuple[float, float]]:
        return [
            (
                self._to_output_unit_rounded_up(x),
                self._to_output_unit_rounded_up(y),
            )
            for x, y in self._raw_vertices_centimeters()
        ]

    @property
    def vertices(self) -> List[Tuple[float, float]]:
        """Get all court vertices in the configured measurement unit.
            Returns a list of coordinate pairs representing key points on the
            basketball court for geometry calculations and visualization.

            Returns:
                `List[Tuple[float, float]]`: List of (x, y) coordinate pairs
                    in centimeters or feet based on `measurement_unit` setting.
            """
        return self._vertices_in_unit()

    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        # left sideline
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
        # left paint
        (2, 9), (11, 3), (9, 10), (10, 11),
        # left three point borders
        (1, 4),
        # top border
        (0, 12), (12, 15), (15, 18),
        # center line
        (15, 16), (16, 17),
        # bottom border
        (5, 14), (14, 17), (17, 20),
        # right sideline
        (27, 28), (28, 29), (29, 30), (30, 31), (31, 32),
        # right three point borders
        (28, 31),
        # right paint
        (29, 21), (21, 22), (22, 23), (23, 30),
    ])

    labels: List[str] = field(default_factory=lambda: [
        "01","02","04","05","07","08","09","10","11","12","13","14",
        "15","16","17","19","21","23","25","26","27","28","29","30",
        "31","32","33","34","35","37","38","40","41"
    ])

    colors: List[str] = field(default_factory=lambda: [
        "#A4F84B","#FF1493","#FF1493","#FF1493","#FF1493","#00BFFF",
        "#FF1493","#FF1493","#FF1493","#FF1493","#FF1493","#FF1493",
        "#A4F84B","#FF1493","#00BFFF","#A4F84B","#A849F1","#00BFFF",
        "#A4F84B","#52F8C4","#00BFFF","#52F8C4","#52F8C4","#52F8C4",
        "#52F8C4","#52F8C4","#52F8C4","#A4F84B","#52F8C4","#52F8C4",
        "#52F8C4","#52F8C4","#00BFFF"
    ])

    # direct index getters
    @property
    def left_paint_indexes(self) -> List[int]:
        """Get vertex indexes defining the left paint area.
        Returns the indexes into the vertices list that form the
        rectangular paint area on the left side of the court.

        Returns:
            `List[int]`: List of vertex indexes forming left paint polygon
                in order for drawing operations.
        """
        return [2, 3, 11, 9]

    @property
    def right_paint_indexes(self) -> List[int]:
        """Get vertex indexes defining the right paint area.
            Returns the indexes into the vertices list that form the
            rectangular paint area on the right side of the court.

            Returns:
                `List[int]`: List of vertex indexes forming right paint polygon
                    in order for drawing operations.
            """
        return [29, 30, 23, 21]

    @property
    def left_basket_index(self) -> int:
        """Get vertex index for the left basket center position.
            Returns the index into the vertices list that represents the
            center point of the left basketball rim.

            Returns:
                `int`: Vertex index for left basket center coordinates.
        """
        return 6

    @property
    def right_basket_index(self) -> int:
        """Get vertex index for the right basket center position.
            Returns the index into the vertices list that represents the
            center point of the right basketball rim.

            Returns:
                `int`: Vertex index for right basket center coordinates.
            """
        return 26

    @property
    def court_corner_indexes(self) -> List[int]:
        """Get vertex indexes for the four court corners.
            Returns the indexes into the vertices list that represent the
            four corner points of the rectangular court boundary.

            Returns:
                `List[int]`: List of vertex indexes for court corners in order:
                    bottom-left, top-left, top-right, bottom-right.
            """
        return [0, 5, 32, 27]
