from dataclasses import dataclass, field
from math import sqrt
from typing import List, Tuple


@dataclass
class HandballCourtConfiguration:
    length: int = 4000  # [cm]
    width: int = 2000  # [cm]
    goal_width: int = 300  # [cm]
    goal_depth: int = 100  # [cm]
    goal_area_radius: int = 600  # [cm]
    free_throw_radius: int = 900  # [cm]
    seven_meter_line_distance: int = 700  # [cm]
    seven_meter_line_length: int = 100  # [cm]
    goalkeeper_restraining_line_distance: int = 400  # [cm]
    goalkeeper_restraining_line_length: int = 15  # [cm]
    throw_off_area_radius: int = 200  # [cm]
    free_throw_line_segment_length: int = 15  # [cm]
    free_throw_line_gap_length: int = 15  # [cm]
    substitution_line_distance: int = 450  # [cm]
    substitution_line_length: int = 30  # [cm]

    @property
    def goal_top_y(self) -> float:
        return (self.width - self.goal_width) / 2

    @property
    def goal_bottom_y(self) -> float:
        return (self.width + self.goal_width) / 2

    @property
    def center_y(self) -> float:
        return self.width / 2

    @property
    def center_x(self) -> float:
        return self.length / 2

    @property
    def free_throw_sideline_x_offset(self) -> float:
        vertical_distance = self.goal_top_y
        return sqrt(self.free_throw_radius ** 2 - vertical_distance ** 2)

    @property
    def vertices(self) -> List[Tuple[float, float]]:
        seven_meter_half_length = self.seven_meter_line_length / 2
        goalkeeper_line_half_length = self.goalkeeper_restraining_line_length / 2
        free_throw_x = self.free_throw_sideline_x_offset

        return [
            (0, 0),  # 1
            (self.center_x, 0),  # 2
            (self.length, 0),  # 3
            (self.length, self.width),  # 4
            (self.center_x, self.width),  # 5
            (0, self.width),  # 6
            (0, self.goal_top_y),  # 7
            (0, self.goal_bottom_y),  # 8
            (self.length, self.goal_top_y),  # 9
            (self.length, self.goal_bottom_y),  # 10
            (0, self.goal_top_y - self.goal_area_radius),  # 11
            (self.goal_area_radius, self.goal_top_y),  # 12
            (self.goal_area_radius, self.goal_bottom_y),  # 13
            (0, self.goal_bottom_y + self.goal_area_radius),  # 14
            (self.length, self.goal_top_y - self.goal_area_radius),  # 15
            (self.length - self.goal_area_radius, self.goal_top_y),  # 16
            (self.length - self.goal_area_radius, self.goal_bottom_y),  # 17
            (self.length, self.goal_bottom_y + self.goal_area_radius),  # 18
            (free_throw_x, 0),  # 19
            (self.free_throw_radius, self.goal_top_y),  # 20
            (self.free_throw_radius, self.goal_bottom_y),  # 21
            (free_throw_x, self.width),  # 22
            (self.length - free_throw_x, 0),  # 23
            (self.length - self.free_throw_radius, self.goal_top_y),  # 24
            (self.length - self.free_throw_radius, self.goal_bottom_y),  # 25
            (self.length - free_throw_x, self.width),  # 26
            (
                self.seven_meter_line_distance,
                self.center_y - seven_meter_half_length,
            ),  # 27
            (
                self.seven_meter_line_distance,
                self.center_y + seven_meter_half_length,
            ),  # 28
            (
                self.length - self.seven_meter_line_distance,
                self.center_y - seven_meter_half_length,
            ),  # 29
            (
                self.length - self.seven_meter_line_distance,
                self.center_y + seven_meter_half_length,
            ),  # 30
            (
                self.goalkeeper_restraining_line_distance,
                self.center_y - goalkeeper_line_half_length,
            ),  # 31
            (
                self.goalkeeper_restraining_line_distance,
                self.center_y + goalkeeper_line_half_length,
            ),  # 32
            (
                self.length - self.goalkeeper_restraining_line_distance,
                self.center_y - goalkeeper_line_half_length,
            ),  # 33
            (
                self.length - self.goalkeeper_restraining_line_distance,
                self.center_y + goalkeeper_line_half_length,
            ),  # 34
            (self.center_x, self.center_y),  # 35
            (self.center_x - self.substitution_line_distance, 0),  # 36
            (self.center_x + self.substitution_line_distance, 0),  # 37
            (self.center_x - self.substitution_line_distance, self.width),  # 38
            (self.center_x + self.substitution_line_distance, self.width),  # 39
        ]

    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1),
        (2, 5),
        (12, 13), (16, 17),
        (27, 28), (29, 30),
        (31, 32), (33, 34),
    ])

    labels: List[str] = field(default_factory=lambda: [
        "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
        "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
        "21", "22", "23", "24", "25", "26", "27", "28", "29", "30",
        "31", "32", "33", "34", "35", "36", "37", "38", "39",
    ])

    colors: List[str] = field(default_factory=lambda: [
        "#FF1493", "#00BFFF", "#FF1493", "#FF1493", "#00BFFF",
        "#FF1493", "#A4F84B", "#A4F84B", "#A4F84B", "#A4F84B",
        "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347",
        "#FF6347", "#FF6347", "#FF6347", "#FFD700", "#FFD700",
        "#FFD700", "#FFD700", "#FFD700", "#FFD700", "#FFD700",
        "#FFD700", "#52F8C4", "#52F8C4", "#52F8C4", "#52F8C4",
        "#A849F1", "#A849F1", "#A849F1", "#A849F1", "#00BFFF",
        "#FFFFFF", "#FFFFFF", "#FFFFFF", "#FFFFFF",
    ])

    @property
    def court_corner_indexes(self) -> List[int]:
        return [1, 3, 4, 6]

    @property
    def left_goal_indexes(self) -> List[int]:
        return [7, 8]

    @property
    def right_goal_indexes(self) -> List[int]:
        return [9, 10]

    @property
    def left_goal_area_indexes(self) -> List[int]:
        return [11, 12, 13, 14]

    @property
    def right_goal_area_indexes(self) -> List[int]:
        return [15, 16, 17, 18]


CourtConfiguration = HandballCourtConfiguration
