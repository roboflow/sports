"""Indoor volleyball court geometry, expressed in centimeters."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import List, Tuple


@dataclass
class VolleyballCourtConfiguration:
    """Describe the boundary and line landmarks of a rectangular volleyball court.

    Coordinates are centimeters, with the length along the x-axis and width
    along the y-axis. The first four vertices follow the perimeter from (0, 0).
    Remaining vertices describe the net and the two attack lines. Edge indices
    are one-based, matching the soccer configuration convention.

    Args:
        length: Court length in centimeters. Defaults to 1800.
        width: Court width in centimeters. Defaults to 900.
        attack_line_distance: Distance from the net to either attack line in
            centimeters. Must be less than half the length. Defaults to 300.

    Attributes:
        length: Finite positive court length in centimeters.
        width: Finite positive court width in centimeters.
        attack_line_distance: Finite positive attack-line distance in centimeters.

    Raises:
        ValueError: If dimensions are nonnumeric, nonfinite, nonpositive, or the
            attack-line distance reaches or exceeds half the court length.

    Example:
        >>> config = VolleyballCourtConfiguration()
        >>> config.vertices[:4]
        [(0, 0), (1800, 0), (1800, 900), (0, 900)]
    """

    length: float = 1800
    width: float = 900
    attack_line_distance: float = 300

    def __post_init__(self) -> None:
        """Validate dimensions when the dataclass is constructed."""
        self.validate()

    def validate(self) -> None:
        """Check court dimensions, including after an attribute is edited.

        Raises:
            ValueError: If a dimension is not a finite positive number, or the
                attack-line distance is at least half the length. Boolean values
                do not represent dimensions.
        """
        for name in ("length", "width", "attack_line_distance"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number.")
        if self.attack_line_distance >= self.length / 2:
            raise ValueError("attack_line_distance must be less than half the length.")

    @property
    def vertices(self) -> List[Tuple[float, float]]:
        """Return perimeter, net, and attack-line landmarks in court coordinates.

        Returns:
            Ten (x, y) pairs in centimeters. The first four are (0, 0),
            (length, 0), (length, width), and (0, width). The remaining pairs
            are the two endpoints of the net and each attack line.
        """
        middle = self.length / 2
        near_attack = middle - self.attack_line_distance
        far_attack = middle + self.attack_line_distance
        return [
            (0, 0),
            (self.length, 0),
            (self.length, self.width),
            (0, self.width),
            (middle, 0),
            (middle, self.width),
            (near_attack, 0),
            (near_attack, self.width),
            (far_attack, 0),
            (far_attack, self.width),
        ]

    @property
    def edges(self) -> List[Tuple[int, int]]:
        """Return the court lines as pairs of one-based landmark indices.

        Returns:
            Four boundary edges followed by the net and two attack-line edges.
        """
        return [(1, 2), (2, 3), (3, 4), (4, 1), (5, 6), (7, 8), (9, 10)]

    @property
    def labels(self) -> List[str]:
        """Return stable two-digit labels in the same order as the landmarks.

        Returns:
            Labels from "01" through "10", one for each vertex.
        """
        return [f"{index:02d}" for index in range(1, len(self.vertices) + 1)]
