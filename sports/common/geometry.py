"""Small vector helpers shared by soccer analytics."""

from __future__ import annotations

import numpy as np


def unit(vector: np.ndarray) -> np.ndarray:
    """Return the unit vector; zero vector maps to zero."""
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return np.zeros_like(vector)
    return vector / norm
