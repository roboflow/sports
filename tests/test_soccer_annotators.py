import numpy as np
import supervision as sv

from sports.annotators.soccer import draw_paths_on_pitch
from sports.configs.soccer import SoccerPitchConfiguration

CONFIG = SoccerPitchConfiguration()
SCALE = 0.1
PADDING = 50
RED_BGR = np.array(sv.Color.RED.as_bgr(), dtype=np.uint8)


def _blank_pitch() -> np.ndarray:
    height = int(CONFIG.width * SCALE) + 2 * PADDING
    width = int(CONFIG.length * SCALE) + 2 * PADDING
    return np.zeros((height, width, 3), dtype=np.uint8)


def _to_pixel(x_cm: float, y_cm: float) -> tuple:
    return int(x_cm * SCALE) + PADDING, int(y_cm * SCALE) + PADDING


def test_draw_paths_on_pitch_draws_every_path() -> None:
    paths = [
        np.array([[1000, 1000], [3000, 1000]]),
        np.array([[1000, 5000], [3000, 5000]]),
        np.array([[8000, 3000], [10000, 3000]]),
    ]

    result = draw_paths_on_pitch(
        config=CONFIG,
        paths=paths,
        color=sv.Color.RED,
        padding=PADDING,
        scale=SCALE,
        pitch=_blank_pitch(),
    )

    for path in paths:
        x, y = _to_pixel(*path.mean(axis=0))
        assert np.array_equal(result[y, x], RED_BGR), (
            f"path {path.tolist()} was not drawn"
        )


def test_draw_paths_on_pitch_skips_short_paths_and_keeps_drawing() -> None:
    paths = [
        np.array([[1000, 1000]]),
        np.array([[1000, 5000], [3000, 5000]]),
    ]

    result = draw_paths_on_pitch(
        config=CONFIG,
        paths=paths,
        color=sv.Color.RED,
        padding=PADDING,
        scale=SCALE,
        pitch=_blank_pitch(),
    )

    x, y = _to_pixel(2000, 5000)
    assert np.array_equal(result[y, x], RED_BGR)


def test_draw_paths_on_pitch_with_no_paths_returns_pitch() -> None:
    pitch = _blank_pitch()

    result = draw_paths_on_pitch(config=CONFIG, paths=[], pitch=pitch)

    assert result is pitch
