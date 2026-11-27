"""Tests for the soccer pitch configuration."""

from sports.configs.soccer import SoccerPitchConfiguration


def test_pitch_labels_follow_vertex_order():
    """Match every pitch keypoint label to its numbered vertex."""
    config = SoccerPitchConfiguration()

    assert len(config.vertices) == 32
    assert config.labels == [f"{number:02d}" for number in range(1, 33)]
