"""Black-box tests for ``create_batches`` and ``TeamClassifier``.

``TeamClassifier`` downloads a SigLIP vision model from the Hugging Face Hub. That
network boundary is replaced by a tiny deterministic stand-in whose embedding is
derived from a crop's mean colour. Everything else (UMAP, KMeans, batching,
ordering) runs for real, so the tests check the intent: jerseys of two colours are
split into two teams.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import sports.common.team as team_module
from sports.common.team import TeamClassifier, create_batches


class TestCreateBatches:
    """``create_batches`` slices any iterable into consecutive lists."""

    @pytest.mark.parametrize("n, batch_size, expected_sizes", [
        pytest.param(7, 3, [3, 3, 1], id="remainder-batch"),
        pytest.param(6, 3, [3, 3], id="exact-multiple-has-no-empty-tail"),
        pytest.param(2, 5, [2], id="batch-larger-than-input"),
        pytest.param(4, 1, [1, 1, 1, 1], id="batch-of-one"),
        pytest.param(0, 3, [], id="empty-input"),
    ])
    def test_batch_sizes(self, n, batch_size, expected_sizes):
        """All batches are full except possibly the last; empty input gives none."""
        batches = list(create_batches(range(n), batch_size))
        assert [len(b) for b in batches] == expected_sizes

    def test_keeps_elements_and_order(self):
        """Concatenating the batches reproduces the input sequence."""
        batches = list(create_batches(list("abcdefg"), 3))
        assert batches == [["a", "b", "c"], ["d", "e", "f"], ["g"]]

    def test_accepts_one_shot_iterators(self):
        """Any iterable works, e.g. a generator consumed only once."""
        batches = list(create_batches((i * i for i in range(5)), 2))
        assert batches == [[0, 1], [4, 9], [16]]

    @pytest.mark.parametrize("batch_size", [0, -3])
    def test_non_positive_batch_size_still_yields_every_element_once(self, batch_size):
        """A bad size must not hang or drop items; nothing may be lost or repeated."""
        flat = [x for batch in create_batches(range(4), batch_size) for x in batch]
        assert flat == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Fake Hugging Face boundary
# ---------------------------------------------------------------------------

class _FakeBatch(dict):
    """Mimics the processor output: a mapping with a ``.to(device)`` method."""

    def to(self, device):
        return self


class _FakeProcessor:
    """Turns PIL crops into a (B, 3) tensor of mean RGB values in [0, 1]."""

    @classmethod
    def from_pretrained(cls, path):
        return cls()

    def __call__(self, images, return_tensors):
        means = [np.asarray(im).reshape(-1, 3).mean(axis=0) / 255.0 for im in images]
        return _FakeBatch(pixel_values=torch.tensor(np.array(means), dtype=torch.float32))


class _FakeVisionModel:
    """Maps mean colour to a (B, 4, 9) 'token' tensor, deterministic per colour."""

    @classmethod
    def from_pretrained(cls, path):
        return cls()

    def to(self, device):
        return self

    def __call__(self, pixel_values):
        feats = torch.cat([pixel_values, pixel_values ** 2, pixel_values.sqrt()], dim=1)
        return SimpleNamespace(last_hidden_state=feats[:, None, :].repeat(1, 4, 1))


@pytest.fixture(autouse=True)
def offline_model(monkeypatch):
    """Replace the Hub-downloaded SigLIP model and processor with local stand-ins."""
    monkeypatch.setattr(team_module, "SiglipVisionModel", _FakeVisionModel)
    monkeypatch.setattr(team_module, "AutoProcessor", _FakeProcessor)


def jersey(bgr: tuple, jitter: int = 0) -> np.ndarray:
    """A 16x8 BGR crop of one (slightly varied) shirt colour."""
    colour = np.clip(np.array(bgr) + jitter, 0, 255).astype(np.uint8)
    return np.full((16, 8, 3), colour, dtype=np.uint8)


RED_SHIRT = (30, 30, 200)  # BGR
BLUE_SHIRT = (200, 60, 30)  # BGR
PER_TEAM = 20


@pytest.fixture
def crops() -> list:
    """20 red-shirt crops followed by 20 blue-shirt crops, with small shade jitter."""
    reds = [jersey(RED_SHIRT, i % 7) for i in range(PER_TEAM)]
    blues = [jersey(BLUE_SHIRT, i % 7) for i in range(PER_TEAM)]
    return reds + blues


def assert_two_teams_split_at(labels: np.ndarray, boundary: int) -> None:
    """Both halves are internally uniform and carry different labels."""
    first, second = set(labels[:boundary].tolist()), set(labels[boundary:].tolist())
    assert len(first) == 1 and len(second) == 1 and first != second


class TestTeamClassifier:
    """Players are separated into two teams by shirt colour."""

    @pytest.mark.parametrize("batch_size", [1, 7, 32])
    def test_separates_two_shirt_colours_into_two_teams(self, crops, batch_size):
        """Fit on mixed crops, then predict: reds share one label, blues the other.

        The split must not depend on the embedding batch size.
        """
        classifier = TeamClassifier(batch_size=batch_size)
        classifier.fit(crops)
        labels = classifier.predict(crops)
        assert_two_teams_split_at(labels, PER_TEAM)

    def test_predicts_one_binary_label_per_crop(self, crops):
        """Output is an integer array with one entry per crop, drawn from {0, 1}."""
        classifier = TeamClassifier()
        classifier.fit(crops)
        labels = classifier.predict(crops[:5] + crops[-3:])
        assert (labels.shape, set(labels.tolist()) <= {0, 1}) == ((8,), True)

    def test_unseen_crops_get_the_label_of_their_shirt_colour(self, crops):
        """New players are assigned to the team whose shirt colour they wear."""
        unseen = [jersey(RED_SHIRT, 7), jersey(BLUE_SHIRT, 7)]
        assert all(
            not np.array_equal(sample, training_crop)
            for sample in unseen
            for training_crop in crops
        )

        classifier = TeamClassifier()
        classifier.fit(crops)
        seen_labels = classifier.predict([jersey(RED_SHIRT), jersey(BLUE_SHIRT)])
        unseen_labels = classifier.predict(unseen)
        assert seen_labels.tolist() == unseen_labels.tolist() and seen_labels[0] != seen_labels[1]

    def test_predict_on_no_crops_returns_empty_array(self, crops):
        """A frame with no players yields an empty prediction, not an error."""
        classifier = TeamClassifier()
        classifier.fit(crops)
        assert len(classifier.predict([])) == 0

    def test_features_have_one_row_per_crop_in_input_order(self):
        """Embeddings keep crop order across batch boundaries.

        With batch size 2 the three crops span two batches; identical crops must
        get identical rows and a different crop a different row.
        """
        classifier = TeamClassifier(batch_size=2)
        red, blue = jersey(RED_SHIRT), jersey(BLUE_SHIRT)
        features = classifier.extract_features([red, blue, red])
        assert features.shape[0] == 3
        np.testing.assert_allclose(features[0], features[2])
        assert not np.allclose(features[0], features[1])
