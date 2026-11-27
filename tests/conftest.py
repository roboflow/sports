"""Shared pytest fixtures for the whole test tree."""

import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def reset_random_seeds() -> None:
    """Seed every RNG source so clustering/embedding tests are repeatable."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
