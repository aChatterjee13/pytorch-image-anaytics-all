import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)
