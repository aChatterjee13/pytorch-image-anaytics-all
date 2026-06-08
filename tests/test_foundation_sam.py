import pytest

from image_analytics.core.registry import MODELS


def test_sam_registered():
    import image_analytics.foundation  # noqa: F401  (registration)

    assert "sam" in MODELS


def test_sam2_gated_on_old_torch():
    """SAM 2 needs torch>=2.3.1; on the 2.2.x ceiling it must fail clearly."""
    import torch

    from image_analytics.foundation.sam import load_sam2

    version = tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:3])
    if version >= (2, 3, 1):
        pytest.skip("torch>=2.3.1 present; SAM2 gate does not apply")

    with pytest.raises(RuntimeError, match="2.3.1"):
        load_sam2()
