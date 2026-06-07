"""Multi-scale anchor generation over feature pyramid levels."""

from __future__ import annotations

import math

import torch


def retinanet_sizes(
    base_sizes: tuple[int, ...] = (32, 64, 128, 256, 512),
    scales_per_octave: int = 3,
) -> tuple[tuple[float, ...], ...]:
    """RetinaNet-style per-level sizes: base * 2^(i/octaves) for i in 0..octaves-1."""
    return tuple(
        tuple(base * 2 ** (i / scales_per_octave) for i in range(scales_per_octave))
        for base in base_sizes
    )


class AnchorGenerator:
    """Generate anchors for a list of pyramid levels.

    Args:
        sizes: per-level tuple of anchor sizes (sqrt of area), one inner tuple
            per pyramid level.
        aspect_ratios: h/w ratios shared across levels.
        offset: anchor center offset in stride units (0.5 = pixel-center
            aligned).

    Anchors for one level are laid out as (H * W * A, 4) XYXY, grid-major then
    anchor-major — matching head outputs reshaped via
    ``(B, A*K, H, W) -> (B, H*W*A, K)``.
    """

    def __init__(
        self,
        sizes: tuple[tuple[float, ...], ...] = ((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios: tuple[float, ...] = (0.5, 1.0, 2.0),
        offset: float = 0.5,
    ) -> None:
        self.sizes = tuple(tuple(level) for level in sizes)
        self.aspect_ratios = tuple(aspect_ratios)
        self.offset = offset
        self._base_anchors = [self._make_base_anchors(level) for level in self.sizes]

    @property
    def num_levels(self) -> int:
        return len(self.sizes)

    @property
    def num_anchors_per_location(self) -> int:
        counts = {len(level) * len(self.aspect_ratios) for level in self.sizes}
        if len(counts) != 1:
            raise ValueError(
                "All levels must have the same number of anchors per location; "
                f"got sizes {self.sizes} x ratios {self.aspect_ratios}"
            )
        return counts.pop()

    def _make_base_anchors(self, level_sizes: tuple[float, ...]) -> torch.Tensor:
        """Zero-centered (A, 4) anchors for one level."""
        anchors = []
        for size in level_sizes:
            area = float(size) ** 2
            for ratio in self.aspect_ratios:  # ratio = h / w
                w = math.sqrt(area / ratio)
                h = w * ratio
                anchors.append([-w / 2, -h / 2, w / 2, h / 2])
        return torch.tensor(anchors, dtype=torch.float32)

    def __call__(
        self,
        feature_shapes: list[tuple[int, int]],
        strides: list[int],
        device: torch.device | str = "cpu",
    ) -> list[torch.Tensor]:
        """Return per-level anchors, each (H*W*A, 4) XYXY in image coordinates."""
        if len(feature_shapes) != self.num_levels or len(strides) != self.num_levels:
            raise ValueError(
                f"Expected {self.num_levels} feature shapes/strides, got "
                f"{len(feature_shapes)}/{len(strides)}"
            )
        anchors_per_level = []
        for (h, w), stride, base in zip(feature_shapes, strides, self._base_anchors):
            base = base.to(device)
            shift_y = (torch.arange(h, device=device) + self.offset) * stride
            shift_x = (torch.arange(w, device=device) + self.offset) * stride
            cy, cx = torch.meshgrid(shift_y, shift_x, indexing="ij")
            shifts = torch.stack(
                [cx.reshape(-1), cy.reshape(-1), cx.reshape(-1), cy.reshape(-1)], dim=1
            )
            # (HW, 1, 4) + (1, A, 4) -> (HW, A, 4) -> (HW*A, 4): grid-major
            anchors = (shifts[:, None, :] + base[None, :, :]).reshape(-1, 4)
            anchors_per_level.append(anchors)
        return anchors_per_level
