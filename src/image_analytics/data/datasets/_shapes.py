"""Shared procedural shape rasterizer for the synthetic fixtures.

Both the Phase 2 detection dataset (``synthetic_shapes``) and the Phase 3
segmentation datasets (``synthetic_shapes_seg`` semantic,
``synthetic_shapes_instance`` instance) draw from this module, so the *same
(split, index)* yields the *same image* across tasks — the drawn pixels are
simultaneously the detection box, the semantic class-index mask, and the
per-instance mask. Keeping the RNG draw order here in one place is what makes
that pixel-consistency (and determinism) hold.

Foreground classes (0-based): 0=rectangle, 1=circle, 2=triangle. Semantic
masks reserve 0 for background, so a shape with label ``L`` paints ``L + 1``.
"""

from __future__ import annotations

import torch

SHAPE_CLASSES = ("rectangle", "circle", "triangle")


def random_background(image_size: int, generator: torch.Generator) -> torch.Tensor:
    """A dark, slightly noisy (3, S, S) canvas — shapes are painted brighter
    so they stay learnable on CPU in minutes."""
    s = image_size
    image = torch.rand(3, 1, 1, generator=generator) * 0.25
    image = image.expand(3, s, s).clone()
    image += torch.randn(3, s, s, generator=generator) * 0.02
    return image.clamp(0, 1)


def draw_shape(
    image: torch.Tensor,
    label: int,
    generator: torch.Generator,
    image_size: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Draw one shape in-place into ``image`` (3, S, S).

    Returns ``(box, mask)`` where ``box`` is the tight XYXY float32 box and
    ``mask`` is the boolean (S, S) footprint of the shape, or ``(None, None)``
    if the shape rasterized to nothing. The RNG draw order (size params, then
    colour only for a non-degenerate shape) is fixed so callers stay
    deterministic and pixel-consistent across tasks.
    """
    s = image_size
    ys, xs = torch.meshgrid(
        torch.arange(s, dtype=torch.float32),
        torch.arange(s, dtype=torch.float32),
        indexing="ij",
    )

    def rint(low: int, high: int) -> int:
        return int(torch.randint(low, high, (1,), generator=generator))

    if label == 0:  # rectangle
        w = rint(s // 6, s // 3)
        h = rint(s // 6, s // 3)
        cx = rint(w // 2 + 1, s - w // 2 - 1)
        cy = rint(h // 2 + 1, s - h // 2 - 1)
        mask = ((ys - cy).abs() <= h / 2) & ((xs - cx).abs() <= w / 2)
    elif label == 1:  # circle
        r = rint(s // 10, s // 5)
        cx = rint(r + 1, s - r - 1)
        cy = rint(r + 1, s - r - 1)
        mask = (ys - cy) ** 2 + (xs - cx) ** 2 <= r**2
    else:  # upright triangle: width grows linearly from apex to base
        h = rint(s // 5, s // 3)
        cx = rint(h // 2 + 1, s - h // 2 - 1)
        top = rint(1, s - h - 1)
        rel = (ys - top).clamp(min=0)
        mask = (ys >= top) & (ys <= top + h) & ((xs - cx).abs() <= rel / 2)

    nz = mask.nonzero()
    if nz.numel() == 0:
        return None, None

    # Bright colour on dark-ish background so shapes are learnable.
    color = torch.rand(3, generator=generator) * 0.6 + 0.4
    image[:, mask] = color.unsqueeze(1)

    rows, cols = nz[:, 0], nz[:, 1]
    box = torch.tensor(
        [cols.min(), rows.min(), cols.max() + 1, rows.max() + 1],
        dtype=torch.float32,
    )
    return box, mask
