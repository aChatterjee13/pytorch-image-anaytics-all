"""Prithvi (IBM/NASA 2023): a temporal ViT for multi-temporal, multi-spectral
imagery, registered as a backbone.

A 3D patch embedding (Conv3d over T×H×W) tokenises a co-registered time series
``(B, C, T, H, W)``; tokens run through a ViT encoder. Offline-instantiable
(random init); ``pretrained=True`` best-effort loads HF-hub weights (network).
Pairs with the Phase 4 :class:`TemporalStackDataset`.

Forward: ``(B, C, T, H, W) -> (B, embed_dim)`` (CLS-pooled).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from image_analytics.core.registry import BACKBONES
from image_analytics.foundation._vit import TransformerBlock


class PrithviBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        img_size: int = 96,
        patch_size: int = 16,
        num_frames: int = 3,
        tubelet_size: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.patch_embed = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )
        grid_t = num_frames // tubelet_size
        grid = img_size // patch_size
        num_patches = grid_t * grid * grid

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList(
            TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.features_only = False
        self.feature_dim = embed_dim
        self.feature_channels = [embed_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(
                f"Prithvi expects (B, C, T, H, W) temporal input, got shape {tuple(x.shape)}"
            )
        b = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)      # (B, N, D)
        x = x + self.pos_embed
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)[:, 0]


@BACKBONES.register("prithvi_100m")
def build_prithvi(
    pretrained: bool = False,
    in_channels: int = 6,
    features_only: bool = False,
    hf_repo: str = "ibm-nasa-geospatial/Prithvi-100M",
    **kwargs,
) -> PrithviBackbone:
    if features_only:
        raise ValueError("PrithviBackbone provides pooled features only (features_only=False)")
    model = PrithviBackbone(in_channels=in_channels, **kwargs)
    if pretrained:  # pragma: no cover - network
        _load_pretrained(model, hf_repo)
    return model


def _load_pretrained(model: PrithviBackbone, hf_repo: str) -> None:  # pragma: no cover - network
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download Prithvi weights "
            "(pip install 'image-analytics[seg]')."
        ) from exc
    try:
        path = hf_hub_download(hf_repo, "Prithvi_100M.pt")
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("model", state), strict=False)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load Prithvi weights from {hf_repo!r}: {exc}. Pass "
            "pretrained=False to use a randomly-initialised Prithvi backbone."
        ) from exc
