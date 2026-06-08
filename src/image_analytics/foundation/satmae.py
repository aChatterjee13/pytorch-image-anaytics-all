"""SatMAE (Cong, ICML 2023): a ViT with grouped-band patch embedding and
spectral positional encodings, registered as a backbone.

Each spectral group is patch-embedded by its own projection and tagged with a
learned spectral embedding; the group token sequences are concatenated (sharing
a spatial positional encoding) and run through a standard ViT encoder. The
architecture is offline-instantiable (random init); ``pretrained=True``
best-effort loads remapped HF-hub weights (network required) — defaults match
``satmae_base`` so real checkpoints fit.

Registered in ``BACKBONES`` so any head (classifier, U-Net, FPN) consumes it via
the usual config path. Forward: ``(B, C, H, W) -> (B, embed_dim)`` (CLS-pooled).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from image_analytics.core.registry import BACKBONES
from image_analytics.foundation._vit import TransformerBlock

# Default Sentinel-2 (10-band) grouping used by SatMAE.
DEFAULT_BAND_GROUPS = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]]


class SatMAEBackbone(nn.Module):
    def __init__(
        self,
        band_groups: list | None = None,
        img_size: int = 96,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        in_channels: int | None = None,  # accepted for registry compat; groups define bands
    ) -> None:
        super().__init__()
        self.band_groups = [list(g) for g in (band_groups or DEFAULT_BAND_GROUPS)]
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.grid = img_size // patch_size
        num_patches = self.grid * self.grid

        self.patch_embeds = nn.ModuleList(
            nn.Conv2d(len(g), embed_dim, patch_size, stride=patch_size)
            for g in self.band_groups
        )
        # Learned spectral (per-group) + spatial positional encodings.
        self.spectral_embed = nn.Parameter(torch.zeros(len(self.band_groups), 1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.spectral_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList(
            TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Feature interface (pooled mode — what classifier heads consume).
        self.features_only = False
        self.feature_dim = embed_dim
        self.feature_channels = [embed_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        tokens = []
        for g, (group, embed) in enumerate(zip(self.band_groups, self.patch_embeds)):
            t = embed(x[:, group]).flatten(2).transpose(1, 2)   # (B, Np, D)
            tokens.append(t + self.spectral_embed[g] + self.pos_embed)
        x = torch.cat(tokens, dim=1)                            # (B, G*Np, D)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)[:, 0]                               # CLS-pooled


@BACKBONES.register("satmae_base")
def build_satmae(
    pretrained: bool = False,
    in_channels: int | None = None,
    features_only: bool = False,
    hf_repo: str = "MVRL/satmae-pretrain-vit-base",
    **kwargs,
) -> SatMAEBackbone:
    if features_only:
        raise ValueError("SatMAEBackbone provides pooled features only (features_only=False)")
    model = SatMAEBackbone(in_channels=in_channels, **kwargs)
    if pretrained:
        _load_pretrained(model, hf_repo)
    return model


def _load_pretrained(model: SatMAEBackbone, hf_repo: str) -> None:  # pragma: no cover - network
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download SatMAE weights "
            "(pip install 'image-analytics[seg]')."
        ) from exc
    try:
        path = hf_hub_download(hf_repo, "pytorch_model.bin")
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = state.get("model", state)
        model.load_state_dict(_remap_keys(state), strict=False)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load SatMAE weights from {hf_repo!r}: {exc}. Pass "
            "pretrained=False to use a randomly-initialised SatMAE backbone."
        ) from exc


def _remap_keys(state: dict) -> dict:  # pragma: no cover - network
    """Best-effort key remap from common SatMAE checkpoints to this module."""
    out = {}
    for k, v in state.items():
        k = k.replace("blocks.", "blocks.").replace("norm.", "norm.")
        out[k] = v
    return out
