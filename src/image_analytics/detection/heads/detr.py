"""DETR (Carion 2020): end-to-end detection as set prediction.

No anchors, no NMS: a transformer decoder turns N learned object queries into
N (class, box) predictions; bipartite (Hungarian) matching assigns each GT to
exactly one query, and unmatched queries learn the "no object" class.

Implementation notes
- Positional embeddings are injected into Q/K at *every* attention layer
  (DETR's design), which stock ``nn.Transformer`` cannot do — hence the small
  custom encoder/decoder layers below.
- Boxes are predicted as normalized (cx, cy, w, h) through a sigmoid.
- Auxiliary losses on every intermediate decoder layer (crucial for
  convergence) are on by default.
- Known limitation (documented in IMPLEMENTATION_PLAN.md): DETR needs 100+
  epochs on real data; here it is validated by overfit tests and synthetic
  sanity runs. Production transformer detection arrives via wrappers
  (Deformable DETR / RT-DETR) later.

Interface matches the other detectors (0-based foreground labels).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as tvops
from torchvision.ops import generalized_box_iou

from image_analytics.core.registry import MODELS


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------


def sine_position_encoding(
    height: int,
    width: int,
    dim: int,
    device: torch.device | str = "cpu",
    temperature: float = 10000.0,
) -> torch.Tensor:
    """2D sine/cosine positional encoding, (H*W, dim); half the channels
    encode y, half encode x, both normalized to [0, 2*pi]."""
    if dim % 4 != 0:
        raise ValueError(f"dim must be divisible by 4, got {dim}")
    feats = dim // 2

    y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height * 2 * math.pi
    x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width * 2 * math.pi

    dim_t = torch.arange(feats, device=device, dtype=torch.float32)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / feats)

    pos_y = y[:, None] / dim_t                       # (H, feats)
    pos_x = x[:, None] / dim_t                       # (W, feats)
    pos_y = torch.stack([pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()], dim=2).flatten(1)
    pos_x = torch.stack([pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()], dim=2).flatten(1)

    pos = torch.cat(
        [
            pos_y[:, None, :].expand(height, width, feats),
            pos_x[None, :, :].expand(height, width, feats),
        ],
        dim=2,
    )
    return pos.reshape(height * width, dim)


# ---------------------------------------------------------------------------
# Transformer (pos injected at every layer)
# ---------------------------------------------------------------------------


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        q = k = src + pos
        src = self.norm1(src + self.dropout(self.self_attn(q, k, value=src)[0]))
        ff = self.linear2(self.dropout(F.relu(self.linear1(src))))
        return self.norm2(src + self.dropout(ff))


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        pos: torch.Tensor,
        query_pos: torch.Tensor,
    ) -> torch.Tensor:
        q = k = tgt + query_pos
        tgt = self.norm1(tgt + self.dropout(self.self_attn(q, k, value=tgt)[0]))
        cross = self.cross_attn(
            query=tgt + query_pos, key=memory + pos, value=memory
        )[0]
        tgt = self.norm2(tgt + self.dropout(cross))
        ff = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        return self.norm3(tgt + self.dropout(ff))


class MLP(nn.Module):
    """Simple multi-layer perceptron (the DETR box head)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int) -> None:
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


# ---------------------------------------------------------------------------
# Hungarian matching + set criterion
# ---------------------------------------------------------------------------


class HungarianMatcher:
    """Optimal bipartite assignment between queries and GT boxes.

    Cost per (query, gt) pair: -P(class) * w_cls + L1(box) * w_bbox
    + (-GIoU) * w_giou, solved with scipy's Hungarian algorithm.
    """

    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0) -> None:
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def __call__(
        self,
        pred_logits: torch.Tensor,       # (B, Q, K+1)
        pred_boxes: torch.Tensor,        # (B, Q, 4) normalized cxcywh
        targets: list[dict],             # per image: boxes (normalized cxcywh), labels
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        from scipy.optimize import linear_sum_assignment

        indices = []
        for b, target in enumerate(targets):
            tgt_boxes = target["boxes_norm"]
            tgt_labels = target["labels"]
            if len(tgt_boxes) == 0:
                empty = torch.empty(0, dtype=torch.int64)
                indices.append((empty, empty))
                continue

            prob = pred_logits[b].softmax(dim=-1)             # (Q, K+1)
            cost_class = -prob[:, tgt_labels]                 # (Q, N)
            cost_bbox = torch.cdist(pred_boxes[b], tgt_boxes, p=1)
            cost_giou = -generalized_box_iou(
                tvops.box_convert(pred_boxes[b], "cxcywh", "xyxy"),
                tvops.box_convert(tgt_boxes, "cxcywh", "xyxy"),
            )
            cost = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_giou * cost_giou
            )
            row, col = linear_sum_assignment(cost.cpu().numpy())
            indices.append(
                (torch.as_tensor(row, dtype=torch.int64),
                 torch.as_tensor(col, dtype=torch.int64))
            )
        return indices


class SetCriterion(nn.Module):
    """DETR losses: CE over all queries (no-object downweighted by
    ``eos_coef``) + L1 and GIoU on matched pairs, normalized by GT count."""

    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        eos_coef: float = 0.1,
        weight_class: float = 1.0,
        weight_bbox: float = 5.0,
        weight_giou: float = 2.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_class = weight_class
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou
        class_weights = torch.ones(num_classes + 1)
        class_weights[num_classes] = eos_coef
        self.register_buffer("class_weights", class_weights)

    def forward(
        self,
        pred_logits: torch.Tensor,   # (B, Q, K+1)
        pred_boxes: torch.Tensor,    # (B, Q, 4) normalized cxcywh
        targets: list[dict],
    ) -> dict[str, torch.Tensor]:
        indices = self.matcher(pred_logits, pred_boxes, targets)
        num_boxes = max(sum(len(t["labels"]) for t in targets), 1)

        # Classification: default everything to "no object"
        target_classes = torch.full(
            pred_logits.shape[:2], self.num_classes,
            dtype=torch.int64, device=pred_logits.device,
        )
        for b, (query_idx, tgt_idx) in enumerate(indices):
            target_classes[b, query_idx] = targets[b]["labels"][tgt_idx]
        loss_class = F.cross_entropy(
            pred_logits.flatten(0, 1), target_classes.flatten(),
            weight=self.class_weights,
        )

        # Boxes: only matched pairs contribute
        matched_pred, matched_tgt = [], []
        for b, (query_idx, tgt_idx) in enumerate(indices):
            if len(query_idx):
                matched_pred.append(pred_boxes[b, query_idx])
                matched_tgt.append(targets[b]["boxes_norm"][tgt_idx])
        if matched_pred:
            pred_cat = torch.cat(matched_pred)
            tgt_cat = torch.cat(matched_tgt)
            loss_bbox = F.l1_loss(pred_cat, tgt_cat, reduction="sum") / num_boxes
            loss_giou = (
                1.0
                - generalized_box_iou(
                    tvops.box_convert(pred_cat, "cxcywh", "xyxy"),
                    tvops.box_convert(tgt_cat, "cxcywh", "xyxy"),
                ).diagonal()
            ).sum() / num_boxes
        else:
            loss_bbox = pred_boxes.sum() * 0.0
            loss_giou = pred_boxes.sum() * 0.0

        return {
            "loss_class": loss_class * self.weight_class,
            "loss_bbox": loss_bbox * self.weight_bbox,
            "loss_giou": loss_giou * self.weight_giou,
        }


# ---------------------------------------------------------------------------
# DETR model
# ---------------------------------------------------------------------------


@MODELS.register("detr")
class DETR(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,             # single-scale (C5) pyramid backbone
        num_classes: int,
        hidden_dim: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_queries: int = 100,
        aux_loss: bool = True,
        eos_coef: float = 0.1,
        score_thresh: float = 0.05,
    ) -> None:
        super().__init__()
        if not getattr(backbone, "features_only", False):
            raise ValueError(
                "DETR requires a pyramid backbone; set backbone.features_only: true "
                "(out_indices=[4] for C5)"
            )
        self.backbone = backbone
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.aux_loss = aux_loss
        self.score_thresh = score_thresh

        self.input_proj = nn.Conv2d(backbone.feature_channels[-1], hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.encoder = nn.ModuleList(
            EncoderLayer(hidden_dim, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        )
        self.decoder = nn.ModuleList(
            DecoderLayer(hidden_dim, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        )
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, num_layers=3)
        self.hidden_dim = hidden_dim

        self.criterion = SetCriterion(
            num_classes, HungarianMatcher(), eos_coef=eos_coef
        )

    # -- forward ---------------------------------------------------------

    def forward(self, images: torch.Tensor, targets: list[dict] | None = None):
        features = self.backbone(images)[-1]                  # (B, C, H, W) C5
        b, _, h, w = features.shape

        src = self.input_proj(features).flatten(2).transpose(1, 2)   # (B, HW, D)
        pos = sine_position_encoding(h, w, self.hidden_dim, device=src.device)
        pos = pos.unsqueeze(0)                                       # (1, HW, D)

        memory = src
        for layer in self.encoder:
            memory = layer(memory, pos)

        query_pos = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        tgt = torch.zeros_like(query_pos)
        intermediates = []
        for layer in self.decoder:
            tgt = layer(tgt, memory, pos, query_pos)
            intermediates.append(self.decoder_norm(tgt))
        hs = torch.stack(intermediates)                              # (L, B, Q, D)

        logits = self.class_embed(hs)                                # (L, B, Q, K+1)
        boxes = self.bbox_embed(hs).sigmoid()                        # (L, B, Q, 4)

        if self.training:
            if targets is None:
                raise ValueError("targets are required in training mode")
            return self._compute_losses(logits, boxes, targets, images.shape[-2:])
        return self._inference(logits[-1], boxes[-1], images.shape[-2:])

    # -- training --------------------------------------------------------

    def _compute_losses(
        self,
        logits: torch.Tensor,
        boxes: torch.Tensor,
        targets: list[dict],
        image_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        h, w = image_size
        scale = torch.tensor([w, h, w, h], dtype=torch.float32, device=boxes.device)
        prepared = []
        for target in targets:
            gt = torch.as_tensor(target["boxes"], dtype=torch.float32).to(boxes.device)
            norm = (
                tvops.box_convert(gt, "xyxy", "cxcywh") / scale
                if len(gt)
                else gt.reshape(0, 4)
            )
            prepared.append(
                {"boxes_norm": norm, "labels": target["labels"].to(boxes.device)}
            )

        losses = self.criterion(logits[-1], boxes[-1], prepared)
        if self.aux_loss:
            for layer_idx in range(logits.shape[0] - 1):
                aux = self.criterion(logits[layer_idx], boxes[layer_idx], prepared)
                for key, value in aux.items():
                    losses[f"{key}_aux{layer_idx}"] = value

        losses["loss"] = sum(losses.values())
        return losses

    # -- inference -------------------------------------------------------

    @torch.no_grad()
    def _inference(
        self,
        logits: torch.Tensor,            # (B, Q, K+1)
        boxes: torch.Tensor,             # (B, Q, 4) normalized cxcywh
        image_size: tuple[int, int],
    ) -> list[dict[str, torch.Tensor]]:
        h, w = image_size
        scale = torch.tensor([w, h, w, h], dtype=torch.float32, device=boxes.device)
        prob = logits.softmax(dim=-1)[..., :-1]                # drop no-object
        scores, labels = prob.max(dim=-1)                      # (B, Q)

        results = []
        for b in range(logits.shape[0]):
            keep = scores[b] > self.score_thresh
            decoded = tvops.box_convert(boxes[b][keep] * scale, "cxcywh", "xyxy")
            decoded[:, 0::2] = decoded[:, 0::2].clamp(0, w)
            decoded[:, 1::2] = decoded[:, 1::2].clamp(0, h)
            # Set prediction: one query = one object, no NMS needed
            results.append(
                {
                    "boxes": decoded,
                    "scores": scores[b][keep],
                    "labels": labels[b][keep],
                }
            )
        return results
