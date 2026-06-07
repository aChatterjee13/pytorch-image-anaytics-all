"""Metric computation.

Evaluators accumulate statistics over batches (``update``) and produce a
metrics dict (``compute``). State is synchronized across processes when
torch.distributed is initialized, so all ranks see identical metrics —
required for consistent early-stopping/checkpoint decisions under DDP.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def _reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce (sum) across ranks when distributed is initialized."""
    if dist.is_available() and dist.is_initialized():
        backend_device = "cuda" if dist.get_backend() == "nccl" else "cpu"
        work = tensor.to(backend_device)
        dist.all_reduce(work, op=dist.ReduceOp.SUM)
        return work.cpu()
    return tensor


class Evaluator:
    """Base interface for streaming metric computation."""

    def reset(self) -> None:
        raise NotImplementedError

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        raise NotImplementedError

    def compute(self) -> dict[str, float]:
        raise NotImplementedError


class ClassificationEvaluator(Evaluator):
    """Single-label classification metrics from a streaming confusion matrix.

    Produces: ``accuracy``, ``macro_precision``, ``macro_recall``, ``macro_f1``
    and ``top{k}_accuracy`` for each requested k > 1. Macro averages are taken
    over classes with support (precision over predicted-positive classes).
    """

    def __init__(self, num_classes: int, topk: tuple[int, ...] = (1,)) -> None:
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        self.num_classes = num_classes
        self.topk = tuple(sorted({k for k in topk if 1 <= k <= num_classes}))
        self.reset()

    def reset(self) -> None:
        self.confusion = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.long
        )
        self.topk_correct = {k: 0 for k in self.topk if k > 1}
        self.total = 0

    @torch.no_grad()
    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        outputs = outputs.detach().float().cpu()
        targets = targets.detach().cpu().long()

        preds = outputs.argmax(dim=1)
        indices = targets * self.num_classes + preds
        self.confusion += torch.bincount(
            indices, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

        if self.topk_correct:
            maxk = max(self.topk_correct)
            topk_preds = outputs.topk(maxk, dim=1).indices
            hits = topk_preds.eq(targets.unsqueeze(1))
            for k in self.topk_correct:
                self.topk_correct[k] += int(hits[:, :k].any(dim=1).sum())

        self.total += targets.numel()

    def compute(self) -> dict[str, float]:
        confusion = _reduce_sum(self.confusion.clone()).float()
        counters = torch.tensor(
            [self.total, *self.topk_correct.values()], dtype=torch.long
        )
        counters = _reduce_sum(counters)
        total = int(counters[0])
        if total == 0:
            return {}

        diag = confusion.diag()
        support = confusion.sum(dim=1)        # true counts per class
        predicted = confusion.sum(dim=0)      # predicted counts per class

        precision = diag / predicted.clamp(min=1)
        recall = diag / support.clamp(min=1)
        f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)

        has_support = support > 0
        has_pred = predicted > 0
        metrics = {
            "accuracy": float(diag.sum() / total),
            "macro_precision": float(precision[has_pred].mean()) if has_pred.any() else 0.0,
            "macro_recall": float(recall[has_support].mean()) if has_support.any() else 0.0,
            "macro_f1": float(f1[has_support].mean()) if has_support.any() else 0.0,
        }
        for i, k in enumerate(self.topk_correct, start=1):
            metrics[f"top{k}_accuracy"] = float(counters[i]) / total
        return metrics


def average_precision(scores: torch.Tensor, targets: torch.Tensor) -> float:
    """AP for one label: mean precision at each true-positive rank."""
    num_pos = int(targets.sum())
    if num_pos == 0:
        return float("nan")
    order = scores.argsort(descending=True)
    hits = targets[order].float()
    cum_tp = hits.cumsum(dim=0)
    precision_at = cum_tp / torch.arange(1, len(hits) + 1, dtype=torch.float32)
    return float((precision_at * hits).sum() / num_pos)


class MultiLabelEvaluator(Evaluator):
    """Multi-label metrics from accumulated sigmoid scores.

    Produces: ``mAP`` (macro over labels with positives), ``micro_f1``,
    ``macro_f1``, ``subset_accuracy``, and ``accuracy`` (alias of micro_f1 so
    the default ``val/accuracy`` monitor works for both task flavors).
    """

    def __init__(self, num_labels: int, threshold: float = 0.5) -> None:
        self.num_labels = num_labels
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self._scores: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []

    @torch.no_grad()
    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        self._scores.append(torch.sigmoid(outputs.detach().float()).cpu())
        self._targets.append(targets.detach().cpu().long())

    def _gather(self) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.cat(self._scores) if self._scores else torch.empty(0, self.num_labels)
        targets = torch.cat(self._targets) if self._targets else torch.empty(0, self.num_labels, dtype=torch.long)
        if dist.is_available() and dist.is_initialized():
            bundle: list[tuple[torch.Tensor, torch.Tensor]] = [None] * dist.get_world_size()  # type: ignore[list-item]
            dist.all_gather_object(bundle, (scores, targets))
            scores = torch.cat([s for s, _ in bundle])
            targets = torch.cat([t for _, t in bundle])
        return scores, targets

    def compute(self) -> dict[str, float]:
        scores, targets = self._gather()
        if scores.numel() == 0:
            return {}

        preds = (scores >= self.threshold).long()
        tp = (preds & targets).sum().float()
        fp = (preds & ~targets.bool()).sum().float()
        fn = ((1 - preds) & targets).sum().float()
        micro_f1 = float(2 * tp / (2 * tp + fp + fn).clamp(min=1e-12))

        per_label_f1 = []
        aps = []
        for i in range(self.num_labels):
            t, p = targets[:, i], preds[:, i]
            ltp = float((p & t).sum())
            lfp = float((p & ~t.bool()).sum())
            lfn = float(((1 - p) & t).sum())
            denom = 2 * ltp + lfp + lfn
            if t.sum() > 0:
                per_label_f1.append(2 * ltp / denom if denom > 0 else 0.0)
                aps.append(average_precision(scores[:, i], t))

        macro_f1 = sum(per_label_f1) / len(per_label_f1) if per_label_f1 else 0.0
        mean_ap = sum(aps) / len(aps) if aps else 0.0
        subset_acc = float((preds == targets).all(dim=1).float().mean())

        return {
            "accuracy": micro_f1,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "mAP": mean_ap,
            "subset_accuracy": subset_acc,
        }


class DetectionEvaluator(Evaluator):
    """COCO-style mean average precision.

    ``update`` consumes per-image prediction dicts (``boxes`` XYXY,
    ``scores``, ``labels``) and target dicts (``boxes``, ``labels``); labels
    are 0-based foreground classes. ``compute`` reports ``mAP`` (mean over
    IoU thresholds 0.50:0.05:0.95), ``mAP50``, and ``mAP75``, following the
    COCO protocol: per-image greedy matching in score order, 101-point
    interpolated AP, averaged over classes present in the ground truth.
    """

    def __init__(
        self,
        num_classes: int,
        iou_thresholds: tuple[float, ...] | None = None,
        max_detections: int = 100,
    ) -> None:
        self.num_classes = num_classes
        self.iou_thresholds = tuple(
            iou_thresholds
            if iou_thresholds is not None
            else [0.5 + 0.05 * i for i in range(10)]
        )
        self.max_detections = max_detections
        self.reset()

    def reset(self) -> None:
        self._preds: list[dict[str, torch.Tensor]] = []
        self._targets: list[dict[str, torch.Tensor]] = []

    @torch.no_grad()
    def update(self, outputs, targets) -> None:
        for pred, target in zip(outputs, targets):
            scores = pred["scores"].detach().float().cpu()
            keep = scores.argsort(descending=True)[: self.max_detections]
            self._preds.append(
                {
                    "boxes": pred["boxes"].detach().float().cpu()[keep],
                    "scores": scores[keep],
                    "labels": pred["labels"].detach().cpu()[keep],
                }
            )
            self._targets.append(
                {
                    "boxes": torch.as_tensor(target["boxes"]).float().cpu(),
                    "labels": target["labels"].detach().cpu(),
                }
            )

    def _gather(self) -> tuple[list, list]:
        preds, targets = self._preds, self._targets
        if dist.is_available() and dist.is_initialized():
            bundle: list = [None] * dist.get_world_size()
            dist.all_gather_object(bundle, (preds, targets))
            preds = [p for chunk, _ in bundle for p in chunk]
            targets = [t for _, chunk in bundle for t in chunk]
        return preds, targets

    @staticmethod
    def _interpolated_ap(scores: torch.Tensor, tps: torch.Tensor, num_gt: int) -> float:
        """COCO 101-point interpolated AP from global score-sorted TP flags."""
        order = scores.argsort(descending=True)
        tp_cum = tps[order].float().cumsum(dim=0)
        fp_cum = (~tps[order]).float().cumsum(dim=0)
        recall = tp_cum / num_gt
        precision = tp_cum / (tp_cum + fp_cum)
        # Monotonic non-increasing precision envelope
        envelope = precision.flip(0).cummax(dim=0).values.flip(0)
        points = torch.linspace(0, 1, 101)
        idx = torch.searchsorted(recall.contiguous(), points)
        valid = idx < len(envelope)
        interp = torch.zeros(101)
        interp[valid] = envelope[idx[valid]]
        return float(interp.mean())

    def compute(self) -> dict[str, float]:
        preds, targets = self._gather()
        if not targets:
            return {}

        thresholds = self.iou_thresholds
        # ap[class][threshold_index]
        aps: dict[int, list[float]] = {}

        for cls in range(self.num_classes):
            num_gt = sum(int((t["labels"] == cls).sum()) for t in targets)
            if num_gt == 0:
                continue

            # Per-image IoU matrices computed once, matched per threshold
            per_image: list[tuple[torch.Tensor, torch.Tensor]] = []  # (scores, ious)
            for pred, target in zip(preds, targets):
                pmask = pred["labels"] == cls
                gmask = target["labels"] == cls
                pboxes, pscores = pred["boxes"][pmask], pred["scores"][pmask]
                gboxes = target["boxes"][gmask]
                if len(pboxes) == 0:
                    continue
                order = pscores.argsort(descending=True)
                pboxes, pscores = pboxes[order], pscores[order]
                if len(gboxes):
                    import torchvision.ops as tvops

                    ious = tvops.box_iou(pboxes, gboxes)
                else:
                    ious = torch.zeros(len(pboxes), 0)
                per_image.append((pscores, ious))

            cls_aps = []
            for thr in thresholds:
                all_scores, all_tps = [], []
                for pscores, ious in per_image:
                    matched = torch.zeros(ious.shape[1], dtype=torch.bool)
                    tps = torch.zeros(len(pscores), dtype=torch.bool)
                    for i in range(len(pscores)):
                        if ious.shape[1] == 0:
                            break
                        candidate = ious[i].clone()
                        candidate[matched] = -1.0
                        best = int(candidate.argmax())
                        if candidate[best] >= thr:
                            matched[best] = True
                            tps[i] = True
                    all_scores.append(pscores)
                    all_tps.append(tps)
                if all_scores:
                    cls_aps.append(
                        self._interpolated_ap(
                            torch.cat(all_scores), torch.cat(all_tps), num_gt
                        )
                    )
                else:
                    cls_aps.append(0.0)
            aps[cls] = cls_aps

        if not aps:
            return {"mAP": 0.0, "mAP50": 0.0, "mAP75": 0.0}

        per_class = torch.tensor(list(aps.values()))  # (C_present, T)
        idx50 = thresholds.index(0.5) if 0.5 in thresholds else 0
        metrics = {
            "mAP": float(per_class.mean()),
            "mAP50": float(per_class[:, idx50].mean()),
        }
        if 0.75 in thresholds:
            metrics["mAP75"] = float(per_class[:, thresholds.index(0.75)].mean())
        return metrics
