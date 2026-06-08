"""Data/embedding drift monitoring — pure numpy, no required deps.

``DriftMonitor.fit(reference)`` captures a reference distribution (per-feature
samples + mean vector); ``score(batch)`` reports the Population Stability Index
(PSI) and Kolmogorov–Smirnov (KS) statistic per feature plus the cosine shift
of the mean embedding, and flags an alert when PSI exceeds the threshold
(default 0.2). Works on any ``(N, D)`` matrix — input features, embeddings, or
prediction-probability vectors. Evidently/WhyLabs remain optional.
"""

from __future__ import annotations

import numpy as np


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = 10
) -> float:
    """PSI between two 1D samples: ``Σ (q - p) · ln(q / p)`` over quantile bins
    of the reference (p = reference proportion, q = current proportion)."""
    reference = np.asarray(reference, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    # Deduplicated quantile edges (ties collapse) with finite outer bounds that
    # cover both samples, so np.histogram stays monotonic and loses no mass.
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        return 0.0  # constant reference -> no measurable shift
    edges[0] = min(edges[0], reference.min(), current.min()) - 1e-9
    edges[-1] = max(edges[-1], reference.max(), current.max()) + 1e-9

    ref_counts = np.histogram(reference, edges)[0].astype(float)
    cur_counts = np.histogram(current, edges)[0].astype(float)
    p = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    q = np.clip(cur_counts / cur_counts.sum(), 1e-6, None)
    return float(np.sum((q - p) * np.log(q / p)))


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    """Two-sample Kolmogorov–Smirnov statistic (max empirical-CDF gap)."""
    ref_sorted = np.sort(reference)
    cur_sorted = np.sort(current)
    grid = np.concatenate([ref_sorted, cur_sorted])
    cdf_ref = np.searchsorted(ref_sorted, grid, side="right") / len(ref_sorted)
    cdf_cur = np.searchsorted(cur_sorted, grid, side="right") / len(cur_sorted)
    return float(np.max(np.abs(cdf_ref - cdf_cur)))


def cosine_shift(reference_mean: np.ndarray, current_mean: np.ndarray) -> float:
    """``1 - cos(reference_mean, current_mean)`` — embedding centroid drift."""
    denom = (np.linalg.norm(reference_mean) * np.linalg.norm(current_mean)) + 1e-12
    return float(1.0 - np.dot(reference_mean, current_mean) / denom)


class DriftMonitor:
    def __init__(self, bins: int = 10, psi_alert: float = 0.2) -> None:
        self.bins = bins
        self.psi_alert = psi_alert
        self.reference: np.ndarray | None = None
        self.reference_mean: np.ndarray | None = None

    def fit(self, reference) -> "DriftMonitor":
        ref = np.asarray(reference, dtype=np.float64)
        if ref.ndim != 2:
            raise ValueError(f"reference must be (N, D), got shape {ref.shape}")
        self.reference = ref
        self.reference_mean = ref.mean(axis=0)
        return self

    def score(self, batch) -> dict:
        if self.reference is None:
            raise RuntimeError("DriftMonitor.score called before fit()")
        cur = np.asarray(batch, dtype=np.float64)
        if cur.ndim != 2 or cur.shape[1] != self.reference.shape[1]:
            raise ValueError(
                f"batch must be (M, {self.reference.shape[1]}), got shape {cur.shape}"
            )

        psi = [
            population_stability_index(self.reference[:, d], cur[:, d], self.bins)
            for d in range(cur.shape[1])
        ]
        ks = [
            ks_statistic(self.reference[:, d], cur[:, d]) for d in range(cur.shape[1])
        ]
        psi_max = max(psi)
        return {
            "psi": psi,
            "psi_max": psi_max,
            "ks": ks,
            "ks_max": max(ks),
            "cosine_shift": cosine_shift(self.reference_mean, cur.mean(axis=0)),
            "alert": psi_max > self.psi_alert,
        }
