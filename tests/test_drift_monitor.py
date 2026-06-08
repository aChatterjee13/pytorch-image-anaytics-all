import numpy as np
import pytest

from image_analytics.serving.monitoring import (
    DriftMonitor,
    cosine_shift,
    ks_statistic,
    population_stability_index,
)


class TestPSI:
    def test_identical_is_near_zero(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=2000)
        assert population_stability_index(x, x) < 0.01

    def test_large_shift_exceeds_alert(self):
        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, 2000)
        cur = rng.normal(3, 1, 2000)
        assert population_stability_index(ref, cur) > 0.2


class TestKS:
    def test_identical_is_zero(self):
        x = np.linspace(0, 1, 100)
        assert ks_statistic(x, x) == pytest.approx(0.0)

    def test_disjoint_is_one(self):
        ref = np.array([0.0, 1, 2, 3])
        cur = np.array([10.0, 11, 12, 13])
        assert ks_statistic(ref, cur) == pytest.approx(1.0)

    def test_half_shift(self):
        # ref U[0,1], cur U[0.5,1.5]; KS is the max CDF gap (~0.5)
        ref = np.linspace(0, 1, 1000)
        cur = np.linspace(0.5, 1.5, 1000)
        assert ks_statistic(ref, cur) == pytest.approx(0.5, abs=0.02)


class TestCosineShift:
    def test_same_direction_zero(self):
        assert cosine_shift(np.array([1.0, 0]), np.array([2.0, 0])) == pytest.approx(0.0)

    def test_orthogonal_one(self):
        assert cosine_shift(np.array([1.0, 0]), np.array([0.0, 1])) == pytest.approx(1.0)


class TestDriftMonitor:
    def test_no_drift_no_alert(self):
        rng = np.random.default_rng(0)
        ref = rng.normal(size=(1000, 3))
        monitor = DriftMonitor().fit(ref)
        report = monitor.score(rng.normal(size=(500, 3)))
        assert not report["alert"]
        assert report["psi_max"] < 0.1
        assert report["cosine_shift"] < 0.2

    def test_drift_triggers_alert(self):
        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, size=(1000, 2))
        monitor = DriftMonitor().fit(ref)
        report = monitor.score(rng.normal(4, 1, size=(500, 2)))
        assert report["alert"] and report["psi_max"] > 0.2
        assert len(report["psi"]) == 2 and len(report["ks"]) == 2

    def test_dimension_mismatch_raises(self):
        monitor = DriftMonitor().fit(np.zeros((10, 3)))
        with pytest.raises(ValueError):
            monitor.score(np.zeros((5, 4)))

    def test_score_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            DriftMonitor().score(np.zeros((5, 3)))
