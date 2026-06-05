"""Tests for drift detection and statistical evaluation."""
import numpy as np
import pytest
from shadowserve.evaluation import drift as drift_module


@pytest.fixture(autouse=True)
def reset_drift():
    drift_module.reset_state()
    yield
    drift_module.reset_state()


class TestDriftDetection:
    def test_no_drift_identical_distributions(self):
        rng = np.random.default_rng(42)
        samples = rng.uniform(0.3, 0.7, 100).tolist()
        for s in samples:
            drift_module.record_production(s)
            drift_module.record_shadow(s + rng.normal(0, 0.001))

        state = drift_module.get_state()
        assert not state.drift_detected, "Identical distributions should not trigger drift"
        assert state.last_ks_pvalue > 0.05

    def test_drift_detected_on_shifted_distribution(self):
        rng = np.random.default_rng(0)
        # production: low-risk cluster
        for _ in range(100):
            drift_module.record_production(float(rng.uniform(0.6, 0.95)))
        # shadow: high-risk cluster (clearly different)
        for _ in range(100):
            drift_module.record_shadow(float(rng.uniform(0.05, 0.35)))

        state = drift_module.get_state()
        assert state.drift_detected, "Highly shifted distributions should trigger drift"
        assert state.last_ks_statistic > 0.5

    def test_psi_increases_with_shift(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            drift_module.record_production(float(rng.uniform(0.5, 0.8)))
        for _ in range(50):
            drift_module.record_shadow(float(rng.uniform(0.1, 0.4)))

        state = drift_module.get_state()
        assert state.last_psi > 0.0

    def test_minimum_sample_size(self):
        # Fewer than 30 samples should not trigger evaluation
        for _ in range(10):
            drift_module.record_production(0.5)
            drift_module.record_shadow(0.9)

        state = drift_module.get_state()
        assert state.last_ks_statistic == 0.0, "Should not evaluate with < 30 samples"

    def test_drift_count_increments(self):
        rng = np.random.default_rng(2)
        for _ in range(50):
            drift_module.record_production(float(rng.uniform(0.7, 0.95)))
        for _ in range(50):
            drift_module.record_shadow(float(rng.uniform(0.05, 0.20)))

        state = drift_module.get_state()
        assert state.drift_count >= 1

    def test_wasserstein_greater_than_zero_on_shift(self):
        rng = np.random.default_rng(3)
        for _ in range(50):
            drift_module.record_production(float(rng.uniform(0.6, 0.9)))
        for _ in range(50):
            drift_module.record_shadow(float(rng.uniform(0.1, 0.4)))

        state = drift_module.get_state()
        assert state.last_wasserstein > 0.1
