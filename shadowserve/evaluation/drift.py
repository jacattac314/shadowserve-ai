"""
Statistical drift detection between production and challenger prediction distributions.

Maintains a rolling window of production and shadow probabilities, then computes:
  - Kolmogorov-Smirnov (KS) two-sample test
  - Population Stability Index (PSI)
  - KL Divergence
  - Wasserstein Distance (Earth Mover's Distance)
"""
import math
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque
import numpy as np
from scipy import stats
from shadowserve.config import settings

logger = logging.getLogger(__name__)


@dataclass
class DriftState:
    production_scores: Deque[float] = field(default_factory=lambda: deque(maxlen=500))
    shadow_scores: Deque[float] = field(default_factory=lambda: deque(maxlen=500))
    drift_count: int = 0
    rollback_count: int = 0
    last_ks_statistic: float = 0.0
    last_ks_pvalue: float = 1.0
    last_psi: float = 0.0
    last_kl: float = 0.0
    last_wasserstein: float = 0.0
    drift_detected: bool = False


_state = DriftState()


def record_production(probability: float) -> None:
    _state.production_scores.append(probability)


def record_shadow(probability: float) -> None:
    _state.shadow_scores.append(probability)
    _maybe_evaluate()


def _maybe_evaluate() -> None:
    if len(_state.production_scores) < 30 or len(_state.shadow_scores) < 30:
        return

    prod = np.array(_state.production_scores)
    shadow = np.array(_state.shadow_scores)

    ks_result = stats.ks_2samp(prod, shadow)
    _state.last_ks_statistic = float(ks_result.statistic)
    _state.last_ks_pvalue = float(ks_result.pvalue)
    _state.last_psi = _compute_psi(prod, shadow)
    _state.last_kl = _compute_kl_divergence(prod, shadow)
    _state.last_wasserstein = float(stats.wasserstein_distance(prod, shadow))

    drift = (
        _state.last_ks_pvalue < settings.ks_drift_threshold
        or _state.last_psi > settings.psi_drift_threshold
    )

    if drift and not _state.drift_detected:
        _state.drift_count += 1
        logger.warning(
            "DRIFT DETECTED — KS p=%.4f, PSI=%.4f, KL=%.4f, W=%.4f",
            _state.last_ks_pvalue,
            _state.last_psi,
            _state.last_kl,
            _state.last_wasserstein,
        )
        if settings.rollback_on_drift:
            _state.rollback_count += 1
            logger.warning("Auto-rollback triggered — challenger traffic suppressed")

    _state.drift_detected = drift


def _compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index across equal-width bins [0, 1]."""
    eps = 1e-6
    bins = np.linspace(0, 1, buckets + 1)
    e_counts, _ = np.histogram(expected, bins=bins)
    a_counts, _ = np.histogram(actual, bins=bins)
    e_pct = (e_counts / len(expected)) + eps
    a_pct = (a_counts / len(actual)) + eps
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def _compute_kl_divergence(p: np.ndarray, q: np.ndarray, buckets: int = 10) -> float:
    """KL divergence D(P||Q) via histogram approximation."""
    eps = 1e-6
    bins = np.linspace(0, 1, buckets + 1)
    p_hist, _ = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bins, density=True)
    p_hist = p_hist + eps
    q_hist = q_hist + eps
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()
    return float(np.sum(p_hist * np.log(p_hist / q_hist)))


def get_state() -> DriftState:
    return _state


def reset_state() -> None:
    global _state
    _state = DriftState()
