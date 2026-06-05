"""Prometheus metrics registry for ShadowServe."""
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    CollectorRegistry,
    CONTENT_TYPE_LATEST,
    generate_latest,
)

registry = CollectorRegistry(auto_describe=True)

# Latency histograms per route target
inference_latency = Histogram(
    "shadowserve_inference_latency_ms",
    "End-to-end inference latency in milliseconds",
    labelnames=["model_version", "route_target"],
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
    registry=registry,
)

# Request counters
request_total = Counter(
    "shadowserve_requests_total",
    "Total inference requests",
    labelnames=["route_target"],
    registry=registry,
)

shadow_total = Counter(
    "shadowserve_shadow_requests_total",
    "Total shadow (async forked) requests",
    registry=registry,
)

rollback_total = Counter(
    "shadowserve_rollbacks_total",
    "Number of auto-rollbacks triggered by drift",
    registry=registry,
)

# Drift gauges
ks_statistic = Gauge(
    "shadowserve_ks_statistic",
    "Kolmogorov-Smirnov test statistic (production vs shadow)",
    registry=registry,
)

ks_pvalue = Gauge(
    "shadowserve_ks_pvalue",
    "KS test p-value",
    registry=registry,
)

psi_score = Gauge(
    "shadowserve_psi_score",
    "Population Stability Index",
    registry=registry,
)

kl_divergence = Gauge(
    "shadowserve_kl_divergence",
    "KL divergence D(production||shadow)",
    registry=registry,
)

wasserstein_distance = Gauge(
    "shadowserve_wasserstein_distance",
    "Wasserstein (Earth Mover's) distance",
    registry=registry,
)

drift_detected = Gauge(
    "shadowserve_drift_detected",
    "1 if drift is currently detected, 0 otherwise",
    registry=registry,
)

canary_weight_gauge = Gauge(
    "shadowserve_canary_weight",
    "Current canary traffic weight [0, 1]",
    registry=registry,
)


def metrics_output() -> tuple[bytes, str]:
    return generate_latest(registry), CONTENT_TYPE_LATEST
