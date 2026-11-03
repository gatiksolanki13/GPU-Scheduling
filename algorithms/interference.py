"""Resource-fit and slowdown helpers for simulated co-location."""

from gpu_model import normalize_job, normalize_server, resource_compatible


def compatibility_score(jobs, server):
    """Return a 0..1 score; complementary compute/memory jobs score better."""
    g = normalize_server(server)
    normalized = [normalize_job(j) for j in jobs]
    if not resource_compatible(normalized, g):
        return 0.0
    compute = sum(j["compute_demand"] for j in normalized) / max(g["compute_capacity"], 0.01)
    memory = sum(j["memory_bandwidth_demand"] for j in normalized) / max(g["memory_bandwidth"], 0.01)
    # Complementarity is the separation between normalized compute and memory
    # pressure; bottleneck pressure is penalized more strongly.
    pressure = max(compute, memory)
    complement = abs(compute - memory)
    return max(0.0, min(1.0, 1.0 - pressure * 0.65 + min(0.2, complement * 0.1)))


def slowdown_factor(jobs, server):
    score = compatibility_score(jobs, server)
    return float("inf") if score <= 0 else 1.0 + (1.0 - score) * 1.5
