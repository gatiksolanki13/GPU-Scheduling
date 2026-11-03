"""Configurable, explicitly simulated GPU hardware and workload defaults."""

GPU_DEFAULTS = {
    "compute_capacity": 1.0, "memory_bandwidth": 900.0, "pcie_bandwidth": 24.0,
    "active_power_w": 280.0, "idle_power_w": 90.0, "low_power_w": 24.0,
    "startup_energy_wh": 18.0, "shutdown_energy_wh": 3.0,
    "startup_cost": 0.0, "shutdown_cost": 0.0,
    "startup_delay_s": 35.0, "shutdown_delay_s": 12.0,
    "thermal_threshold_c": 85.0, "temperature_c": 35.0, "power_limit_w": 320.0,
    "min_on_minutes": 10.0, "min_off_minutes": 5.0,
    "compute_efficiency": 0.9, "memory_efficiency": 0.85, "pcie_efficiency": 0.9,
    "overlap_compute_memory": 0.7, "supports_sharing": True,
    "power_limit_w": 320.0,
    "initial_state": None, "power_state": "IDLE",
    "allocated_vram": 0.0, "compute_utilization": 0.0,
    "low_power_after_minutes": 8.0, "electricity_tariff": 8.0,
}

JOB_DEFAULTS = {
    "job_type": "Training", "compute_demand": 1.0,
    "memory_bandwidth_demand": 0.0, "iterations": 1.0, "compute_work": None,
    "memory_work_gb": None, "sharing": True,
}

PROFILES = {
    "Illustrative balanced GPU": {"vram": 16, "compute_capacity": 1.0,
        "memory_bandwidth": 900, "pcie_bandwidth": 24, "cost": 120,
        "active_power_w": 280, "idle_power_w": 90, "low_power_w": 24},
    "Illustrative compact GPU": {"vram": 8, "compute_capacity": 0.55,
        "memory_bandwidth": 420, "pcie_bandwidth": 12, "cost": 60,
        "active_power_w": 150, "idle_power_w": 48, "low_power_w": 15},
    "Illustrative high-capacity GPU": {"vram": 32, "compute_capacity": 2.0,
        "memory_bandwidth": 1600, "pcie_bandwidth": 48, "cost": 260,
        "active_power_w": 480, "idle_power_w": 135, "low_power_w": 30},
}


def normalize_server(server):
    result = {**GPU_DEFAULTS, **server}
    initial = server.get("initial_state")
    if not initial:
        initial = server.get("power_state") if "power_state" in server else ("IDLE" if server.get("on", True) else "OFF")
    result["initial_state"] = str(initial).upper()
    if result["initial_state"] not in {"OFF", "IDLE", "LOW_POWER", "ACTIVE"}:
        result["initial_state"] = "IDLE" if result.get("on", True) else "OFF"
    return result


def normalize_job(job):
    result = {**JOB_DEFAULTS, **job}
    result.pop("dl", None)  # Ignore legacy deadline fields.
    result.setdefault("rt", 1.0)
    result.setdefault("arr", 0.0)
    result.setdefault("pri", "Medium")
    # Legacy runtime is preserved on a unit-capacity GPU. New workloads can
    # specify compute_work directly or use the runtime estimate as calibrated
    # reference work scaled by iteration count. Demand controls fit/contention.
    if result["compute_work"] is None:
        result["compute_work"] = max(0.001, float(result.get("rt", 1))) * max(0.001, float(result["iterations"]))
    if result["memory_work_gb"] is None:
        result["memory_work_gb"] = max(0.0, float(result["memory_bandwidth_demand"])) * max(0.0, float(result.get("rt", 0))) * 60
    return result


def execution_estimate(job, server, concurrent=()):
    """Return an explainable bottleneck estimate in minutes for one node.

    Compute and memory can overlap by a configurable fraction. A thermal and
    co-location factor reduces effective throughput.
    """
    g, j = normalize_server(server), normalize_job(job)
    peers = [normalize_job(p) for p in concurrent]
    comp_load = j["compute_demand"] + sum(p["compute_demand"] for p in peers)
    bw_load = j["memory_bandwidth_demand"] + sum(p["memory_bandwidth_demand"] for p in peers)
    interference = max(1.0, comp_load / max(g["compute_capacity"], 0.01),
                      bw_load / max(g["memory_bandwidth"], 0.01))
    thermal = 0.82 if g["temperature_c"] >= g["thermal_threshold_c"] else 1.0
    power_factor = min(1.0, max(0.25, g["power_limit_w"] / max(1.0, g["active_power_w"])))
    comp_cap = max(0.01, g["compute_capacity"] * g["compute_efficiency"] * thermal * power_factor)
    mem_cap = max(0.01, g["memory_bandwidth"] * g["memory_efficiency"] * thermal)
    # Workload classes apply modest illustrative calibration multipliers on
    # top of the user-supplied demand volumes; they are not hardware profiles.
    kind = j["job_type"].lower()
    compute_factor = 1.25 if kind == "compute intensive" else (0.9 if kind == "inference" else 1.0)
    memory_factor = 1.25 if kind == "memory intensive" else (1.1 if kind == "batch processing" else 1.0)
    comp_min = j["compute_work"] * compute_factor / comp_cap
    mem_min = j["memory_work_gb"] * memory_factor / mem_cap / 60
    overlap = min(1.0, max(0.0, float(g["overlap_compute_memory"])))
    device_min = max(comp_min, mem_min) * overlap + (comp_min + mem_min) * (1 - overlap)
    peer_comp = sum(p["compute_demand"] for p in peers) / max(g["compute_capacity"], 0.01)
    peer_mem = sum(p["memory_bandwidth_demand"] for p in peers) / max(g["memory_bandwidth"], 0.01)
    slowdown = 1.0 + 0.8 * min(1.0, comp_load / max(g["compute_capacity"], .01)) * peer_comp + 1.2 * min(1.0, bw_load / max(g["memory_bandwidth"], .01)) * peer_mem
    return max(0.01, device_min * slowdown), {
        "compute_min": comp_min, "memory_min": mem_min,
        "slowdown": slowdown, "compute_util": min(1.0, j["compute_demand"] / g["compute_capacity"]),
        "memory_util": min(1.0, j["memory_bandwidth_demand"] / g["memory_bandwidth"]),
    }


def resource_compatible(jobs, server):
    g = normalize_server(server)
    normalized = [normalize_job(j) for j in jobs]
    return (sum(j["vram"] for j in normalized) <= g["vram"] and
            sum(j["compute_demand"] for j in normalized) <= g["compute_capacity"] and
            sum(j["memory_bandwidth_demand"] for j in normalized) <= g["memory_bandwidth"] and
            (len(normalized) <= 1 or (g["supports_sharing"] and all(j.get("sharing", True) for j in normalized))))
