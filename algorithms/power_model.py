"""Node-level simulated energy accounting; values are estimates, not readings."""

from gpu_model import normalize_server


def gpu_ready_time(server, arrival=0.0):
    """Earliest legal execution time given the GPU's initial power state."""
    node = normalize_server(server)
    if node["initial_state"] == "OFF":
        return max(float(arrival), node["min_off_minutes"] + node["startup_delay_s"] / 60.0)
    return float(arrival)


def break_even_idle_minutes(server):
    g = normalize_server(server)
    # Called after entering low-power mode: compare shutdown transition energy
    # with the energy the node would consume if it remained low-power.
    savings_w = max(0.0, g["low_power_w"])
    transition_wh = max(0.0, g["startup_energy_wh"] + g["shutdown_energy_wh"] +
                        g["idle_power_w"] * (g["startup_delay_s"] + g["shutdown_delay_s"]) / 3600)
    if savings_w <= 0:
        return None
    return transition_wh / (savings_w / 60.0)


def _segment(start, end, state, node):
    if end <= start:
        return None
    watts = {"ACTIVE": min(node["active_power_w"], node["power_limit_w"]), "IDLE": node["idle_power_w"],
             "LOW_POWER": node["low_power_w"], "OFF": 0.0,
             "STARTING": node["idle_power_w"], "SHUTTING_DOWN": node["idle_power_w"]}[state]
    return {"start": start, "end": end, "state": state,
            "energy_wh": watts * (end - start) / 60.0}


def _telemetry(node, jobs):
    points = sorted({t for j in jobs for t in (j["start"], j["fin"])})
    peaks = {"allocated_vram_gb": 0.0, "compute_utilization": 0.0,
             "memory_bandwidth_utilization": 0.0}
    for t in points:
        running = [j for j in jobs if j["start"] <= t < j["fin"]]
        peaks["allocated_vram_gb"] = max(peaks["allocated_vram_gb"], sum(j.get("vram", 0) for j in running))
        peaks["compute_utilization"] = max(peaks["compute_utilization"], sum(j.get("compute_demand", 0) for j in running) / max(.01, node["compute_capacity"]))
        peaks["memory_bandwidth_utilization"] = max(peaks["memory_bandwidth_utilization"], sum(j.get("memory_bandwidth_demand", 0) for j in running) / max(.01, node["memory_bandwidth"]))
    return peaks


def simulate_power(servers, schedule, horizon_minutes=None, allow_shutdown=True):
    """Build per-node state timelines and disaggregated Wh totals.

    Nodes are the power-management unit. Unused nodes can remain off. Active
    intervals are supplied by the resource scheduler; gaps are idle or low
    power. Terminal shutdown is selected only when the idle saving beats the
    configured startup+shutdown energy and minimum-on-time guard.
    """
    gpu_ids = [gpu.get("id") for gpu in servers]
    if len({type(gpu_id) for gpu_id in gpu_ids}) > 1:
        raise ValueError("GPU IDs must use one consistent type")
    if len({str(gpu_id) for gpu_id in gpu_ids}) != len(gpu_ids):
        raise ValueError("GPU IDs must be unique for telemetry")
    natural_horizon = max((s.get("fin", 0) for s in schedule), default=0)
    horizon = max(float(natural_horizon), float(horizon_minutes or natural_horizon + 30))
    per_node, totals = {}, {"active_wh": 0.0, "idle_wh": 0.0, "low_power_wh": 0.0,
                            "startup_wh": 0.0, "shutdown_wh": 0.0,
                            "startup_cost": 0.0, "shutdown_cost": 0.0,
                            "transition_delay_minutes": 0.0, "transitions": 0}
    for raw in servers:
        node = normalize_server(raw)
        jobs = sorted((x for x in schedule if x.get("sid") == raw.get("id")), key=lambda x: x["start"])
        timeline = []
        node_startup_wh = node_shutdown_wh = 0.0
        initial = node["initial_state"]
        if not raw.get("on", True):
            timeline.append(_segment(0, horizon, "OFF", node))
            per_node[str(raw.get("id"))] = {"name": raw.get("name"), "timeline": [t for t in timeline if t],
                "startup_wh": 0.0, "shutdown_wh": 0.0, "break_even_idle_minutes": break_even_idle_minutes(node),
                **_telemetry(node, jobs)}
            continue
        elif not jobs:
            low_start = min(horizon, node["low_power_after_minutes"])
            if initial == "OFF":
                timeline.append(_segment(0, horizon, "OFF", node))
                per_node[str(raw.get("id"))] = {"name": raw.get("name"), "timeline": [t for t in timeline if t],
                    "startup_wh": 0.0, "shutdown_wh": 0.0, "break_even_idle_minutes": break_even_idle_minutes(node),
                    **_telemetry(node, jobs)}
                continue
            shutdown_start = low_start + node["min_on_minutes"]
            shutdown_end = shutdown_start + node["shutdown_delay_s"] / 60
            off_duration = horizon - shutdown_end
            shutdown = (allow_shutdown and initial != "OFF" and
                        off_duration >= node["min_off_minutes"] + (break_even_idle_minutes(node) or 0))
            if shutdown:
                if low_start: timeline.append(_segment(0, low_start, "IDLE", node))
                if shutdown_start > low_start: timeline.append(_segment(low_start, shutdown_start, "LOW_POWER", node))
                timeline.append(_segment(shutdown_start, shutdown_end, "SHUTTING_DOWN", node))
                if horizon > shutdown_end: timeline.append(_segment(shutdown_end, horizon, "OFF", node))
                totals["shutdown_wh"] += node["shutdown_energy_wh"]
                totals["shutdown_cost"] += node["shutdown_cost"]
                node_shutdown_wh += node["shutdown_energy_wh"]
                totals["transitions"] += 1
                totals["transition_delay_minutes"] += node["shutdown_delay_s"] / 60
            else:
                if low_start: timeline.append(_segment(0, low_start, "IDLE", node))
                if horizon > low_start: timeline.append(_segment(low_start, horizon, "LOW_POWER", node))
        else:
            cursor = 0.0
            active_intervals = []
            for job in jobs:
                if active_intervals and job["start"] <= active_intervals[-1][1]:
                    active_intervals[-1][1] = max(active_intervals[-1][1], job["fin"])
                else:
                    active_intervals.append([job["start"], job["fin"]])
            if initial == "OFF":
                startup = node["startup_delay_s"] / 60
                startup_start = max(0.0, active_intervals[0][0] - startup)
                seg = _segment(0, startup_start, "OFF", node)
                if seg: timeline.append(seg)
                seg = _segment(startup_start, active_intervals[0][0], "STARTING", node)
                if seg: timeline.append(seg)
                totals["startup_wh"] += node["startup_energy_wh"]
                totals["startup_cost"] += node["startup_cost"]
                node_startup_wh += node["startup_energy_wh"]
                totals["transitions"] += 1
                totals["transition_delay_minutes"] += startup
                cursor = active_intervals[0][0]
            for active_start, active_end in active_intervals:
                gap = max(0.0, active_start - cursor)
                low_after = node["low_power_after_minutes"]
                low_start = cursor + min(gap, low_after)
                shutdown_delay = node["shutdown_delay_s"] / 60
                startup_delay = node["startup_delay_s"] / 60
                shutdown_start = cursor + low_after + node["min_on_minutes"]
                startup_start = active_start - startup_delay
                off_duration = startup_start - (shutdown_start + shutdown_delay)
                breakeven = break_even_idle_minutes(node) or 0
                cycle_off = (allow_shutdown and gap >= low_after + node["min_on_minutes"] + shutdown_delay +
                             node["min_off_minutes"] + startup_delay + breakeven and
                             off_duration >= node["min_off_minutes"] + breakeven)
                if cycle_off:
                    seg = _segment(cursor, min(active_start, low_start), "IDLE", node)
                    if seg: timeline.append(seg)
                    seg = _segment(low_start, shutdown_start, "LOW_POWER", node)
                    if seg: timeline.append(seg)
                    seg = _segment(shutdown_start, shutdown_start + shutdown_delay, "SHUTTING_DOWN", node)
                    if seg: timeline.append(seg)
                    seg = _segment(shutdown_start + shutdown_delay, startup_start, "OFF", node)
                    if seg: timeline.append(seg)
                    seg = _segment(startup_start, active_start, "STARTING", node)
                    if seg: timeline.append(seg)
                    totals["shutdown_wh"] += node["shutdown_energy_wh"]
                    totals["startup_wh"] += node["startup_energy_wh"]
                    totals["shutdown_cost"] += node["shutdown_cost"]
                    totals["startup_cost"] += node["startup_cost"]
                    node_shutdown_wh += node["shutdown_energy_wh"]
                    node_startup_wh += node["startup_energy_wh"]
                    totals["transitions"] += 2
                    totals["transition_delay_minutes"] += shutdown_delay + startup_delay
                else:
                    idle_state = "LOW_POWER" if initial == "LOW_POWER" and not timeline else "IDLE"
                    seg = _segment(cursor, min(active_start, low_start), idle_state, node)
                    if seg: timeline.append(seg)
                    if low_start < active_start:
                        seg = _segment(low_start, active_start, "LOW_POWER", node)
                        if seg: timeline.append(seg)
                seg = _segment(active_start, active_end, "ACTIVE", node)
                if seg: timeline.append(seg)
                cursor = max(cursor, active_end)
            remaining = max(0.0, horizon - cursor)
            breakeven = break_even_idle_minutes(node) or 0
            low_start = cursor + min(remaining, node["low_power_after_minutes"])
            stop_start = low_start + node["min_on_minutes"]
            stop_end = stop_start + node["shutdown_delay_s"] / 60
            shutdown = (allow_shutdown and horizon - stop_end >= node["min_off_minutes"] + breakeven)
            if shutdown:
                if low_start > cursor:
                    timeline.append(_segment(cursor, low_start, "IDLE", node))
                if stop_start > low_start:
                    timeline.append(_segment(low_start, stop_start, "LOW_POWER", node))
                timeline.append(_segment(stop_start, stop_end, "SHUTTING_DOWN", node))
                if horizon > stop_end: timeline.append(_segment(stop_end, horizon, "OFF", node))
                totals["shutdown_wh"] += node["shutdown_energy_wh"]
                totals["shutdown_cost"] += node["shutdown_cost"]
                node_shutdown_wh += node["shutdown_energy_wh"]
                totals["transitions"] += 1
                totals["transition_delay_minutes"] += node["shutdown_delay_s"] / 60
            elif remaining:
                if low_start > cursor: timeline.append(_segment(cursor, low_start, "IDLE", node))
                if horizon > low_start: timeline.append(_segment(low_start, horizon, "LOW_POWER", node))
        timeline = [t for t in timeline if t]
        for seg in timeline:
            key = {"ACTIVE": "active_wh", "IDLE": "idle_wh", "LOW_POWER": "low_power_wh"}.get(seg["state"])
            if key:
                totals[key] += seg["energy_wh"]
            elif seg["state"] == "STARTING":
                totals["startup_wh"] += seg["energy_wh"]
                node_startup_wh += seg["energy_wh"]
            elif seg["state"] == "SHUTTING_DOWN":
                totals["shutdown_wh"] += seg["energy_wh"]
                node_shutdown_wh += seg["energy_wh"]
        per_node[str(raw.get("id"))] = {"name": raw.get("name"), "timeline": timeline,
                                         "startup_wh": node_startup_wh, "shutdown_wh": node_shutdown_wh,
                                         "break_even_idle_minutes": break_even_idle_minutes(node),
                                         **_telemetry(node, jobs)}
    totals["total_wh"] = sum(totals[k] for k in ("active_wh", "idle_wh", "low_power_wh", "startup_wh", "shutdown_wh"))
    totals["total_kwh"] = totals["total_wh"] / 1000.0
    totals["horizon_minutes"] = horizon
    return {"nodes": per_node, "totals": totals}
