import math
from gpu_model import normalize_server
from power_model import simulate_power

PRIORITY_WEIGHT = {"High": 3, "Medium": 2, "Low": 1}


def js_round(value):
    """Round half up, matching the numbers shown in the interface."""
    return math.floor(value + 0.5)


def number_text(value):
    """Format a number without a trailing .0 so traces read cleanly."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def objective_weights(objective):
    return {"cost": (1, 0, 0, 0, 0), "energy": (0, 1, 0, 0, 0),
            "makespan": (0, 0, 1, 0, 0), "waiting": (0, 0, 0, 1, 0),
            "priority": (0, 0, 0, 0, 1),
            "balanced": (1, 0.02, 10, 1, 1)}.get(objective, (1, 0.02, 10, 1, 1))


def compute_metrics(schedule, unscheduled, servers, power=None):
    nodes = [normalize_server(s) for s in servers]
    enabled = sum(1 for s in nodes if s.get("on", True))
    makespan = max((x["fin"] for x in schedule), default=0)
    busy_job_minutes = sum(x["fin"] - x["start"] for x in schedule)
    wall_active_minutes = 0.0
    for node in nodes:
        if not node.get("on", True):
            continue
        intervals = sorted((x["start"], x["fin"]) for x in schedule if x["sid"] == node["id"])
        if not intervals:
            continue
        start, finish = intervals[0]
        for next_start, next_finish in intervals[1:]:
            if next_start <= finish:
                finish = max(finish, next_finish)
            else:
                wall_active_minutes += finish - start
                start, finish = next_start, next_finish
        wall_active_minutes += finish - start
    power = power or simulate_power(nodes, schedule)
    totals = power["totals"]
    by_id = {str(s["id"]): s for s in nodes}
    electricity_cost = 0.0
    for node_id, result in power.get("nodes", {}).items():
        wh = sum(seg.get("energy_wh", 0) for seg in result.get("timeline", []))
        # Node transition totals include both transition fees in Wh and the
        # STARTING/SHUTTING_DOWN segment draw already present in the timeline.
        node = by_id.get(str(node_id), {})
        tariff = float(node.get("electricity_tariff", 8.0))
        startup_segment_wh = sum(seg.get("energy_wh", 0) for seg in result.get("timeline", [])
                                 if seg.get("state") == "STARTING")
        shutdown_segment_wh = sum(seg.get("energy_wh", 0) for seg in result.get("timeline", [])
                                  if seg.get("state") == "SHUTTING_DOWN")
        wh += max(0.0, result.get("startup_wh", 0) - startup_segment_wh)
        wh += max(0.0, result.get("shutdown_wh", 0) - shutdown_segment_wh)
        electricity_cost += wh / 1000 * tariff
    powered_node_minutes = sum(max(0.0, seg["end"] - seg["start"])
                               for node in power.get("nodes", {}).values()
                               for seg in node.get("timeline", []) if seg.get("state") != "OFF")
    denom = max(1e-9, powered_node_minutes)
    def avg_resource(key):
        return min(1.0, sum((x.get(key, 0) or 0) * (x["fin"] - x["start"]) for x in schedule) / denom)
    rental_cost = sum(float(by_id.get(str(node_id), {}).get("cost", 0)) *
                      sum(max(0.0, seg["end"] - seg["start"]) for seg in node_result.get("timeline", []) if seg.get("state") != "OFF") / 60
                      for node_id, node_result in power.get("nodes", {}).items())
    runtime_cost = sum(float(x.get("cost", 0)) for x in schedule)
    transition_cost = totals.get("startup_cost", 0.0) + totals.get("shutdown_cost", 0.0)
    legacy = {
        "scheduled": len(schedule),
        "unscheduled": len(unscheduled),
        "makespan": makespan,
        "cost": runtime_cost,
        "avgWait": sum(x["wait"] for x in schedule) / len(schedule) if schedule else 0,
        "util": wall_active_minutes / (enabled * makespan) if makespan and enabled else 0,
        "busyJobTimeUtil": busy_job_minutes / (enabled * makespan) if makespan and enabled else 0,
    }
    legacy.update({
        "activeEnergyWh": totals["active_wh"], "idleEnergyWh": totals["idle_wh"],
        "lowPowerEnergyWh": totals["low_power_wh"], "startupEnergyWh": totals["startup_wh"],
        "shutdownEnergyWh": totals["shutdown_wh"], "totalEnergyWh": totals["total_wh"],
        "totalEnergyKWh": totals["total_kwh"], "electricityCost": electricity_cost,
        "gpuRuntimeCost": runtime_cost, "rentalCost": rental_cost, "transitionCost": transition_cost,
        "totalOperatingCost": rental_cost + electricity_cost + transition_cost,
        "avgComputeUtil": avg_resource("compute_util"),
        "avgVramUtil": min(1.0, sum(x.get("vram", 0) / max(1, by_id.get(str(x["sid"]), {}).get("vram", 1)) * (x["fin"] - x["start"]) for x in schedule) / denom),
        "avgMemoryBandwidthUtil": avg_resource("memory_util"),
        "activeNodes": len({x["sid"] for x in schedule}),
        "powerTransitions": totals["transitions"],
        "transitionDelayMinutes": totals["transition_delay_minutes"],
        "avgCompletionTime": sum(x["fin"] for x in schedule) / len(schedule) if schedule else 0,
        "avgWaitingTime": sum(x["wait"] for x in schedule) / len(schedule) if schedule else 0,
        "priorityWeightedWait": sum(x["wait"] * PRIORITY_WEIGHT[x["pri"]] for x in schedule),
        "contentionPenalty": sum(max(0, x.get("slowdown", 1) - 1) * (x["fin"] - x["start"]) for x in schedule),
        "energy": totals,
    })
    return legacy
