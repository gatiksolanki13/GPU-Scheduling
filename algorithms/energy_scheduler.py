"""Energy consolidation and GPU resource-aware deterministic heuristics."""

from simulation import schedule_gpu_aware


def energy_aware(snapshot, objective="balanced", options=None):
    options = options or {}
    result = schedule_gpu_aware(snapshot, objective=objective, consolidation=True,
                                allow_sharing=True,
                                planning_horizon_minutes=options.get("planning_horizon_minutes"))
    result["status"] = "Energy-aware consolidation; active nodes consolidated and idle nodes power-managed"
    result["breakEvenMinutes"] = {str(s["id"]): result["power"]["nodes"][str(s["id"])]["break_even_idle_minutes"]
                                  for s in snapshot["servers"] if str(s["id"]) in result["power"]["nodes"]}
    return result


def resource_aware(snapshot, objective="balanced", options=None):
    result = schedule_gpu_aware(snapshot, objective=objective, consolidation=False,
                                allow_sharing=True,
                                planning_horizon_minutes=(options or {}).get("planning_horizon_minutes"))
    result["status"] = "GPU resource-aware best-fit placement with simulated co-location checks"
    return result
