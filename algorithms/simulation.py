"""Deterministic discrete-event placement over simulated GPU resources."""

from gpu_model import execution_estimate, normalize_job, normalize_server, resource_compatible
from interference import compatibility_score
from power_model import gpu_ready_time, simulate_power


def _slot(job, server, start, duration, detail):
    finish = start + duration
    return {"jid": job.get("id"), "name": job["name"], "sid": server["id"],
            "sname": server["name"], "pri": job["pri"], "start": round(start, 4),
            "fin": round(finish, 4), "rt_estimate": job["rt"],
            "predicted_runtime": round(duration, 4), "cost": duration / 60 * server["cost"],
            "wait": max(0.0, start - job["arr"]),
            "vram": job["vram"], "compute_util": detail["compute_util"],
            "memory_util": detail["memory_util"],
            "compute_demand": job["compute_demand"], "memory_bandwidth_demand": job["memory_bandwidth_demand"],
            "slowdown": detail["slowdown"], "job_type": job["job_type"]}


def _fits_at(job, server, existing, start, allow_share):
    overlapping = [slot for slot in existing if slot["start"] < start + 1e-8 and slot["fin"] > start + 1e-8]
    if overlapping and not allow_share:
        return False, [], 0.0, None
    peers = [normalize_job(s["_job"]) for s in overlapping]
    group = [job, *peers]
    if not resource_compatible(group, server):
        return False, [], 0.0, None
    duration, detail = execution_estimate(job, server, peers)
    # A conservative fixed point: the job cannot outlive a peer whose resource
    # use makes the combined workload infeasible. Peer duration affects only
    # throughput, not the capacity check itself.
    if overlapping and not all(s.get("sharing", True) for s in group):
        return False, [], 0.0, None
    return True, overlapping, duration, detail


def _interval_fits(rows, server):
    """Check the complete candidate interval, including future arrivals/jobs.

    Checking each peer independently is insufficient: three jobs can each fit
    with the other two while their combined demand exceeds a node budget.
    """
    events = {t for row in rows for t in (row["start"], row["fin"])}
    ordered = sorted(events)
    for left, right in zip(ordered, ordered[1:]):
        if right <= left:
            continue
        midpoint = (left + right) / 2
        active = [row for row in rows if row["start"] <= midpoint < row["fin"]]
        if len(active) <= 1:
            continue
        peers = [normalize_job(row["_job"]) for row in active]
        group = peers
        if not server["supports_sharing"] or not all(peer.get("sharing", True) for peer in peers):
            return False, "sharing is disabled on the GPU or by a participating job"
        if not resource_compatible(group, server):
            return False, "aggregate VRAM, compute, or memory-bandwidth capacity would be exceeded"
    return True, None


def _recompute_durations(rows, server):
    """Apply contention symmetrically until peer sets and durations stabilize.

    Peer sets only grow during this fixed point, so a duration extension cannot
    silently make an earlier job appear to finish sooner.
    """
    peer_sets = {i: set() for i in range(len(rows))}
    for _ in range(len(rows) * len(rows) + 2):
        changed = False
        for i, row in enumerate(rows):
            for j, other in enumerate(rows):
                if i != j and row["start"] < other["fin"] and other["start"] < row["fin"]:
                    if j not in peer_sets[i]:
                        peer_sets[i].add(j)
                        changed = True
            peers = [rows[j]["_job"] for j in sorted(peer_sets[i])]
            duration, detail = execution_estimate(row["_job"], server, peers)
            duration = round(duration, 4)
            finish = round(row["start"] + duration, 4)
            if duration != row["predicted_runtime"] or finish != row["fin"]:
                changed = True
                row["predicted_runtime"] = duration
                row["fin"] = finish
                row["cost"] = duration / 60 * server["cost"]
                row["compute_util"] = detail["compute_util"]
                row["memory_util"] = detail["memory_util"]
                row["slowdown"] = detail["slowdown"]
        if not changed:
            return True
    return False


def schedule_gpu_aware(snapshot, order=None, objective="balanced", consolidation=False,
                       allow_sharing=True, assignments=None, planning_horizon_minutes=None):
    jobs = [normalize_job(j) for j in snapshot["jobs"]]
    servers = [normalize_server(s) for s in snapshot["servers"]]
    order = [normalize_job(j) for j in (order or sorted(jobs, key=lambda j: (j["arr"], -{"High": 3, "Medium": 2, "Low": 1}.get(j["pri"], 1), j.get("id", 0))))]
    # Preserve references from normalized jobs for interval overlap checks.
    schedule, unscheduled = [], []
    for ji, job in enumerate(order):
        candidates = []
        rejections = []
        dynamic_rejections = []
        for si, server in enumerate(servers):
            if not server.get("on", True):
                rejections.append(f"{server['name']}: disabled")
                continue
            if job["vram"] > server["vram"]:
                rejections.append(f"{server['name']}: needs {job['vram']} GB VRAM, has {server['vram']} GB")
                continue
            if job["compute_demand"] > server["compute_capacity"]:
                rejections.append(f"{server['name']}: compute demand {job['compute_demand']} exceeds {server['compute_capacity']}")
                continue
            if job["memory_bandwidth_demand"] > server["memory_bandwidth"]:
                rejections.append(f"{server['name']}: bandwidth demand {job['memory_bandwidth_demand']} GB/s exceeds {server['memory_bandwidth']} GB/s")
                continue
            existing = [s for s in schedule if s["sid"] == server["id"]]
            if isinstance(assignments, dict):
                preferred = assignments.get(job["id"])
            else:
                # Retain compatibility with callers using the previous ordered
                # list form; optimizers now pass stable job-ID keyed mappings.
                preferred = assignments[ji] if assignments is not None and ji < len(assignments) else None
            earliest = gpu_ready_time(server, job["arr"])
            points = sorted(set([earliest] +
                                [max(earliest, event) for s in existing for event in (s["start"], s["fin"])]))
            for start in points:
                fits, peers, duration, detail = _fits_at(job, server, existing, start, allow_sharing and job["sharing"])
                if not fits:
                    if any(s["start"] < start + 1e-8 and s["fin"] > start + 1e-8 for s in existing):
                        dynamic_rejections.append(f"{server['name']}: sharing or aggregate resource limits prevent overlap")
                    continue
                slot = _slot(job, server, start, duration, detail)
                slot["_job"] = job
                trial_rows = [{**row} for row in existing] + [slot]
                if not _recompute_durations(trial_rows, server):
                    dynamic_rejections.append(f"{server['name']}: contention estimates did not stabilize")
                    continue
                interval_fits, interval_reason = _interval_fits(trial_rows, server)
                if not interval_fits:
                    dynamic_rejections.append(f"{server['name']}: {interval_reason}")
                    continue
                slot = next(row for row in trial_rows if row["jid"] == job["id"] and row["start"] == round(start, 4))
                duration = slot["predicted_runtime"]
                peers = [row["_job"] for row in trial_rows if row is not slot and
                         row["start"] < slot["fin"] and row["fin"] > slot["start"]]
                first_start = not existing or start < min(row["start"] for row in existing)
                active_wh = duration / 60 * min(server["active_power_w"], server["power_limit_w"])
                startup_wh = server["startup_energy_wh"] if server["initial_state"] == "OFF" and first_start else 0.0
                energy_wh = active_wh + startup_wh
                rental = duration / 60 * server["cost"]
                electricity = energy_wh / 1000 * server["electricity_tariff"]
                finish = start + duration
                wait = max(0.0, start - job["arr"])
                if objective == "energy": score = (energy_wh, finish, si)
                elif objective == "cost": score = (rental + electricity, finish, si)
                elif objective == "makespan": score = (finish, rental, si)
                elif objective == "waiting": score = (wait, finish, si)
                elif objective == "priority": score = (wait * {"High": 3, "Medium": 2, "Low": 1}.get(job["pri"], 1), finish, si)
                else: score = (rental + electricity + .02 * energy_wh + finish + wait, finish, si)
                score = (*score, -compatibility_score([job, *peers], server))
                if consolidation:
                    already_used = bool(existing) or server["initial_state"] == "ACTIVE"
                    score = (0 if already_used else 1, start + duration, duration / 60 * server["cost"], si)
                if preferred is not None:
                    score = (0 if preferred == si else 1, *score)
                candidates.append((score, slot, trial_rows))
                break
        if not candidates:
            failure_details = rejections + dynamic_rejections
            if not servers:
                reason = "No GPU servers are configured."
            elif failure_details:
                reason = "No feasible GPU placement. " + "; ".join(dict.fromkeys(failure_details))
            else:
                reason = "No feasible placement after checking sharing, resource contention, and power-start timing."
            unscheduled.append({"name": job["name"], "pri": job["pri"],
                                "reason": reason})
            continue
        candidates.sort(key=lambda x: x[0])
        _, _, best_rows = candidates[0]
        schedule = [row for row in schedule if row["sid"] != best_rows[0]["sid"]] + best_rows
    for row in schedule:
        row.pop("_job", None)
    horizon = planning_horizon_minutes
    power = simulate_power(servers, schedule, horizon, allow_shutdown=True)
    return {"sch": schedule, "uns": unscheduled, "power": power,
            "status": "deterministic GPU resource and power simulation",
            "planning_horizon_minutes": power["totals"]["horizon_minutes"]}
