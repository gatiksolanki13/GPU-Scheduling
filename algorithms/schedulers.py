import time

from metrics import PRIORITY_WEIGHT, compute_metrics, objective_weights
from gpu_model import execution_estimate, normalize_job, normalize_server
from energy_scheduler import energy_aware, resource_aware
from genetic import run_genetic
from nsga2 import run_nsga2
from power_model import gpu_ready_time, simulate_power

TIME_LIMIT_SECONDS = 3


def make_slot(job, server, start):
    job = normalize_job(job)
    server = normalize_server(server)
    duration, detail = execution_estimate(job, server)
    finish = start + duration
    return {
        "jid": job["id"],
        "name": job["name"],
        "sid": server["id"],
        "sname": server["name"],
        "pri": job["pri"],
        "start": start,
        "fin": finish,
        "rt_estimate": job["rt"],
        "predicted_runtime": duration,
        "cost": duration / 60 * server["cost"],
        "wait": start - job["arr"],
        "vram": job["vram"],
        "compute_util": detail["compute_util"],
        "memory_util": detail["memory_util"],
        "compute_demand": job["compute_demand"],
        "memory_bandwidth_demand": job["memory_bandwidth_demand"],
        "slowdown": detail["slowdown"],
    }


def unscheduled_entry(job, servers, additional_reasons=None):
    reasons = []
    enabled = [server for server in servers if server.get("on", True)]
    for server in servers:
        if not server.get("on", True):
            reasons.append(f"{server.get('name', server.get('id', 'GPU'))}: disabled")
            continue
        node = normalize_server(server)
        compatible = True
        if job["vram"] > node["vram"]:
            reasons.append(f"{node['name']}: insufficient VRAM ({job['vram']} GB required, {node['vram']} GB available)")
            compatible = False
        if job["compute_demand"] > node["compute_capacity"]:
            reasons.append(f"{node['name']}: insufficient compute capacity ({job['compute_demand']} required, {node['compute_capacity']} available)")
            compatible = False
        if job["memory_bandwidth_demand"] > node["memory_bandwidth"]:
            reasons.append(f"{node['name']}: insufficient memory bandwidth ({job['memory_bandwidth_demand']} GB/s required, {node['memory_bandwidth']} GB/s available)")
            compatible = False
        if compatible:
            reasons.append(f"{node['name']}: individually feasible, but no conflict-free slot was found")
    if not servers:
        reason = "No GPU servers are configured."
    elif not enabled:
        reason = "No enabled GPU is available; " + "; ".join(reasons)
    elif reasons:
        reason = "No individually compatible enabled GPU; " + "; ".join(reasons)
    else:
        reason = "No feasible slot was found due to sharing or concurrent resource constraints."
    if additional_reasons:
        reason += " Scheduling checks: " + "; ".join(dict.fromkeys(additional_reasons))
    return {"name": job["name"], "pri": job["pri"], "reason": reason}


def earliest_finish(slot):
    return (slot["fin"], slot["cost"], slot["sid"])


def list_schedule(order, pick_key, snap):
    free = {s["id"]: 0 for s in snap["servers"]}
    schedule, unscheduled = [], []
    for job in order:
        options = [
            make_slot(job, s, max(free[s["id"]], gpu_ready_time(s, job["arr"])))
            for s in snap["servers"]
            if s["on"] and s["vram"] >= job["vram"]
            and normalize_job(job)["compute_demand"] <= normalize_server(s)["compute_capacity"]
            and normalize_job(job)["memory_bandwidth_demand"] <= normalize_server(s)["memory_bandwidth"]
        ]
        if not options:
            unscheduled.append(unscheduled_entry(job, snap["servers"]))
            continue
        options.sort(key=pick_key)
        best = options[0]
        free[best["sid"]] = best["fin"]
        schedule.append(best)
    return {"sch": schedule, "uns": unscheduled}


def fcfs(snap, objective):
    order = sorted(snap["jobs"], key=lambda j: (j["arr"], j["id"]))
    return list_schedule(order, earliest_finish, snap)


def sjf(snap, objective):
    order = sorted(snap["jobs"], key=lambda j: (j["rt"], j["id"]))
    return list_schedule(order, earliest_finish, snap)


def priority_schedule(snap, objective):
    def rank(job):
        return (-PRIORITY_WEIGHT[job["pri"]], job["arr"], job["id"])

    return list_schedule(sorted(snap["jobs"], key=rank), earliest_finish, snap)


def score(schedule, unscheduled, servers, objective, custom_weights=None, planning_horizon_minutes=None):
    power = simulate_power(servers, schedule, planning_horizon_minutes)
    metrics = compute_metrics(schedule, unscheduled, servers, power)
    total = 0
    for item in unscheduled:
        total += 1e5 * PRIORITY_WEIGHT[item["pri"]]
    selected_weights = custom_weights or {}
    default = objective_weights(objective)
    if selected_weights:
        wc, we, wm, ww, wp = (selected_weights.get("cost", default[0]), selected_weights.get("energy", default[1]),
                               selected_weights.get("makespan", default[2]), selected_weights.get("waiting", default[3]),
                               selected_weights.get("priority", default[4]))
    else:
        wc, we, wm, ww, wp = default
    value = (wc * metrics["totalOperatingCost"] + we * metrics["totalEnergyKWh"] +
             wm * metrics["makespan"] + ww * metrics["avgWaitingTime"] +
             wp * metrics["priorityWeightedWait"])
    return total + value


def exact_search(snap, objective, options=None):
    options = options or {}
    started = time.perf_counter()
    jobs, servers = snap["jobs"], snap["servers"]
    time_limit = max(0.05, min(10.0, float(options.get("time_limit_seconds", TIME_LIMIT_SECONDS))))
    candidates = [
        [s for s in servers if s["on"] and s["vram"] >= j["vram"]
         and normalize_job(j)["compute_demand"] <= normalize_server(s)["compute_capacity"]
         and normalize_job(j)["memory_bandwidth_demand"] <= normalize_server(s)["memory_bandwidth"]] for j in jobs
    ]
    unscheduled = [unscheduled_entry(j, servers) for j, c in zip(jobs, candidates) if not c]
    chosen = [None] * len(jobs)
    state = {"best": None, "best_score": float("inf"), "nodes": 0, "cut": False}

    def build_schedule():
        per_server = {}
        for i in range(len(jobs)):
            if chosen[i] is not None:
                per_server.setdefault(chosen[i]["id"], []).append(i)
        schedule = []
        for server_id in sorted(per_server):
            indexes = per_server[server_id]
            indexes.sort(
                key=lambda i: (
                    -PRIORITY_WEIGHT[jobs[i]["pri"]],
                    jobs[i]["arr"],
                )
            )
            clock = 0
            for i in indexes:
                node = normalize_server(chosen[i])
                ready = gpu_ready_time(node, jobs[i]["arr"]) if clock == 0 else jobs[i]["arr"]
                slot = make_slot(jobs[i], chosen[i], max(clock, ready))
                clock = slot["fin"]
                schedule.append(slot)
        return schedule

    def search(i):
        if state["cut"]:
            return
        if i == len(jobs):
            state["nodes"] += 1
            schedule = build_schedule()
            value = score(schedule, unscheduled, servers, objective, options.get("weights"),
                          options.get("planning_horizon_minutes"))
            if value < state["best_score"]:
                state["best_score"], state["best"] = value, schedule
            if state["nodes"] & 1023 == 0:
                if time.perf_counter() - started > TIME_LIMIT_SECONDS:
                    state["cut"] = True
            return
        if not candidates[i]:
            chosen[i] = None
            search(i + 1)
            return
        for server in candidates[i]:
            # Check before descending into another assignment. If the final
            # leaf just completed, no pending branch remains and the result is
            # still correctly reported as exhaustive.
            if state["nodes"] and time.perf_counter() - started >= time_limit:
                state["cut"] = True
                return
            chosen[i] = server
            search(i + 1)

    search(0)
    status = (
        "time limit \u2013 best feasible found"
        if state["cut"]
        else "exhaustive optimum over GPU assignments with fixed priority/arrival order per GPU"
    )
    return {
        "sch": state["best"] or [],
        "uns": unscheduled,
        "status": status,
        "nodes": state["nodes"],
        "search_space": "GPU assignments; per-GPU order fixed by priority then arrival",
    }


ALGORITHMS = {
    "fcfs": ("FCFS", fcfs),
    "sjf": ("SJF", sjf),
    "prio": ("Priority + arrival", priority_schedule),
    "exact": ("Exact search", exact_search),
    "energy": ("Energy-Aware Consolidation", None),
    "gpu_aware": ("GPU Resource-Aware Placement", None),
    "genetic": ("Genetic Algorithm", None),
    "nsga2": ("NSGA-II Pareto", None),
}


def run_algorithm(key, snap, objective, options=None):
    if key not in ALGORITHMS:
        raise ValueError(f"unknown algorithm: {key}")
    name, solver = ALGORITHMS[key]
    started = time.perf_counter()
    engine_snap = {"servers": [normalize_server(s) for s in snap["servers"]],
                   "jobs": [normalize_job(j) for j in snap["jobs"]]}
    if key == "energy": result = energy_aware(engine_snap, objective, options)
    elif key == "gpu_aware": result = resource_aware(engine_snap, objective, options)
    elif key == "genetic": result = run_genetic(engine_snap, objective, options)
    elif key == "nsga2": result = run_nsga2(engine_snap, objective, options)
    elif key == "exact": result = exact_search(engine_snap, objective, options)
    else: result = solver(engine_snap, objective)
    if "power" not in result:
        result["power"] = simulate_power(engine_snap["servers"], result["sch"],
                                         (options or {}).get("planning_horizon_minutes"))
    result["ms"] = (time.perf_counter() - started) * 1000
    result["algo"] = name
    result.setdefault("status", "heuristic")
    result["m"] = compute_metrics(result["sch"], result["uns"], engine_snap["servers"], result.get("power"))
    result["snap"] = {**snap, "jobs": [{k: v for k, v in job.items() if k != "dl"} for job in snap["jobs"]]}
    return result
