"""Deterministic-seed NSGA-II over five simulator metrics (including priority-weighted wait)."""

import random
import time
import math

from metrics import compute_metrics
from simulation import schedule_gpu_aware


def _objectives(result):
    m = result["m"]
    return (m["totalEnergyKWh"], m["totalOperatingCost"], m["makespan"],
            m["avgWaitingTime"], m["priorityWeightedWait"] + m["unscheduled"] * 10000)


def _dominates(a, b):
    # Feasibility is a lexicographic constraint: a solution scheduling more
    # jobs always outranks one scheduling fewer, regardless of cost tradeoffs.
    if a.get("scheduled") is not None and b.get("scheduled") is not None and a["scheduled"] != b["scheduled"]:
        return a["scheduled"] > b["scheduled"]
    objectives_a = a.get("objectives", a)
    objectives_b = b.get("objectives", b)
    return all(x <= y for x, y in zip(objectives_a, objectives_b)) and any(
        x < y for x, y in zip(objectives_a, objectives_b))


def _fronts(pop):
    dominates, count, fronts = {}, {}, [[]]
    for p in range(len(pop)):
        dominates[p], count[p] = [], 0
        for q in range(len(pop)):
            if p == q: continue
            if _dominates(pop[p], pop[q]): dominates[p].append(q)
            elif _dominates(pop[q], pop[p]): count[p] += 1
        if count[p] == 0: pop[p]["rank"] = 0; fronts[0].append(p)
    i = 0
    while i < len(fronts) and fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in dominates[p]:
                count[q] -= 1
                if count[q] == 0: pop[q]["rank"] = i + 1; nxt.append(q)
        if nxt: fronts.append(nxt)
        i += 1
    return fronts


def _environmental_select(pop, size):
    fronts = _fronts(pop)
    selected = []
    for front in fronts:
        _crowding(pop, front)
        ranked = sorted((pop[i] for i in front), key=lambda item: -item["crowding"])
        selected.extend(ranked[:size - len(selected)])
        if len(selected) >= size:
            break
    return selected


def _crowding(pop, front):
    for i in front: pop[i]["crowding"] = 0.0
    if len(front) < 3:
        for i in front: pop[i]["crowding"] = float("inf")
        return
    dims = len(pop[front[0]]["objectives"])
    for d in range(dims):
        ordered = sorted(front, key=lambda i: pop[i]["objectives"][d])
        pop[ordered[0]]["crowding"] = pop[ordered[-1]]["crowding"] = float("inf")
        lo, hi = pop[ordered[0]]["objectives"][d], pop[ordered[-1]]["objectives"][d]
        if hi == lo: continue
        for k in range(1, len(ordered) - 1):
            if pop[ordered[k]]["crowding"] != float("inf"):
                pop[ordered[k]]["crowding"] += (pop[ordered[k + 1]]["objectives"][d] - pop[ordered[k - 1]]["objectives"][d]) / (hi - lo)


_OBJECTIVE_INDEX = {"energy": 0, "cost": 1, "makespan": 2, "waiting": 3, "priority": 4}


def _representative_key(item, objective):
    index = _OBJECTIVE_INDEX.get(objective, _OBJECTIVE_INDEX["cost"])
    values = item.get("objectives", ())
    value = values[index] if isinstance(values, (list, tuple)) and len(values) > index else float("inf")
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        value = float("inf")
    return value


def _dedupe_objectives(front):
    unique = {}
    for item in front:
        vector = tuple(item.get("objectives", ()))
        unique.setdefault(vector, item)
    return list(unique.values())


def _select_representative(front, objective):
    if not front:
        raise ValueError("cannot select a representative from an empty Pareto front")
    max_scheduled = max(item.get("scheduled", 0) for item in front)
    eligible = [item for item in front if item.get("scheduled", 0) == max_scheduled]
    return min(eligible, key=lambda item: _representative_key(item, objective))


def run_nsga2(snapshot, objective="balanced", options=None):
    options = options or {}; rng = random.Random(int(options.get("seed", 7)))
    jobs, server_count = list(snapshot["jobs"]), max(1, len(snapshot["servers"]))
    n = len(jobs); size = max(6, min(100, int(options.get("population_size", 30))))
    generations = max(1, min(100, int(options.get("generations", 25))))
    time_limit = max(.05, min(10.0, float(options.get("time_limit_seconds", 2.0))))
    started = time.perf_counter()
    deadline = started + time_limit
    evaluations = 0
    completed_generations = 0
    limit_reached = False
    def gene():
        return (rng.sample(jobs, n), {job["id"]: rng.randrange(server_count) for job in jobs})
    def evaluate(g):
        r = schedule_gpu_aware(snapshot, order=g[0], assignments=g[1],
                               planning_horizon_minutes=options.get("planning_horizon_minutes"))
        r["m"] = compute_metrics(r["sch"], r["uns"], snapshot["servers"], r["power"])
        return {"gene": g, "result": r, "objectives": _objectives(r),
                "scheduled": r["m"]["scheduled"]}
    def breed(a, b):
        cut = rng.randrange(n + 1) if n else 0
        order = a[0][:cut] + [j for j in b[0] if j not in a[0][:cut]]
        assignment = {job["id"]: (a[1][job["id"]] if rng.random() < .5
                                   else b[1][job["id"]]) for job in jobs}
        if n and rng.random() < .35:
            i, j = rng.randrange(n), rng.randrange(n); order[i], order[j] = order[j], order[i]
        if n and rng.random() < .35:
            assignment[rng.choice(jobs)["id"]] = rng.randrange(server_count)
        return order, assignment
    pop = []
    while len(pop) < size:
        if evaluations and time.perf_counter() >= deadline:
            limit_reached = True
            break
        pop.append(evaluate(gene()))
        evaluations += 1
    for _ in range(generations):
        if len(pop) < 2 or time.perf_counter() >= deadline:
            limit_reached = True
            break
        fronts = _fronts(pop)
        for front in fronts: _crowding(pop, front)
        def tournament():
            a, b = rng.sample(pop, 2)
            return a if (a["rank"], -a["crowding"]) < (b["rank"], -b["crowding"]) else b
        children = []
        while len(children) < size:
            if time.perf_counter() >= deadline:
                limit_reached = True
                break
            children.append(evaluate(breed(tournament()["gene"], tournament()["gene"])))
            evaluations += 1
        pop = _environmental_select(pop + children, size)
        if len(children) < size:
            break
        completed_generations += 1
        if time.perf_counter() >= deadline:
            limit_reached = True
            break
    fronts = _fronts(pop); pareto = _dedupe_objectives([pop[i] for i in fronts[0]])
    pareto.sort(key=lambda x: (x["objectives"][0], x["objectives"][2], x["objectives"][1]))
    # `_dominates` keeps the front at the maximum feasible job count; retain an
    # explicit guard here so display-solution selection cannot trade jobs away.
    selected = _select_representative(pareto, objective)
    result = selected["result"]
    result["status"] = ("NSGA-II time limit reached; valid Pareto front returned"
                         if limit_reached else
                         f"NSGA-II requested generations completed; selected display solution minimizes {objective if objective in _OBJECTIVE_INDEX else 'cost'}")
    result["seed"] = int(options.get("seed", 7))
    result["generations"] = completed_generations
    result["requested_generations"] = generations
    result["evaluations"] = evaluations
    result["execution_limit_reached"] = limit_reached
    result["pareto"] = [{"objectives": list(p["objectives"]), "m": p["result"]["m"],
                         "sch": p["result"]["sch"], "uns": p["result"]["uns"]} for p in pareto[:40]]
    return result
