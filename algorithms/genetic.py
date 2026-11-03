"""Seeded genetic search over workload order and per-job GPU preferences."""

import random
import time

from simulation import schedule_gpu_aware


def _evaluate(snapshot, order, assignment, objective, options):
    result = schedule_gpu_aware(snapshot, order=order, objective=objective,
                                assignments=assignment, allow_sharing=True,
                                planning_horizon_minutes=options.get("planning_horizon_minutes"))
    # All terms are real simulator metrics. Penalties scale with workload size
    # so infeasible solutions cannot beat feasible ones on a small cost value.
    m = result["m"] if "m" in result else None
    return result


def _fitness(result, objective, weights, job_count):
    m = result["m"]
    energy, cost, span, wait, priority_wait = (m["totalEnergyKWh"], m["totalOperatingCost"], m["makespan"],
                                     m["avgWaitingTime"], m["priorityWeightedWait"])
    if objective == "energy": base = energy
    elif objective == "makespan": base = span
    elif objective == "waiting": base = wait
    elif objective == "priority": base = priority_wait
    elif objective == "cost": base = cost
    else:
        base = (weights["energy"] * energy + weights["cost"] * cost +
                weights["makespan"] * span + weights["waiting"] * wait + weights["priority"] * priority_wait)
    return base + (job_count - m["scheduled"]) * max(1.0, abs(base)) * 1000


def run_genetic(snapshot, objective="balanced", options=None):
    options = options or {}
    rng = random.Random(int(options.get("seed", 7)))
    jobs, servers = list(snapshot["jobs"]), list(snapshot["servers"])
    n, g = len(jobs), max(1, len(servers))
    pop_size = max(4, min(100, int(options.get("population_size", 24))))
    generations = max(1, min(100, int(options.get("generations", 30))))
    time_limit = max(0.05, min(10.0, float(options.get("time_limit_seconds", 2.0))))
    started = time.perf_counter()
    deadline = started + time_limit
    evaluations = 0
    completed_generations = 0
    limit_reached = False
    weights = {"energy": 1.0, "cost": 1.0, "makespan": 0.5, "waiting": 0.5, "priority": 1.0}
    weights.update(options.get("weights", {}))
    def random_gene():
        # Store each preference against its job, not its current chromosome slot.
        return (rng.sample(jobs, n), {job["id"]: rng.randrange(g) for job in jobs})
    def evaluate(gene):
        result = schedule_gpu_aware(snapshot, order=gene[0], assignments=gene[1],
                                    planning_horizon_minutes=options.get("planning_horizon_minutes"))
        from metrics import compute_metrics
        result["m"] = compute_metrics(result["sch"], result["uns"], snapshot["servers"], result["power"])
        return _fitness(result, objective, weights, n), result
    population = [random_gene() for _ in range(pop_size)]
    best = None
    for _ in range(generations):
        scored = []
        for gene in population:
            if evaluations and time.perf_counter() >= deadline:
                limit_reached = True
                break
            outcome = evaluate(gene)
            evaluations += 1
            scored.append((outcome, gene))
            if best is None or outcome[0] < best[0]:
                best = outcome
        if not scored:
            break
        scored.sort(key=lambda x: x[0][0])
        if scored[0][0][0] < best[0]: best = scored[0][0]
        if len(scored) < pop_size:
            limit_reached = True
            break
        completed_generations += 1
        if time.perf_counter() >= deadline:
            limit_reached = True
            break
        next_pop = [scored[0][1], scored[1][1]]  # elitism
        while len(next_pop) < pop_size:
            if time.perf_counter() >= deadline:
                limit_reached = True
                break
            contestants = rng.sample(scored, min(3, len(scored)))
            parent_a = min(contestants, key=lambda x: x[0][0])[1]
            contestants = rng.sample(scored, min(3, len(scored)))
            parent_b = min(contestants, key=lambda x: x[0][0])[1]
            cut = rng.randrange(n + 1) if n else 0
            order = parent_a[0][:cut] + [j for j in parent_b[0] if j not in parent_a[0][:cut]]
            assign = {job["id"]: (parent_a[1][job["id"]] if rng.random() < .5
                                   else parent_b[1][job["id"]]) for job in jobs}
            if n and rng.random() < .35:
                a, b = rng.randrange(n), rng.randrange(n); order[a], order[b] = order[b], order[a]
            if n and rng.random() < .35:
                assign[rng.choice(jobs)["id"]] = rng.randrange(g)
            next_pop.append((order, assign))
        population = next_pop
        if len(population) < pop_size:
            break
    result = best[1]
    result.update({"status": ("Genetic time limit reached; best evaluated candidate returned"
                               if limit_reached else "Genetic search completed"),
                   "seed": int(options.get("seed", 7)), "generations": completed_generations,
                   "requested_generations": generations, "evaluations": evaluations,
                   "execution_limit_reached": limit_reached, "fitness": best[0]})
    return result
