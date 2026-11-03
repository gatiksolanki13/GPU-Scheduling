import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "algorithms"))

from energy_scheduler import energy_aware, resource_aware
from genetic import run_genetic
from gpu_model import execution_estimate, normalize_job, normalize_server, resource_compatible
from metrics import compute_metrics
from nsga2 import (_dedupe_objectives, _dominates, _environmental_select,
                   _fronts, _select_representative, run_nsga2)
from power_model import break_even_idle_minutes, gpu_ready_time, simulate_power
from schedulers import ALGORITHMS, exact_search, run_algorithm
from simulation import schedule_gpu_aware


def server(**overrides):
    value = {"id": 1, "name": "GPU-A", "vram": 16, "cost": 100, "on": True,
             "compute_capacity": 1.0, "memory_bandwidth": 900, "pcie_bandwidth": 24,
             "active_power_w": 250, "idle_power_w": 80, "low_power_w": 20,
             "startup_energy_wh": 18, "shutdown_energy_wh": 3,
             "startup_delay_s": 60, "shutdown_delay_s": 12,
             "initial_state": "IDLE", "supports_sharing": True}
    value.update(overrides)
    return value


def job(i=1, **overrides):
    value = {"id": i, "name": f"Job-{i}", "vram": 4, "rt": 10,
             "pri": "Medium", "arr": 0, "compute_demand": .3,
             "memory_bandwidth_demand": 100,
             "iterations": 1, "sharing": True}
    value.update(overrides)
    return value


def snap(jobs, servers=None):
    return {"jobs": jobs, "servers": servers or [server()]}


class GpuSimulationTests(unittest.TestCase):
    def test_job_id_preferences_survive_reordering_and_fall_back_when_infeasible(self):
        small = server(id=1, name="GPU-small", vram=8)
        large = server(id=2, name="GPU-large", vram=16)
        task_a = job(1, name="A", vram=12)
        task_b = job(2, name="B", vram=4)
        # A prefers the small GPU (index 0), where it cannot fit; B prefers it
        # too. The execution order is reversed from the source job list.
        result = schedule_gpu_aware(snap([task_a, task_b], [small, large]),
                                    order=[task_b, task_a], assignments={1: 0, 2: 0})
        placed = {row["jid"]: row["sid"] for row in result["sch"]}
        self.assertEqual(placed, {1: 2, 2: 1})
        self.assertEqual(result["uns"], [])

    def test_nsga_dominance_prioritizes_scheduled_job_count(self):
        more_jobs = {"scheduled": 6, "objectives": [100, 100, 100, 100, 100]}
        fewer_jobs = {"scheduled": 3, "objectives": [0, 0, 0, 0, 0]}
        tradeoff = {"scheduled": 6, "objectives": [0, 200, 20, 20, 10]}
        population = [more_jobs, fewer_jobs, tradeoff]
        fronts = _fronts(population)
        self.assertEqual({population[i]["scheduled"] for i in fronts[0]}, {6})
        survivors = _environmental_select(population, 1)
        self.assertEqual(survivors[0]["scheduled"], 6)
        self.assertTrue(_dominates(more_jobs, fewer_jobs))
        self.assertFalse(_dominates(fewer_jobs, more_jobs))

    def test_nsga_representative_respects_selected_objective_and_job_count(self):
        front = [
            {"scheduled": 6, "objectives": [1, 9, 8, 7, 6]},
            {"scheduled": 6, "objectives": [2, 3, 4, 5, 6]},
            {"scheduled": 6, "objectives": [3, 4, 5, 2, 6]},
            {"scheduled": 5, "objectives": [0, 0, 0, 0, 0]},
        ]
        expected = {"energy": 0, "cost": 1, "makespan": 1, "waiting": 2}
        for objective, index in expected.items():
            self.assertEqual(_select_representative(front, objective), front[index], objective)
        self.assertEqual(len(_dedupe_objectives([front[0], dict(front[0]), front[1]])), 2)
        self.assertEqual(_select_representative([{"scheduled": 1, "objectives": None}], "energy")["scheduled"], 1)
        self.assertEqual(_select_representative(front, "unknown"), front[1])

    def test_startup_readiness_is_consistent_across_all_algorithms(self):
        node = server(initial_state="OFF", min_off_minutes=3, startup_delay_s=90)
        for arrival, expected in [(2, 4.5), (4, 4.5), (5, 5)]:
            task = job(arr=arrival, compute_demand=.2, memory_bandwidth_demand=100)
            self.assertEqual(gpu_ready_time(node, arrival), expected)
            for key in ALGORITHMS:
                result = run_algorithm(key, snap([task], [node]), "balanced", {
                    "seed": 5, "population_size": 6, "generations": 1,
                    "time_limit_seconds": .1, "planning_horizon_minutes": 30,
                })
                self.assertEqual(len(result["sch"]), 1, (key, arrival))
                self.assertGreaterEqual(result["sch"][0]["start"], expected - 1e-6, (key, arrival))

    def test_ga_and_nsga_respect_initial_off_readiness_after_reordering(self):
        node = server(initial_state="OFF", min_off_minutes=3, startup_delay_s=90)
        tasks = [job(i, arr=(i - 1) * 2, compute_demand=.2, memory_bandwidth_demand=100)
                 for i in range(1, 5)]
        data = snap(tasks, [node])
        ready_at = node["min_off_minutes"] + node["startup_delay_s"] / 60
        for key in ("genetic", "nsga2"):
            for seed in range(3):
                result = run_algorithm(key, data, "balanced", {"seed": seed,
                    "population_size": 8, "generations": 2, "time_limit_seconds": .5})
                self.assertEqual(result["m"]["scheduled"], len(tasks), (key, seed))
                self.assertGreaterEqual(min(row["start"] for row in result["sch"]), ready_at, (key, seed))
                startup = next((seg for seg in result["power"]["nodes"][str(node["id"])]["timeline"]
                                if seg["state"] == "STARTING"), None)
                self.assertIsNotNone(startup, (key, seed))
                self.assertGreaterEqual(startup["start"], node["min_off_minutes"])

    def test_contention_updates_all_concurrent_job_runtimes(self):
        tasks = [job(1, rt=20, compute_demand=.35, memory_bandwidth_demand=100),
                 job(2, rt=20, compute_demand=.35, memory_bandwidth_demand=100)]
        node = server(compute_capacity=1, memory_bandwidth=900)
        result = schedule_gpu_aware(snap(tasks, [node]))
        self.assertEqual({row["start"] for row in result["sch"]}, {0})
        for row in result["sch"]:
            own = next(task for task in tasks if task["id"] == row["jid"])
            peer = next(task for task in tasks if task["id"] != row["jid"])
            expected, _ = execution_estimate(own, node, [peer])
            self.assertAlmostEqual(row["predicted_runtime"], expected, places=3)
        self.assertAlmostEqual(result["sch"][0]["fin"], result["sch"][1]["fin"], places=3)

    def test_wall_clock_gpu_utilization_unions_overlapping_intervals(self):
        tasks = [job(1, rt=20, compute_demand=.35, memory_bandwidth_demand=100),
                 job(2, rt=20, compute_demand=.35, memory_bandwidth_demand=100)]
        data = snap(tasks, [server(compute_capacity=1, memory_bandwidth=900)])
        result = run_algorithm("gpu_aware", data, "balanced")
        self.assertAlmostEqual(result["m"]["util"], 1.0)
        self.assertAlmostEqual(result["m"]["busyJobTimeUtil"], 2.0, places=3)

    def test_metaheuristics_return_partial_results_when_limit_expires(self):
        data = snap([job()])
        with patch("genetic.time.perf_counter", side_effect=[0.0, 1.0]):
            ga = run_genetic(data, options={"population_size": 4, "generations": 10,
                                             "time_limit_seconds": .05})
        self.assertEqual(ga["evaluations"], 1)
        self.assertEqual(ga["generations"], 0)
        self.assertTrue(ga["execution_limit_reached"])
        with patch("nsga2.time.perf_counter", side_effect=[0.0, 1.0]):
            nsga = run_nsga2(data, options={"population_size": 6, "generations": 10,
                                             "time_limit_seconds": .05})
        self.assertEqual(nsga["evaluations"], 1)
        self.assertEqual(nsga["generations"], 0)
        self.assertTrue(nsga["execution_limit_reached"])

    def test_exact_search_reports_exhaustive_and_early_stop_accurately(self):
        tasks = [job(1), job(2, arr=2)]
        nodes = [server(id=1), server(id=2, name="GPU-B", compute_capacity=1.4)]
        exhaustive = exact_search(snap(tasks, nodes), "balanced", {"time_limit_seconds": 2})
        self.assertEqual(exhaustive["nodes"], 4)
        self.assertIn("exhaustive optimum", exhaustive["status"])
        with patch("schedulers.time.perf_counter", side_effect=[0.0] + [1.0] * 100):
            partial = exact_search(snap(tasks, nodes), "balanced", {"time_limit_seconds": .05})
        self.assertEqual(partial["nodes"], 1)
        self.assertIn("time limit", partial["status"])
        self.assertTrue(partial["sch"])

    def test_metaheuristic_unscheduled_reason_names_resource_blocker(self):
        impossible = job(7, name="Impossible", compute_demand=2.0,
                         memory_bandwidth_demand=2000)
        nodes = [server(id=1, name="GPU-A", compute_capacity=1, memory_bandwidth=900),
                 server(id=2, name="GPU-B", compute_capacity=.5, memory_bandwidth=400)]
        for key in ("genetic", "nsga2"):
            result = run_algorithm(key, snap([impossible], nodes), "balanced",
                                   {"seed": 7, "population_size": 8,
                                    "generations": 2, "time_limit_seconds": 1})
            self.assertEqual(result["m"]["scheduled"], 0, key)
            self.assertIn("compute demand", result["uns"][0]["reason"], key)
            self.assertIn("GPU-A", result["uns"][0]["reason"], key)

    def test_all_algorithms_schedule_feasible_six_job_demo_snapshot(self):
        nodes = [server(id=1, name="GPU-01", vram=16, cost=120, compute_capacity=1,
                        memory_bandwidth=900),
                 server(id=2, name="GPU-02", vram=24, cost=180, compute_capacity=1.5,
                        memory_bandwidth=1200),
                 server(id=3, name="GPU-03", vram=8, cost=60, compute_capacity=.55,
                        memory_bandwidth=420)]
        tasks = [job(1, name="Model Training", vram=8, rt=30, pri="High", arr=0,
                     job_type="Training", compute_demand=.55, memory_bandwidth_demand=310),
                 job(2, name="Vision Inference", vram=4, rt=10, pri="High", arr=0,
                     job_type="Inference", compute_demand=.32, memory_bandwidth_demand=90),
                 job(3, name="Batch Embeddings", vram=12, rt=25, pri="Medium", arr=0,
                     job_type="Batch Processing", compute_demand=.42, memory_bandwidth_demand=260),
                 job(4, name="Simulation Run", vram=8, rt=40, pri="Low", arr=5,
                     job_type="Compute Intensive", compute_demand=.7, memory_bandwidth_demand=120),
                 job(5, name="Fine-tuning", vram=16, rt=45, pri="High", arr=0,
                     job_type="Memory Intensive", compute_demand=.48, memory_bandwidth_demand=520),
                 job(6, name="Nightly Export", vram=6, rt=18, pri="Low", arr=12,
                     job_type="Batch Processing", compute_demand=.25, memory_bandwidth_demand=180)]
        data = snap(tasks, nodes)
        options = {"seed": 7, "population_size": 12, "generations": 4,
                   "time_limit_seconds": 1, "planning_horizon_minutes": 120}
        for key in ALGORITHMS:
            result = run_algorithm(key, data, "balanced", options)
            self.assertEqual(result["m"]["scheduled"], 6, key)
            self.assertEqual(len({row["jid"] for row in result["sch"]}), 6, key)
            for row in result["sch"]:
                self.assertGreaterEqual(row["start"], tasks[row["jid"] - 1]["arr"], key)
                self.assertAlmostEqual(row["wait"], row["start"] - tasks[row["jid"] - 1]["arr"], places=3)
            for node in nodes:
                rows = [row for row in result["sch"] if row["sid"] == node["id"]]
                events = sorted({t for row in rows for t in (row["start"], row["fin"])})
                for left, right in zip(events, events[1:]):
                    t = (left + right) / 2
                    running = [row for row in rows if row["start"] <= t < row["fin"]]
                    self.assertLessEqual(sum(row["compute_demand"] for row in running), node["compute_capacity"], key)
                    self.assertLessEqual(sum(row["memory_bandwidth_demand"] for row in running), node["memory_bandwidth"], key)
                    self.assertLessEqual(sum(row["vram"] for row in running), node["vram"], key)
                    if len(running) > 1:
                        self.assertTrue(node["supports_sharing"], key)
                        self.assertTrue(all(tasks[row["jid"] - 1]["sharing"] for row in running), key)

    def test_all_algorithms_keep_aggregate_resources_within_capacity(self):
        tasks = [job(1, vram=4, rt=20, compute_demand=.65, memory_bandwidth_demand=300),
                 job(2, vram=4, rt=20, compute_demand=.25, memory_bandwidth_demand=100),
                 job(3, vram=4, rt=20, compute_demand=.55, memory_bandwidth_demand=250)]
        node = server(compute_capacity=1.4, memory_bandwidth=800, vram=16)
        data = snap(tasks, [node])
        options = {"seed": 17, "population_size": 10, "generations": 4,
                   "time_limit_seconds": 1, "planning_horizon_minutes": 120}
        for key in ALGORITHMS:
            result = run_algorithm(key, data, "balanced", options)
            schedule = result["sch"]
            events = sorted({t for row in schedule for t in (row["start"], row["fin"])})
            for left, right in zip(events, events[1:]):
                t = (left + right) / 2
                running = [row for row in schedule if row["start"] <= t < row["fin"]]
                self.assertLessEqual(sum(row["compute_demand"] for row in running), node["compute_capacity"], key)
                self.assertLessEqual(sum(row["memory_bandwidth_demand"] for row in running), node["memory_bandwidth"], key)
                self.assertLessEqual(sum(row["vram"] for row in running), node["vram"], key)

    def test_electricity_cost_does_not_double_count_transition_segment_draw(self):
        node = server(initial_state="OFF", electricity_tariff=20)
        data = snap([job()], [node])
        result = run_algorithm("gpu_aware", data, "balanced")
        timeline = result["power"]["nodes"]["1"]["timeline"]
        startup_wh = result["power"]["nodes"]["1"]["startup_wh"]
        shutdown_wh = result["power"]["nodes"]["1"]["shutdown_wh"]
        startup_segment = sum(x["energy_wh"] for x in timeline if x["state"] == "STARTING")
        shutdown_segment = sum(x["energy_wh"] for x in timeline if x["state"] == "SHUTTING_DOWN")
        expected_wh = sum(x["energy_wh"] for x in timeline)
        expected_wh += startup_wh - startup_segment + shutdown_wh - shutdown_segment
        self.assertAlmostEqual(result["m"]["electricityCost"], expected_wh / 1000 * node["electricity_tariff"])

    def test_prioritize_objective_is_accepted_by_api(self):
        server_js = (Path(__file__).resolve().parents[1] / "server.js").read_text()
        self.assertIn('["cost", "energy", "makespan", "waiting", "priority", "balanced"]', server_js)

    def test_single_job_on_idle_gpu_has_runtime_and_energy(self):
        result = schedule_gpu_aware(snap([job()]))
        self.assertEqual(len(result["sch"]), 1)
        metrics = compute_metrics(result["sch"], result["uns"], snap([job()])["servers"], result["power"])
        self.assertGreater(result["sch"][0]["predicted_runtime"], 0)
        self.assertGreater(metrics["activeEnergyWh"], 0)
        self.assertGreaterEqual(metrics["totalEnergyWh"], 0)

    def test_off_gpu_startup_delay_and_energy_are_included(self):
        node = server(initial_state="OFF", min_off_minutes=3)
        result = schedule_gpu_aware(snap([job()], [node]))
        slot = result["sch"][0]
        self.assertGreaterEqual(slot["start"], node["min_off_minutes"] + node["startup_delay_s"] / 60)
        expected_startup_wh = node["startup_energy_wh"] + node["idle_power_w"] * node["startup_delay_s"] / 3600
        self.assertAlmostEqual(result["power"]["totals"]["startup_wh"], expected_startup_wh)
        self.assertEqual(result["power"]["totals"]["transition_delay_minutes"], node["startup_delay_s"] / 60)
        self.assertIn("STARTING", [x["state"] for x in result["power"]["nodes"]["1"]["timeline"]])

    def test_power_telemetry_rejects_colliding_mixed_type_gpu_ids(self):
        mixed = [server(id=1), server(id="1", name="GPU-B")]
        with self.assertRaisesRegex(ValueError, "one consistent type"):
            simulate_power(mixed, [])
        duplicate = [server(id="gpu"), server(id="gpu", name="GPU-B")]
        with self.assertRaisesRegex(ValueError, "unique for telemetry"):
            simulate_power(duplicate, [])

    def test_string_ids_remain_stable_through_all_python_algorithms_and_telemetry(self):
        node = server(id="gpu-a")
        task = job("job-a")
        for key in ALGORITHMS:
            result = run_algorithm(key, snap([task], [node]), "balanced", {
                "seed": 2, "population_size": 6, "generations": 1,
                "time_limit_seconds": .1,
            })
            self.assertEqual(result["sch"][0]["jid"], "job-a", key)
            self.assertEqual(result["sch"][0]["sid"], "gpu-a", key)
            self.assertIn("gpu-a", result["power"]["nodes"], key)

    def test_short_idle_keeps_node_out_of_shutdown(self):
        power = simulate_power([server()], [], horizon_minutes=20)
        self.assertEqual(power["totals"]["shutdown_wh"], 0)
        self.assertNotIn("SHUTTING_DOWN", [x["state"] for x in power["nodes"]["1"]["timeline"]])

    def test_long_idle_shutdown_beats_transition_cost(self):
        node = server()
        self.assertGreater(break_even_idle_minutes(node), 0)
        power = simulate_power([node], [], horizon_minutes=160)
        expected_shutdown_wh = node["shutdown_energy_wh"] + node["idle_power_w"] * node["shutdown_delay_s"] / 3600
        self.assertAlmostEqual(power["totals"]["shutdown_wh"], expected_shutdown_wh)
        self.assertIn("SHUTTING_DOWN", [x["state"] for x in power["nodes"]["1"]["timeline"]])
        shutting = next(x for x in power["nodes"]["1"]["timeline"] if x["state"] == "SHUTTING_DOWN")
        self.assertAlmostEqual(shutting["end"] - shutting["start"], node["shutdown_delay_s"] / 60)
        keep_on = simulate_power([node], [], horizon_minutes=160, allow_shutdown=False)
        self.assertLess(power["totals"]["total_wh"], keep_on["totals"]["total_wh"])

    def test_long_gap_between_jobs_powers_node_down_and_back_up(self):
        result = schedule_gpu_aware(snap([job(1), job(2, arr=200)]))
        states = [x["state"] for x in result["power"]["nodes"]["1"]["timeline"]]
        self.assertIn("SHUTTING_DOWN", states)
        self.assertIn("OFF", states)
        self.assertIn("STARTING", states)
        self.assertEqual(result["power"]["totals"]["transitions"], 2)

    def test_energy_cost_uses_electricity_tariff_separately(self):
        low = server(electricity_tariff=2)
        high = server(electricity_tariff=20)
        task = job()
        a = run_algorithm("gpu_aware", snap([task], [low]), "cost")["m"]
        b = run_algorithm("gpu_aware", snap([task], [high]), "cost")["m"]
        self.assertEqual(a["rentalCost"], b["rentalCost"])
        self.assertGreater(b["electricityCost"], a["electricityCost"])

    def test_legacy_transfer_fields_are_accepted_and_ignored(self):
        task = job(h2d_gb=500, d2h_gb=300, pcie_bandwidth_demand=1e6)
        slow_node = server(pcie_bandwidth=0.01, pcie_efficiency=0.01)
        fast_node = server(id=2, name="GPU-B", pcie_bandwidth=10000, pcie_efficiency=1)
        slow_runtime, _ = execution_estimate(task, slow_node)
        fast_runtime, _ = execution_estimate(task, fast_node)
        self.assertAlmostEqual(slow_runtime, fast_runtime)
        self.assertTrue(resource_compatible([task], slow_node))
        result = run_algorithm("gpu_aware", snap([task], [slow_node]), "balanced")
        self.assertNotIn("avgPcieUtil", result["m"])
        self.assertNotIn("pcie_util", result["sch"][0])

    def test_vram_incompatible_workload_is_never_scheduled(self):
        result = schedule_gpu_aware(snap([job(vram=17)]))
        self.assertEqual(result["sch"], [])
        self.assertEqual(len(result["uns"]), 1)

    def test_two_memory_bound_jobs_do_not_overcommit_bandwidth(self):
        jobs = [job(1, memory_bandwidth_demand=600), job(2, memory_bandwidth_demand=600)]
        result = schedule_gpu_aware(snap(jobs))
        self.assertEqual(len(result["sch"]), 2)
        self.assertFalse(result["sch"][0]["start"] < result["sch"][1]["fin"] and result["sch"][1]["start"] < result["sch"][0]["fin"])

    def test_complementary_compute_and_memory_jobs_can_share(self):
        jobs = [job(1, compute_demand=.65, memory_bandwidth_demand=80),
                job(2, compute_demand=.2, memory_bandwidth_demand=350)]
        result = schedule_gpu_aware(snap(jobs))
        self.assertEqual(len(result["sch"]), 2)
        self.assertLess(max(x["start"] for x in result["sch"]), min(x["fin"] for x in result["sch"]))

    def test_job_arriving_during_existing_execution_affects_runtime_estimate(self):
        early = job(1, rt=20, compute_demand=.2, memory_bandwidth_demand=100)
        later = job(2, rt=20, arr=5, compute_demand=.2, memory_bandwidth_demand=100)
        node = server()
        result = schedule_gpu_aware(snap([early, later], [node]), order=[later, early])
        early_slot = next(row for row in result["sch"] if row["jid"] == early["id"])
        later_slot = next(row for row in result["sch"] if row["jid"] == later["id"])
        standalone, _ = execution_estimate(early, node)
        self.assertLess(early_slot["start"], later_slot["start"])
        self.assertGreater(early_slot["predicted_runtime"], standalone)

    def test_compute_over_capacity_falls_back_to_exclusive(self):
        jobs = [job(1, compute_demand=.65), job(2, compute_demand=.55)]
        result = schedule_gpu_aware(snap(jobs))
        self.assertFalse(result["sch"][0]["start"] < result["sch"][1]["fin"] and result["sch"][1]["start"] < result["sch"][0]["fin"])

    def test_node_without_sharing_support_uses_exclusive_execution(self):
        tasks = [job(1, compute_demand=.2), job(2, compute_demand=.2)]
        result = schedule_gpu_aware(snap(tasks, [server(supports_sharing=False)]))
        self.assertFalse(result["sch"][0]["start"] < result["sch"][1]["fin"] and result["sch"][1]["start"] < result["sch"][0]["fin"])

    def test_different_nodes_predict_different_runtime(self):
        nodes = [server(id=1, compute_capacity=.5), server(id=2, name="GPU-B", compute_capacity=2)]
        task = job(compute_demand=.4)
        a = schedule_gpu_aware(snap([task], nodes), assignments=[0])["sch"][0]
        b = schedule_gpu_aware(snap([task], nodes), assignments=[1])["sch"][0]
        self.assertGreater(a["predicted_runtime"], b["predicted_runtime"])

    def test_priority_algorithm_schedules_higher_tier_first(self):
        tasks = [job(1, pri="Low", rt=2), job(2, pri="High", rt=2), job(3, pri="Medium", rt=2)]
        result = run_algorithm("prio", snap(tasks), "balanced")
        self.assertEqual([x["jid"] for x in sorted(result["sch"], key=lambda x: x["start"])], [2, 3, 1])
        self.assertGreater(result["m"]["priorityWeightedWait"], 0)
        self.assertNotIn("misses", result["m"])
        self.assertNotIn("lateMin", result["m"])

    def test_legacy_deadline_input_is_ignored(self):
        task = job(dl=-100)
        result = run_algorithm("gpu_aware", snap([task]), "balanced")
        self.assertEqual(len(result["sch"]), 1)
        self.assertNotIn("dl", result["sch"][0])
        self.assertNotIn("dl", result["snap"]["jobs"][0])

    def test_arrivals_are_respected(self):
        tasks = [job(1, arr=8), job(2, arr=20)]
        result = schedule_gpu_aware(snap(tasks))
        self.assertTrue(all(x["start"] >= tasks[x["jid"] - 1]["arr"] for x in result["sch"]))

    def test_empty_queue_returns_zero_metrics(self):
        result = run_algorithm("gpu_aware", snap([]), "balanced")
        self.assertEqual(result["m"]["scheduled"], 0)
        self.assertGreaterEqual(result["m"]["totalEnergyWh"], 0)
        self.assertEqual(result["m"]["scheduled"], 0)

    def test_nsga_front_is_non_dominated(self):
        task_list = [job(1, rt=8, compute_demand=.4), job(2, rt=15, compute_demand=.5)]
        result = run_nsga2(snap(task_list, [server(id=1), server(id=2, name="GPU-B", compute_capacity=1.5, cost=180)]),
                           options={"seed": 11, "population_size": 8, "generations": 3, "time_limit_seconds": 1})
        values = [p["objectives"] for p in result["pareto"]]
        self.assertTrue(values)
        for i, a in enumerate(values):
            for j, b in enumerate(values):
                if i != j:
                    self.assertFalse(all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b)))

    def test_genetic_algorithm_returns_valid_feasible_schedule(self):
        result = run_genetic(snap([job(1), job(2, arr=3)]), options={"seed": 3, "population_size": 8, "generations": 3, "time_limit_seconds": 1})
        self.assertEqual(len(result["sch"]) + len(result["uns"]), 2)
        self.assertTrue(all(x["vram"] <= 16 for x in result["sch"]))

    def test_all_algorithms_support_priority_objective_without_deadline_metrics(self):
        data = snap([job(1, pri="Low"), job(2, pri="High")])
        for key in ALGORITHMS:
            result = run_algorithm(key, data, "priority", {"seed": 2, "population_size": 6, "generations": 2, "time_limit_seconds": .3})
            self.assertIn("priorityWeightedWait", result["m"], key)
            self.assertNotIn("misses", result["m"], key)
            self.assertNotIn("lateMin", result["m"], key)
            self.assertTrue(all("dl" not in slot for slot in result["sch"]), key)

    def test_all_algorithms_respect_vram_feasibility(self):
        data = snap([job(vram=20)])
        for key in ALGORITHMS:
            result = run_algorithm(key, data, "balanced", {"seed": 2, "population_size": 6, "generations": 2, "time_limit_seconds": .3})
            self.assertEqual(result["sch"], [], key)
            self.assertEqual(len(result["uns"]), 1, key)

    def test_all_algorithms_respect_compute_and_memory_capacity(self):
        for key in ALGORITHMS:
            data = snap([job(compute_demand=1.1)], [server()])
            result = run_algorithm(key, data, "balanced", {"seed": 2, "population_size": 6, "generations": 2, "time_limit_seconds": .3})
            self.assertEqual(result["sch"], [], key)
            self.assertEqual(len(result["uns"]), 1, key)

    def test_baseline_legacy_data_gets_gpu_defaults(self):
        old = {"servers": [{"id": 1, "name": "old", "vram": 8, "cost": 20, "on": True}],
               "jobs": [{"id": 1, "name": "old job", "vram": 4, "rt": 5, "pri": "Low", "arr": 0}]}
        result = run_algorithm("fcfs", old, "balanced")
        self.assertEqual(result["m"]["scheduled"], 1)
        self.assertGreater(result["sch"][0]["predicted_runtime"], 0)


if __name__ == "__main__":
    unittest.main()
