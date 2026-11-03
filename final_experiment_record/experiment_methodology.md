# Experiment methodology

## Scope and provenance

The final comparison workload is the six-job fixture in `tests/test_gpu_simulation.py::test_all_algorithms_schedule_feasible_six_job_demo_snapshot`. Three GPU server objects use the `server()` test helper with per-GPU overrides. The exact input values and normalized effective defaults are archived in `workload_configuration.json` and `gpu_configuration.json`.

## Comparison run

All eight algorithms were run individually through the shared `run_algorithm()` interface on the same snapshot, in the order listed in `final_results.json`. Objective was `balanced`; options were seed 7, population 12, generations 4, execution limit 1 second, planning horizon 120 minutes. The options object was passed to each algorithm; `algorithm_parameters.json` distinguishes settings consumed by each solver from ignored settings. GA and NSGA-II each completed four generations without hitting their time limits (48 and 60 evaluations respectively).

## Seed experiments

GA and NSGA-II were each run 30 times on the same workload with seeds 0 through 29, objective `balanced`, population 12, four generations, one-second limit and 120-minute planning horizon. All runs completed the requested generations without reaching their execution limits. `random_seeds.json` lists every seed. `final_results.json` gives means and sample standard deviations (n-1 denominator) for scheduling count, energy, operating cost, makespan, wait, utilization, and execution time.

## Metrics and checks

For the one-run comparison, the record includes scheduled/unscheduled counts, scheduled IDs, energy, total operating cost, makespan, average waiting, both utilization fields, and elapsed algorithm time. `gpu_utilization_wall_clock` is the simulator's bounded wall-clock active interval metric. `gpu_busy_job_time_utilization` is the separate busy-job-time metric and can be higher when jobs share a GPU.

The comparison was checked for unique scheduled job IDs, starts no earlier than arrivals, waiting equal to start minus arrival, nonnegative duration, and aggregate per-GPU VRAM, compute, and memory-bandwidth budgets over overlapping intervals. Energy was checked against the power model's active, idle, low-power, startup and shutdown Wh totals. Operating cost was independently decomposed into rental from powered timeline duration, electricity from timeline energy/tariffs, and transition fees. These checks matched the reported simulator metrics for all eight methods. Existing test-suite checks cover sharing support and additional constraints.

## API tests

The HTTP smoke requests were sent to the running Express app at `http://localhost:3100/api/run`. All eight algorithm identifiers were tested. Invalid requests were followed by valid requests to ensure the server remained responsive. The request fixture and response summaries are documented in `api_smoke_test_results.md`.

## Limitations

Execution times are single-host measurements and vary with system load; 30-seed timing aggregates are more informative than the one-run comparison but are not portable performance guarantees. The six-job workload is a small deterministic unit-test fixture, not a representative production trace. No large-scale workload or physical GPU measurement was part of this final verification.
