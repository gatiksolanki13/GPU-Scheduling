# GPU Scheduler: Project and Backend Guide

This guide describes how the GPU Scheduler project is organized, what each backend function does, and how its API responses are used by the browser interface. It is written against the current implementation in this repository.

## 1. What the application does

The app models GPU jobs and a small pool of GPU servers. A job has a VRAM requirement, runtime, priority tier, and arrival time. A server has a VRAM capacity, hourly price, and enabled state. The user can edit this data in the browser and ask the backend to:

1. Schedule jobs with one selected method.
2. Run all scheduling methods on the same input and compare them.
3. Solve an educational transportation allocation using VAM and MODI.

The application is a simulation. It does not connect to real GPUs, launch workloads, store data in a database, or retain edits after the page is refreshed. Hardware defaults are illustrative values, not specifications for particular devices.

## 2. Project layout

| File | Responsibility |
| --- | --- |
| `server.js` | Express web server, static asset hosting, API validation, Python process launch, and HTTP error responses. |
| `algorithms/run.py` | Reads one JSON request from standard input, dispatches by action, and writes JSON to standard output. |
| `algorithms/schedulers.py` | Baseline schedulers and dispatch to GPU-aware methods. |
| `algorithms/metrics.py` | Objective weights, energy/cost/resource metrics, and schedule metrics. |
| `algorithms/transport.py` | Divisible-minute transportation model, VAM initial solution, and MODI improvement loop. |
| `algorithms/gpu_model.py` | Backward-compatible hardware/workload profiles and node-specific runtime estimates. |
| `algorithms/interference.py` | Resource compatibility scoring for simulated co-location. |
| `algorithms/power_model.py` | Node power-state timelines, transition energy, and idle break-even calculation. |
| `algorithms/simulation.py` | Deterministic placement, resource-fit checks, concurrency, and power simulation. |
| `algorithms/energy_scheduler.py` | Energy-aware consolidation and GPU resource-aware placement. |
| `algorithms/genetic.py` / `algorithms/nsga2.py` | Seeded genetic and multi-objective Pareto search. |
| `tests/test_gpu_simulation.py` | Unit tests for scheduling, power, contention, and metaheuristics. |
| `public/index.html` | Page shell, navigation, dialogs, workspace tabs, and script/style references. |
| `public/app.js` | Browser state, editing forms, API requests, and rendering of tables, charts, metrics, and timeline. |
| `public/style.css` | Visual styling and responsive layout. |

## 3. Runtime and request lifecycle

Run from the `gpu-scheduler` directory with `npm start`. Node serves the `public` directory and listens on `PORT`, defaulting to port 3000. Python is selected using `PYTHON`; otherwise it uses `python` on Windows and `python3` on other platforms.

For each API operation, the browser makes a JSON `POST` request to `/api/...`. `server.js` checks the request shape, starts a fresh Python process running `algorithms/run.py`, writes the request JSON to that process's stdin, and parses its stdout as JSON. Python errors are written to stderr. The server rejects nonzero exits and malformed JSON, and kills the solver process after 30 seconds. The response JSON is returned to the browser.

```text
Browser state (app.js)
  -> POST /api/run | /api/compare | /api/transport
  -> Express route validation (server.js)
  -> child process: python algorithms/run.py
  -> schedulers.py / transport.py (+ metrics.py)
  <- JSON result
  <- app.js renders results
```

There is no persistent backend state: each request contains a complete snapshot of jobs and servers. The solver works from that snapshot and exits.

## 4. Data model and API contract

The browser sends this shared snapshot shape:

```json
{
  "servers": [
    {"id": 1, "name": "GPU-01", "vram": 16, "cost": 120, "on": true}
  ],
  "jobs": [
    {"id": 1, "name": "Model Training", "vram": 8, "rt": 30,
     "pri": "High", "arr": 0}
  ]
}
```

Units are VRAM in GB, runtime/arrival in minutes, and server price in rupees per hour. Priority tiers are `High`, `Medium`, and `Low`, weighted 3, 2, and 1. Legacy `dl` input fields are ignored for backward compatibility.

| Endpoint | Additional request fields | Python action | Response |
| --- | --- | --- | --- |
| `POST /api/run` | One of eight `algo` keys; `objective`: `cost`, `energy`, `makespan`, `waiting`, `priority`, or `balanced`; optional metaheuristic `options` | Runs one baseline or GPU-aware optimizer. | Result object with schedule, power timeline, metrics, status, runtime, input snapshot, and optional Pareto front. |
| `POST /api/compare` | `objective` and optional `options` | Runs all eight schedulers sequentially over the same snapshot. | Array of result objects, in algorithm registry order. |
| `POST /api/transport` | `capacity`: positive minutes per enabled GPU | Runs the VAM + MODI transportation calculation. | Allocation matrix, cost matrix, row/column names, supplies/demands, total cost, and trace; `null` if there are no enabled servers or no jobs. |

Invalid snapshots and unsupported algorithms/objectives/capacities produce HTTP 400 JSON errors. Solver startup, timeout, or output failures produce HTTP 500 JSON errors.

## 5. Express backend (`server.js`)

### Configuration and middleware

- `PORT`, `PYTHON`, and `RUNNER` select the listener, interpreter, and Python entrypoint.
- `ALGORITHMS` and `OBJECTIVES` are allowlists used before invoking Python.
- `express.json({ limit: "1mb" })` parses request bodies and limits their size.
- `express.static(.../public)` serves the frontend files.

### `solve(payload)`

This is the Node-to-Python bridge. It creates a child process with `spawn(PYTHON, [RUNNER])`, buffers stdout and stderr, and sends the serialized payload to stdin. A timer kills the process after 30 seconds. On close, a nonzero exit rejects with the final stderr line (or a generic failure); a successful exit must contain valid JSON. The API route awaits this promise and sends the parsed value as its response.

### `validSnapshot(s)`

Checks that a snapshot exists, `servers` and `jobs` are arrays, server entries have a boolean `on`, positive `vram`, and nonnegative `cost`, and job entries have positive runtime/VRAM and a recognized priority. It is a boundary check, not a full schema validator: for example, it does not comprehensively validate numeric finiteness, IDs, names, arrival values, or priority tier and arrival values. The browser's add/edit forms perform some additional checks, but API callers can bypass those forms.

### `route(action, check)` and route-specific checks

`route` factors the three POST handlers into one wrapper. It validates `body.snapshot`, runs an action-specific validator, then adds the fixed `action` field and invokes `solve`.

- `/api/run` permits only a known algorithm and objective.
- `/api/compare` validates the objective; Python chooses all algorithms.
- `/api/transport` checks that capacity is greater than zero.

## 6. Python entrypoint (`algorithms/run.py`)

`main()` is intentionally a thin dispatcher:

1. Parse one JSON object from stdin.
2. Read `action`, `snapshot`, and optional `objective` (default `balanced`).
3. For `run`, call `run_algorithm` with the requested algorithm.
4. For `compare`, call `run_algorithm` once per key in `ALGORITHMS`.
5. For `transport`, call `solve_transport` with capacity defaulting to 60.
6. Serialize the result as JSON to stdout.

The Node server owns process errors and timeouts; the Python runner does not implement an HTTP server.

## 7. Whole-job scheduling (`algorithms/schedulers.py`)

### Shared scheduling helpers

`make_slot(job, server, start)` creates a schedule row. Finish is `start + rt`; cost is `rt / 60 * server.cost`; wait is `start - arr`. The row retains job/server IDs and names, and priority for metrics and display.

`unscheduled_entry(job)` records a job with the explanation that no enabled GPU has enough VRAM. This applies when there are no enabled compatible servers; it does not mean the solver ran out of time.

`list_schedule(order, pick_key, snap)` implements the common greedy scheduler. It tracks the next free time for each server, then for each job finds enabled servers with sufficient VRAM. A job starts at the later of its arrival and its server's free time. The supplied `pick_key` chooses a server option. The selected server's free time advances to the job finish. If there are no options, the job goes to the unscheduled list.

### Heuristic algorithms

- **FCFS (`fcfs`)** orders jobs by arrival time and then ID. It assigns each job to the compatible GPU with earliest finish; ties prefer lower execution cost, then lower server ID.
- **SJF (`sjf`)** orders by runtime and then ID, using the same earliest-finish GPU selection.
- **Priority (`priority_schedule`)** orders higher tiers first, then earlier arrivals, then ID. Higher-paying customer tiers map to higher priority weights: High 3, Medium 2, Low 1.

These algorithms are heuristics: their job ordering and greedy GPU selection are fixed and do not optimize the full schedule globally. The objective field does not alter these three heuristics' ordering or selection.

### Exact search (`exact_search`)

Despite the UI label, “Exact search” is exact only within this implementation's model and explored assignments. It first builds each job's compatible enabled-server list. Jobs with no compatible GPU are pre-marked unscheduled. It recursively enumerates a server assignment for each remaining job. For each complete assignment, jobs assigned to each GPU are ordered by higher priority first, then arrival time; the server schedule is built in that order, with each start respecting arrival and the preceding job's finish.

Each completed assignment is scored using `score()`. It adds a dominant penalty for unscheduled jobs, then weighted execution cost, energy, makespan, waiting, and priority-weighted waiting. The time limit is checked before descending into each remaining assignment; at least one complete candidate is evaluated, and a final completed assignment is reported as exhaustive even if the clock crosses the limit during its evaluation. The result distinguishes an exhaustive optimum over GPU assignments from an early stop. It is not an exhaustive search over every possible job sequence per GPU; the sequence is fixed by the ordering rule.

### `run_algorithm(key, snap, objective)`

Looks up the algorithm, measures its elapsed time, runs it, attaches the display name and default `heuristic` status, calculates metrics with `compute_metrics`, and includes the input snapshot in the result. `exact_search` can set its own status and node count.

## 8. Metrics and objective (`algorithms/metrics.py`)

`compute_metrics(schedule, unscheduled, servers)` derives values from schedule rows:

- **scheduled / unscheduled:** row counts.
- **makespan:** largest finish time, or zero for an empty schedule.
- **cost:** sum of row execution costs.
- **average wait:** mean of `start - arrival` for scheduled jobs, or zero.
- **wall-clock GPU active time:** union of scheduled intervals per enabled GPU divided by `enabled server count × makespan`, or zero when the denominator is zero.
- **busy-job time utilization:** sum of job durations divided by the same denominator. This may exceed 100% when jobs overlap on a shared GPU.

`objective_weights()` selects the scalar metric or default balanced coefficients:

| Objective | Optimized metric |
| --- | --- |
| `cost` | Total operating cost (electricity + powered-time rental + configured transition cost). |
| `energy` | Total simulated energy in kWh. |
| `makespan` | Maximum completion time. |
| `waiting` | Average job waiting time. |
| `priority` | Priority-weighted waiting time; higher-tier wait has a larger penalty. |
| `balanced` | Weighted combination; five weights, including priority-weighted wait, are configurable in Optimize. |

Exact search and the Genetic Algorithm use selected objective components; Energy-Aware and GPU Resource-Aware placement use the objective when ranking node candidates. NSGA-II retains five separate objectives regardless of the selected scalar objective. Unscheduled work remains an explicit penalty for scalar-search methods.

`js_round()` and `number_text()` support the transportation trace and align displayed rounding/text with JavaScript-style half-up rounding.

## 9. Transportation solver (`algorithms/transport.py`)

`solve_transport(snapshot, capacity)` treats each enabled GPU as a supply row with the same minute capacity and each job as a demand column equal to its runtime. Unlike the whole-job scheduler, this model can split a job's minutes over multiple GPUs; the UI explicitly labels it educational and not an executable schedule.

The cost of allocating one minute from a GPU to a job is `server.cost / 60`. If VRAM is insufficient, the pair is assigned `BIG_M = 10000` per minute. If total supply and demand differ, a dummy unused-capacity column or unmet-demand row balances the transportation table.

The solver then:

1. Uses Vogel's Approximation Method (VAM): compute the gap between the two smallest costs in each open row and column, select the largest penalty, and allocate as much as possible to its cheapest cell.
2. Completes a valid basis if degeneracy leaves fewer than `rows + columns - 1` basic cells.
3. Uses MODI potentials (`u`, `v`) to compute reduced costs for non-basic cells.
4. Enters a negative reduced-cost cell, finds a closed row/column cycle, shifts allocations by the limiting alternating quantity, and repeats (up to 50 iterations).
5. Returns the allocation and cost matrices, row/column labels, totals, and human-readable iteration trace.

The big-M value discourages incompatible GPU/job assignments but does not make the matrix a hard feasibility constraint. This model also ignores job indivisibility, arrivals, and the one-job-at-a-time execution timeline.

## 10. Frontend-to-backend mapping (`public/app.js`)

### Browser state and payload construction

`seed()` creates the initial in-memory sample jobs and GPUs. `S` is the live browser state; edits and additions mutate it. `snapshot()` makes a JSON deep copy so a request represents a stable input at the time the user clicks Run. `api(path, body)` sends JSON to `/api/${path}`, parses the response, and throws a readable error for network or HTTP failures.

### User actions to routes

| UI action / utility | Request | Backend path | Response consumer |
| --- | --- | --- | --- |
| Optimize tab `doRun()` | `{algo, objective, snapshot}` | `/api/run` → `run_algorithm()` → selected scheduler | `run` state; `optV()` renders status, metrics, analysis, timeline, schedule rows, and unscheduled jobs. |
| Compare tab `doCmp()` | `{objective, snapshot}` | `/api/compare` → all `ALGORITHMS` → each `run_algorithm()` | `cmp` array; `cmpV()` renders comparison table and `cmpAnal()` charts. |
| VAM + MODI `doTr()` | `{capacity, snapshot}` | `/api/transport` → `solve_transport()` | `trRes`; `trV()` renders allocation table and solver trace. |

All three actions pass the same snapshot structure. The selected objective is sent for compare as well, because each result's exact-search member uses it.

### Result fields consumed in the UI

For a whole-job result, `sch` is the list of slots and `uns` is the list of jobs with no compatible enabled GPU. `m` contains the metrics. `algo`, `status`, `ms`, and optional `nodes` supply the run heading/status line. `snap` is used by the timeline and to describe the comparison input.

- `mcards(m)` turns metrics into the KPI cards.
- `anal(r)` and `chart(...)` draw per-job cost and per-GPU utilization bars.
- `gantt(r)` groups schedule slots by enabled server and scales their start/finish times to the makespan.
- `optV()` formats schedule rows and highlights late jobs.
- `cmpAnal(c)` and `cmpV()` show each algorithm's cost, makespan, average wait, and other metrics.
- `trV()` consumes `x`, `rn`, `cn`, `sup`, `dem`, `tot`, and `tr` from the transportation result.

Other frontend utilities (`jobsV`, `srvV`, `addJob`, `addSrv`, `ed`, `del`, `openWS`, and `go`) manage user input and presentation; they do not call backend endpoints. `updC()` updates the visible queue/server count. The task modal and task table both write into the same `S.jobs` array. Data exists only in browser memory until reload/reset.

## 11. Reading the displayed numbers

- A server's price is hourly. Per-job GPU cost is proportional to predicted runtime, while reported rental cost uses the time the node remains powered.
- Start time is relative to the simulation's time zero; arrival may force a positive start even on an idle GPU.
- Wait is time from arrival until start.
- Makespan is the maximum finish timestamp, not the sum of runtimes.
- Legacy `util` is busy duration divided by enabled GPUs times makespan. New resource metrics use time-weighted powered node-minutes, excluding OFF periods.
- The transport allocation costs use cost per minute and may allocate portions of one job to several GPUs. Do not interpret it as a runnable schedule.

## 12. Implementation boundaries and caveats

- This is a single-page frontend with in-memory state, not a persistent multi-user service.
- A solver process is created per API call; comparison invokes all eight schedulers within one Python process/request.
- Baseline methods remain exclusive. GPU-aware methods permit concurrency only when simulated VRAM, compute, memory bandwidth, and sharing constraints fit.
- GPU-aware algorithms use the selected objective; baseline FCFS/SJF/priority order remains fixed for comparison, while baseline exact search uses the selected objective weights.
- Exact search explores server assignments but uses a fixed priority-and-arrival ordering within each GPU, so “optimal” is limited to that modeled search space.
- Transportation modeling deliberately allows divisible runtime and penalized incompatible pairs; it is separate from whole-job scheduling semantics.
- Backend validates supported objectives, hardware/workload numeric fields, power states, and sharing flags; values are still estimates and are not a guarantee of real-device feasibility.

## 13. GPU energy and resource simulation additions

The earlier sections describe the original project. This section documents the upgraded implementation. The original `fcfs`, `sjf`, `prio`, `exact`, and `transport` request shapes remain supported. New server and job fields are optional, so older snapshots receive simulated defaults through `gpu_model.normalize_server()` and `normalize_job()`. VAM + MODI remains separate and unchanged.

### New API options

`/api/run` and `/api/compare` accept the four baseline keys plus `energy`, `gpu_aware`, `genetic`, and `nsga2`. Objectives include `cost`, `energy`, `makespan`, `waiting`, `priority`, and `balanced`. Stochastic algorithms accept `options.seed`, `population_size`, `generations`, `time_limit_seconds`, and optional balanced weights.

- **Energy-Aware Consolidation** prefers already-used nodes, avoids unnecessary node starts, and uses `power_model.break_even_idle_minutes()` before shutdown. Shutdown is skipped when the expected low-power savings cannot recover transition energy during the planning horizon.
- **GPU Resource-Aware Placement** checks VRAM, normalized compute demand, memory bandwidth, sharing flags, and per-node runtime. Concurrent jobs must fit those resource budgets; otherwise a job waits for a resource release.
- **Genetic Algorithm** candidates encode job ordering and GPU preferences keyed by stable job ID. Seeded tournament selection, crossover, mutation, and elitism evaluate actual simulator metrics. Time checks occur between candidate evaluations; a single in-progress schedule evaluation may finish after the requested limit.
- **NSGA-II** separately optimizes energy, operating cost, makespan, wait, and priority-weighted waiting. A candidate scheduling more jobs dominates every candidate scheduling fewer jobs. Its `pareto` response contains non-dominated solutions at the best scheduled-job count found. Time checks occur between evaluations; a single in-progress schedule evaluation may finish after the requested limit.

### Hardware, jobs, and calculations

Server fields retain `id`, `name`, `vram`, `cost`, and `on`. Optional fields add normalized compute capacity, memory/PCIe bandwidth, active/idle/low-power watts, startup/shutdown energy, delay and cost, electricity tariff, power limit, temperature threshold, minimum on/off time, efficiency/overlap factors, sharing support, and initial power state. The GPU Servers tab exposes editable values and three explicitly illustrative hardware profiles.

Jobs retain `rt` as a user runtime estimate and add `job_type`, `compute_demand`, `memory_bandwidth_demand`, iterations, and a sharing flag. Predicted time estimates compute work divided by effective compute capacity and memory work divided by effective bandwidth. Compute and memory overlap according to a configurable factor. Legacy API payloads may still include PCIe demand or host/device transfer fields; these fields are ignored and do not affect scheduling or metrics. Workload classes, thermal/power limits, and co-location apply modest documented simulation factors.

Energy is accumulated from node-level timelines for OFF, STARTING, IDLE, ACTIVE, LOW_POWER, and SHUTTING_DOWN states. Wh totals include active, idle, low-power, startup, and shutdown energy; kWh is Wh divided by 1000. Electricity cost uses tariff × kWh, rental cost uses powered node time, and optional startup/shutdown fees remain separate. Total operating cost adds electricity, rental, and transition costs. Wall-clock active time uses the union of scheduled intervals; compute, VRAM, and memory allocation metrics are time-weighted over powered node-minutes, excluding OFF periods. When a new shared job joins an interval, the simulator recalculates all affected jobs' durations until their overlapping peer sets stabilize.

The optimization view shows separate metrics, power-state timelines, peak simulated VRAM, compute and memory-bandwidth allocation/utilization, and predicted versus user runtime. Simulation outputs are not physical measurements. Co-location is only a resource-budgeted simulation; it does not claim CUDA MPS or MIG isolation. The transportation result still permits divisible job minutes and is not an executable whole-job schedule.

### Run and test

From `gpu-scheduler/`, run `npm ci` and `npm start`. Set `PYTHON` to the intended Python interpreter if needed. Run Python unit tests with `python -m unittest discover -s tests -v` and API validation tests with `npm run test:api`; the optimizer uses only the Python standard library.
