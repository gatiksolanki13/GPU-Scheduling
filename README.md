# GPU Scheduler Simulation

A browser-based teaching project for scheduling GPU workloads across a small pool of simulated GPU servers. It compares eight scheduling methods and includes an educational VAM + MODI transportation solver.

> **This is a simulation.** It does not connect to GPUs or run workloads. Hardware values and outputs are estimates calculated from user-configured inputs; they are not measurements or guarantees about real devices.

## Run locally

Requirements: Node.js, npm, and Python 3. The scheduling code uses Python's standard library.

```bash
cd gpu-scheduler
npm start
```

Open [http://localhost:3000](http://localhost:3000). Set `PORT` to change the web port and `PYTHON` to select a Python interpreter. The browser keeps edits in memory; refreshing resets the current session to the demo data.

## What the inputs mean

### Workloads

| Input | Meaning and effect |
| --- | --- |
| Name | Display label for the workload. |
| Workload type | Selects modest simulated calibration factors. Compute Intensive increases compute work by 25%; Inference reduces it by 10%; Memory Intensive increases memory work by 25%; Batch Processing increases memory work by 10%. Training uses the unadjusted work. |
| Peak VRAM (GB) | Memory required by the job. A job cannot run on a node with less VRAM. Concurrent jobs' VRAM requirements are added and must fit. |
| Runtime estimate (minutes) | User's reference runtime. It helps define the modeled compute and memory work; GPU-aware methods can predict a different runtime for each node. |
| Priority | Customer tier: High = 3, Medium = 2, Low = 1. The Priority scheduler places higher tiers first. Priority-aware objectives and penalties give higher tiers more weight. |
| Arrival (minutes) | Earliest time the job may start, measured from simulation time zero. |
| Compute demand | Normalized share of the node's compute capacity required by the job. It is checked for feasibility and co-location. |
| Memory bandwidth demand (GB/s) | Estimated GPU-memory bandwidth needed by the job. It is checked against node and co-location limits. |
| Iterations | Multiplies the runtime estimate when the simulator derives compute work. |
| Allow sharing | Whether this job may participate in simulated concurrent placement. It does not enable CUDA MPS or MIG. |

The submission and editing forms do not have a deadline field. Older API snapshots may contain `dl`; the backend ignores it and omits it from returned job snapshots.

### GPU server settings

All hardware profiles are illustrative. The GPU Servers view exposes the following settings:

| Factor | Meaning and effect |
| --- | --- |
| Enabled (`on`) | Disabled nodes are not scheduling candidates and are counted as OFF in power output. |
| VRAM (GB) | Maximum job memory footprint, including the sum for co-located jobs. |
| Cost per hour (₹) | Simulated rental rate. A scheduled job's displayed runtime cost is predicted minutes / 60 × this rate. Total rental cost is based on how long the node remains powered. |
| Compute capacity | Normalized simulated compute throughput. Higher capacity can reduce compute-bound predicted runtime. |
| Memory bandwidth (GB/s) | Simulated GPU-memory throughput. Higher bandwidth can reduce memory-bound predicted runtime. |
| Active / idle / low-power watts | Power draw used for ACTIVE, IDLE, and LOW_POWER timeline segments. Active watts are capped by the configured power limit. |
| Power limit (W) | Caps active watts and can lower effective simulated compute throughput when below active power. |
| Compute / memory efficiency | Fractions from 0 to 1 that scale effective compute capacity and memory bandwidth. |
| Compute-memory overlap | Fraction from 0 to 1 describing how much compute and memory work overlap. Zero models the work as fully sequential; one models duration as the slower of the two components. |
| Temperature and thermal threshold | At or above the threshold, the model reduces effective compute and memory throughput by 18%. This is a simple fixed throttle factor, not a thermal simulation. |
| Electricity tariff (₹/kWh) | Converts simulated energy to electricity cost. |
| Initial power state | OFF, IDLE, LOW_POWER, or ACTIVE at simulation start. An OFF node may require startup delay and energy before its first job. |
| Startup / shutdown delay | Simulated transition time. Startup delay can delay the first job; transition delays are also reported in power metrics. |
| Startup / shutdown energy | One-time transition energy added to the node's energy total. |
| Startup / shutdown cost | Optional extra per-transition charge, separate from electricity and rental. |
| Low-power-after minutes | Idle time before the node enters LOW_POWER in its power timeline. |
| Minimum on / off minutes | Hysteresis settings used by the simulated power manager when considering transitions. |
| Sharing support | Whether the server allows resource-budgeted concurrent jobs. Each participating job must also allow sharing. |
| PCIe settings | Legacy illustrative server fields remain available for old profiles. PCIe bandwidth, PCIe efficiency, and job transfer volumes do not affect current runtime, placement, or metrics. |

## How runtime is estimated

For each job, the model derives reference work from the user runtime and iteration count:

```text
compute work = runtime estimate × iterations
memory work (GB) = memory bandwidth demand (GB/s) × runtime estimate (minutes) × 60
```

Effective compute throughput is based on node compute capacity × compute efficiency, adjusted for the power limit and the fixed thermal factor. Effective memory bandwidth is node memory bandwidth × memory efficiency, with the same thermal adjustment. The job-type factors above scale the relevant work.

The model turns compute work into compute minutes and memory work into memory minutes. With overlap `o` between 0 and 1:

```text
device minutes = max(compute minutes, memory minutes) × o
               + (compute minutes + memory minutes) × (1 - o)
```

For an isolated job, that is the predicted runtime (with a small 0.01-minute floor). When jobs overlap, the model applies an additional slowdown based on how much compute and memory demand the node and its peers are using. Concurrent placement is allowed only if total VRAM, compute demand, and memory bandwidth demand fit, and sharing is enabled on the node and all jobs.

This is a deliberately compact model. It does not model CUDA kernels, measured clock speeds, real memory allocation, or actual device contention.

## How costs and energy are calculated

- **Per-job runtime cost:** predicted runtime in minutes / 60 × node hourly price.
- **Node energy:** each power timeline segment contributes watts × segment minutes / 60 in watt-hours (Wh). The model sums ACTIVE, IDLE, LOW_POWER, startup, and shutdown energy; kWh = Wh / 1000.
- **Electricity cost:** simulated kWh × the node tariff in ₹/kWh.
- **Rental cost:** node hourly price × powered node minutes / 60. Powered time includes non-OFF states, including startup and shutdown intervals.
- **Transition cost:** configured startup and shutdown fees.
- **Total operating cost:** electricity cost + rental cost + transition cost.

The `cost` objective uses total operating cost. The job table's GPU cost column is the per-job runtime cost, so it is not the same as total operating cost. Energy, rental, and electricity values are model outputs, not invoices or power-meter readings.

## Scheduling methods

The Optimize and Compare views use these eight methods:

| Method | What it does |
| --- | --- |
| **FCFS** | Sorts jobs by arrival time, then ID. Places each on the compatible node with the earliest finish; ties prefer lower per-job runtime cost. |
| **SJF** | Sorts by the user runtime estimate, then ID. Uses the same earliest-finish placement rule as FCFS. |
| **Priority + Arrival** | Sorts High before Medium before Low, then earlier arrival and ID. The higher-tier job is scheduled first. |
| **Exact Search** | Enumerates compatible server assignments and scores each resulting schedule. For each server it uses a fixed higher-priority, then earlier-arrival job order. It searches assignments, not every possible per-server job sequence, and has an internal time limit. “Exact” is limited to this model and ordering. |
| **Energy-Aware Consolidation** | Uses the GPU-aware simulator and favors already-used nodes to avoid spreading work across extra nodes. It models node power states and only shuts down when the projected low-power savings can recover transition energy and satisfy timing guards. |
| **GPU Resource-Aware Placement** | Uses predicted node-specific runtimes and checks VRAM, compute, memory bandwidth, sharing, and power-start timing. It allows concurrency when the simulated budgets fit and otherwise waits for a release. |
| **Genetic Algorithm** | Searches job orders and server assignments with seeded selection, crossover, mutation, and elitism. Candidate schedules are evaluated by the simulator against the chosen objective and user-set weights. |
| **NSGA-II** | Evolves a set of non-dominated candidate schedules over five measures: energy, total operating cost, makespan, average wait, and priority-weighted wait plus an unscheduled-job penalty. It returns a Pareto front and selects a display schedule with minimum operating cost. |

FCFS, SJF, and Priority + Arrival keep their built-in ordering regardless of the selected objective. The objective affects Exact Search and the GPU-aware methods; Genetic Algorithm uses objective fitness. NSGA-II always maintains its five separate Pareto measures. Comparison runs every method against the same input snapshot.

### Objectives

| Objective | What is minimized |
| --- | --- |
| Cost | Total operating cost: electricity + powered-time rental + transition fees. |
| Energy | Simulated energy in kWh. |
| Makespan | Time at which the last scheduled job finishes. |
| Waiting | Average scheduled-job wait after its arrival. |
| Priority | Sum of each scheduled job's wait multiplied by its tier weight (High 3, Medium 2, Low 1). This favors shorter waits for higher tiers. |
| Balanced | Weighted combination of cost, energy, makespan, average waiting, and priority-weighted waiting. The Optimize view exposes these weights. |

Scalar search methods add a large penalty for each unscheduled job, weighted by its customer tier. Baseline FCFS/SJF/Priority do not optimize these scalar objectives; they provide reference scheduling policies.

## Metrics in the results

- **Scheduled / unscheduled:** number of jobs in the schedule and jobs that have no compatible enabled node.
- **Makespan:** maximum finish time from simulation time zero.
- **Average wait:** average start minus arrival over scheduled jobs.
- **Priority-weighted wait:** sum of wait × tier weight for scheduled jobs.
- **GPU wall-clock active time:** union of scheduled active intervals per enabled node divided by enabled-node count × makespan. The Optimize view separately shows busy-job time, which may exceed 100% when jobs share a GPU. GPU-aware metrics also show time-weighted compute, VRAM, and memory-bandwidth utilization over powered node time.
- **Energy breakdown:** Wh by active, idle, low-power, startup, and shutdown category, plus total Wh and kWh.
- **Operating cost breakdown:** electricity, rental, transition, and total operating cost.
- **Power transitions / transition delay:** count and duration of simulated starts and stops.
- **Contention penalty:** estimated extra runtime from the co-location slowdown factor.

There are no deadline-miss metrics or deadline-based scheduling rules.

## VAM + MODI tab

This tab solves a separate transportation problem. Each enabled node supplies a user-set number of minutes, and each job demands its estimated runtime in minutes. Vogel's Approximation Method (VAM) makes an initial allocation; the Modified Distribution Method (MODI) improves it. Since the model may split one job's minutes across nodes, its result is educational and is not an executable whole-job schedule. The model uses hourly price converted to cost per minute and a large penalty for VRAM-incompatible pairs; it does not model arrivals, priorities, resource concurrency, or node power timelines.

## API overview

The Express server validates JSON and launches `algorithms/run.py` as a fresh Python process for each request. The browser posts a complete jobs-and-servers snapshot to:

- `POST /api/run` — one algorithm, objective, snapshot, and optional optimizer settings.
- `POST /api/compare` — all eight algorithms on the same snapshot.
- `POST /api/transport` — VAM + MODI with a minute-capacity value.

Snapshots with legacy deadline (`dl`) fields remain accepted, but those values are discarded by the workload model, are not validated as scheduling inputs, and do not appear in returned job snapshots or schedules.

## Tests

Run the Python unit tests from `gpu-scheduler/`:

```bash
python3 -m unittest discover -s tests -v
```

The tests cover GPU resource fit, runtime and power estimates, priority scheduling, all eight algorithms, and legacy-field handling.
