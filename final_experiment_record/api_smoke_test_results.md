# API HTTP smoke test results

Date: 2026-11-03. Application source was not changed for these tests. The server was started on `http://localhost:3100` because another Node process owned port 3000 and did not answer HTTP. The isolated instance returned HTTP 200 for `GET /`. Requests used `POST /api/run`.

## Cases and actual outcomes

| Case | Request | Expected | Actual | Result / response evidence |
|---|---|---:|---:|---|
| 1 | Valid integer IDs, FCFS | 200 | 200 | PASS; `algo=FCFS`, scheduled 1, unscheduled 1, job ID `[1]`; metrics present. |
| 2 | Valid string IDs, FCFS | 200 | 200 | PASS; `algo=FCFS`, scheduled 1, unscheduled 1, job ID `["job-1"]`; metrics present. |
| 3 | Mixed integer and string IDs | 400 | 400 | PASS; `{"error":"job and GPU IDs must all be integers or all be strings"}` |
| 4 | Duplicate job IDs | 400 | 400 | PASS; `{"error":"duplicate job ID"}` |
| 5 | Duplicate GPU IDs | 400 | 400 | PASS; `{"error":"duplicate GPU ID"}` |
| 6 | Missing required job name | 400 | 400 | PASS; `{"error":"invalid jobs, servers, or options"}` |
| 7 | Job VRAM set to 0 | 400 | 400 | PASS; `{"error":"invalid jobs, servers, or options"}` |
| 8 | Valid FCFS | 200 | 200 | PASS; same base response as case 1. Sent as an independent request. |
| 9 | Valid GA | 200 | 200 | PASS; scheduled 1, unscheduled 1; evaluations/generation/metrics fields present. |
| 10 | Valid NSGA-II | 200 | 200 | PASS; scheduled 1, unscheduled 1; Pareto/evaluations/generation/metrics fields present. |
| 11 | Other supported algorithms: SJF, Priority + arrival, Exact Search, Energy-Aware Consolidation, GPU Resource-Aware Placement | 200 each | 200 each | PASS; each response contained one scheduled feasible job and one unscheduled too-large job with metrics. |
| 12 | Unknown algorithm `not-real` | 400 | 400 | PASS; `{"error":"unknown algorithm"}` |
| 13 | Valid FCFS request after invalid algorithm request | 200 | 200 | PASS; response contained one scheduled job, zero unscheduled jobs, power telemetry and metrics. Confirms server remained operational. |

The valid/algorithm smoke fixture deliberately included one feasible job and one job requiring 32 GB VRAM on a 16 GB GPU. Successful responses consistently reported `scheduled=1`, `unscheduled=1`, job `Too-Large` in `uns`, and valid `energyKWh`, `totalOperatingCost`, `makespan`, and `avgWaitingTime` fields. Their top-level response keys included `algo`, `m`, `ms`, `power`, `sch`, `snap`, `status`, and `uns`; Exact Search and metaheuristic responses also included their algorithm-specific fields.

For the FCFS request the actual metrics were: energy `0.04931590413943355 kWh`, operating cost `100.39452723311547 INR`, makespan `5.908496732026144 min`, average waiting `0 min`. Integer and string ID responses preserved the respective ID type in `sch`.

## Server stability and caveat

All invalid payloads returned client errors rather than HTTP 500; subsequent valid algorithm calls returned HTTP 200. Port 3000 was already occupied by PID 21492 (`node`), and requests to it timed out. That process was not stopped or modified. The tested application instance ran on port 3100.

## Reproduction fixture

See `../workload_configuration.json` under `api_smoke_test_request_fixture` and `../gpu_configuration.json` under `api_smoke_test_gpu`. Use objective `balanced`, options `{"seed":7,"population_size":6,"generations":1,"time_limit_seconds":0.1,"planning_horizon_minutes":60}`, and change the indicated IDs/fields for negative cases.
