# Final experiment record

This directory stores the final HTTP smoke-test evidence and the exact configuration/results from the final eight-algorithm and 30-seed experiments. It is separate from application source.

## Files

- `api_smoke_test_results.md`: end-to-end API cases, HTTP status codes, response evidence, and the port-3000 caveat.
- `workload_configuration.json`: six-job experiment input and separate API smoke fixture.
- `gpu_configuration.json`: three GPU settings and normalization defaults.
- `algorithm_parameters.json`: shared invocation settings and per-algorithm effective settings.
- `random_seeds.json`: comparison seed and all 30 GA/NSGA-II seeds.
- `final_results.json`: one-run comparison metrics and seed experiment mean/sample-standard-deviation summaries.
- `experiment_methodology.md`: procedures, checks, and limitations.

## Reproduction

1. From the project root install dependencies if needed, then start the server with `npm start`. If port 3000 is occupied, set `PORT=3100` (PowerShell: `$env:PORT='3100'`) and start `node server.js`.
2. Replay the API cases in `api_smoke_test_results.md` as JSON `POST /api/run` requests. Use `workload_configuration.json` and `gpu_configuration.json` for the smoke payload; each negative case changes the specified field or IDs.
3. For the six-job benchmark, load the `jobs` and `gpus` from the corresponding configuration files and invoke all eight algorithm keys from `algorithm_parameters.json` with objective `balanced` and the recorded options. The source-level Python entry point is `algorithms.schedulers.run_algorithm(key, snapshot, objective, options)`; the HTTP entry point is `POST /api/run`.
4. Repeat `genetic` and `nsga2` with each seed in `random_seeds.json`, holding workload, objective, population, generations, time limit and planning horizon fixed. Aggregate with arithmetic mean and sample standard deviation.
5. Compare the resulting metrics with `final_results.json`. Small differences in elapsed milliseconds are expected across machines and system loads.

The options for baseline methods are recorded for completeness; several are deliberately ignored by those methods. `algorithm_parameters.json` documents which settings are actually consumed.
