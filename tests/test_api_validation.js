const test = require("node:test");
const assert = require("node:assert/strict");
const { validCapacity, validOptions, validSnapshot, snapshotIdError } = require("../request_validation");

const valid = () => ({
  servers: [{ id: 1, name: "GPU-1", vram: 16, cost: 100, on: true }],
  jobs: [{ id: 1, name: "Job-1", vram: 4, rt: 10, pri: "Medium" }],
});

test("accepts valid snapshots and backward-compatible omitted defaults", () => {
  assert.equal(validSnapshot(valid()), true);
});

test("requires nonempty names and valid unique identifiers", () => {
  const missing = valid();
  delete missing.jobs[0].id;
  assert.equal(validSnapshot(missing), false);
  const duplicate = valid();
  duplicate.jobs.push({ ...duplicate.jobs[0], name: "Job-2" });
  assert.equal(validSnapshot(duplicate), false);
  const badName = valid();
  badName.servers[0].name = "  ";
  assert.equal(validSnapshot(badName), false);
});

test("accepts homogeneous integer or string IDs and rejects mixed/duplicate IDs clearly", () => {
  const numeric = valid();
  numeric.jobs.push({ ...numeric.jobs[0], id: 2, name: "Job-2" });
  assert.equal(validSnapshot(numeric), true);
  const strings = valid();
  strings.servers[0].id = "gpu-1";
  strings.jobs[0].id = "job-1";
  assert.equal(validSnapshot(strings), true);
  for (const [collection, pair, expected] of [
    ["jobs", [1, "1"], "job and GPU IDs must all be integers or all be strings"],
    ["servers", [1, "1"], "job and GPU IDs must all be integers or all be strings"],
    ["jobs", [1, 1], "duplicate job ID"],
    ["servers", ["gpu", "gpu"], "duplicate GPU ID"],
  ]) {
    const data = valid();
    const original = { ...data[collection][0] };
    data[collection].push({ ...original, id: pair[1], name: `${original.name}-2` });
    data[collection][0].id = pair[0];
    const other = collection === "jobs" ? "servers" : "jobs";
    if (typeof pair[0] === "string") data[other][0].id = `other-${pair[0]}`;
    assert.equal(validSnapshot(data), false);
    assert.equal(snapshotIdError(data), expected);
  }
  const crossCollection = valid();
  crossCollection.servers[0].id = "gpu-1";
  assert.equal(validSnapshot(crossCollection), false);
  assert.equal(snapshotIdError(crossCollection), "job and GPU IDs must all be integers or all be strings");
});

test("rejects invalid GPU/job resource values and nonpositive iterations", () => {
  const badGpu = valid();
  badGpu.servers[0].compute_capacity = 0;
  assert.equal(validSnapshot(badGpu), false);
  const badIterations = valid();
  badIterations.jobs[0].iterations = 0;
  assert.equal(validSnapshot(badIterations), false);
  const badWork = valid();
  badWork.jobs[0].compute_work = -1;
  assert.equal(validSnapshot(badWork), false);
});

test("validates optimizer settings before Python dispatch", () => {
  assert.equal(validOptions({ seed: 4, population_size: 16, generations: 8,
    time_limit_seconds: 1, planning_horizon_minutes: 120,
    weights: { cost: 1, energy: 0.5 } }), true);
  assert.equal(validOptions({ population_size: 101 }), false);
  assert.equal(validOptions({ generations: 0 }), false);
  assert.equal(validOptions({ time_limit_seconds: 0.01 }), false);
  assert.equal(validOptions({ seed: "seed" }), false);
  assert.equal(validOptions({ weights: { energy: -1 } }), false);
});

test("requires transport capacity to be a finite positive number", () => {
  assert.equal(validCapacity(60), true);
  assert.equal(validCapacity("60"), false);
  assert.equal(validCapacity(0), false);
  assert.equal(validCapacity(Number.POSITIVE_INFINITY), false);
});
