const JOB_TYPES = ["Training", "Inference", "Batch Processing", "Memory Intensive", "Compute Intensive"];
const PRIORITIES = ["High", "Medium", "Low"];
const POWER_STATES = ["OFF", "IDLE", "LOW_POWER", "ACTIVE"];
const WEIGHT_KEYS = ["cost", "energy", "makespan", "waiting", "priority"];
const SERVER_NONNEGATIVE = [
  "pcie_bandwidth", "active_power_w", "idle_power_w", "low_power_w",
  "startup_energy_wh", "shutdown_energy_wh", "startup_cost", "shutdown_cost",
  "startup_delay_s", "shutdown_delay_s", "electricity_tariff", "min_on_minutes",
  "min_off_minutes", "low_power_after_minutes", "thermal_threshold_c", "temperature_c",
];
const SERVER_POSITIVE = ["compute_capacity", "memory_bandwidth", "power_limit_w"];
const SERVER_EFFICIENCIES = ["compute_efficiency", "memory_efficiency", "pcie_efficiency", "overlap_compute_memory"];
const JOB_NONNEGATIVE = ["compute_demand", "memory_bandwidth_demand", "memory_work_gb"];

const finite = (value) => typeof value === "number" && Number.isFinite(value);
const record = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
const validId = (value) => (typeof value === "string" && value.trim().length > 0) ||
  (finite(value) && Number.isSafeInteger(value));
const uniqueIds = (items) => {
  const ids = items.map((item) => item.id);
  return new Set(ids).size === ids.length;
};
const consistentIdType = (items) => items.length < 2 ||
  items.every((item) => typeof item.id === typeof items[0].id);
const validName = (value) => typeof value === "string" && value.trim().length > 0;

function validSnapshot(snapshot) {
  if (!record(snapshot) || !Array.isArray(snapshot.servers) || !Array.isArray(snapshot.jobs)) return false;
  if (!snapshot.servers.every((gpu) => record(gpu) && validId(gpu.id) && validName(gpu.name) &&
      typeof gpu.on === "boolean" && finite(gpu.vram) && gpu.vram > 0 &&
      finite(gpu.cost) && gpu.cost >= 0 &&
      (gpu.initial_state === undefined || POWER_STATES.includes(gpu.initial_state)) &&
      (gpu.supports_sharing === undefined || typeof gpu.supports_sharing === "boolean") &&
      SERVER_POSITIVE.every((key) => gpu[key] === undefined || (finite(gpu[key]) && gpu[key] > 0)) &&
      SERVER_NONNEGATIVE.every((key) => gpu[key] === undefined || (finite(gpu[key]) && gpu[key] >= 0)) &&
      SERVER_EFFICIENCIES.every((key) => gpu[key] === undefined || (finite(gpu[key]) && gpu[key] >= 0 && gpu[key] <= 1)))) return false;
  if (!snapshot.jobs.every((job) => record(job) && validId(job.id) && validName(job.name) &&
      finite(job.rt) && job.rt > 0 && finite(job.vram) && job.vram > 0 && PRIORITIES.includes(job.pri) &&
      (job.arr === undefined || (finite(job.arr) && job.arr >= 0)) &&
      (job.sharing === undefined || typeof job.sharing === "boolean") &&
      JOB_TYPES.includes(job.job_type || "Training") &&
      JOB_NONNEGATIVE.every((key) => job[key] === undefined ||
        (key === "memory_work_gb" && job[key] === null) || (finite(job[key]) && job[key] >= 0)) &&
      (job.iterations === undefined || (finite(job.iterations) && job.iterations > 0)) &&
      (job.compute_work === undefined || job.compute_work === null || (finite(job.compute_work) && job.compute_work > 0)))) return false;
  return uniqueIds(snapshot.servers) && uniqueIds(snapshot.jobs) &&
    consistentIdType([...snapshot.servers, ...snapshot.jobs]);
}

function snapshotIdError(snapshot) {
  if (!record(snapshot) || !Array.isArray(snapshot.servers) || !Array.isArray(snapshot.jobs)) return null;
  const allItems = [...snapshot.servers, ...snapshot.jobs];
  if (allItems.every((item) => record(item) && validId(item.id)) && !consistentIdType(allItems)) {
    return "job and GPU IDs must all be integers or all be strings";
  }
  for (const [label, items] of [["GPU", snapshot.servers], ["job", snapshot.jobs]]) {
    if (!items.every((item) => record(item) && validId(item.id))) continue;
    if (!consistentIdType(items)) return `${label} IDs must all be integers or all be strings`;
    if (!uniqueIds(items)) return `duplicate ${label} ID`;
  }
  return null;
}

function validOptions(options) {
  if (options === undefined || options === null) return true;
  if (!record(options)) return false;
  if (options.seed !== undefined && (!finite(options.seed) || !Number.isSafeInteger(options.seed))) return false;
  if (options.population_size !== undefined && (!finite(options.population_size) ||
      !Number.isInteger(options.population_size) || options.population_size < 4 || options.population_size > 100)) return false;
  if (options.generations !== undefined && (!finite(options.generations) ||
      !Number.isInteger(options.generations) || options.generations < 1 || options.generations > 100)) return false;
  if (options.time_limit_seconds !== undefined && (!finite(options.time_limit_seconds) ||
      options.time_limit_seconds < 0.05 || options.time_limit_seconds > 10)) return false;
  if (options.planning_horizon_minutes !== undefined && (!finite(options.planning_horizon_minutes) ||
      options.planning_horizon_minutes <= 0)) return false;
  if (options.weights !== undefined) {
    if (!record(options.weights) || !WEIGHT_KEYS.every((key) => options.weights[key] === undefined ||
        (finite(options.weights[key]) && options.weights[key] >= 0))) return false;
  }
  return true;
}

function validCapacity(capacity) {
  return finite(capacity) && capacity > 0;
}

module.exports = { validSnapshot, snapshotIdError, validOptions, validCapacity };
