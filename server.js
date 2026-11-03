const express = require("express");
const path = require("path");
const { spawn } = require("child_process");
const { validSnapshot, snapshotIdError, validOptions, validCapacity } = require("./request_validation");

const PORT = process.env.PORT || 3000;
const PYTHON = process.env.PYTHON || (process.platform === "win32" ? "python" : "python3");
const RUNNER = path.join(__dirname, "algorithms", "run.py");
const ALGORITHMS = ["fcfs", "sjf", "prio", "exact", "energy", "gpu_aware", "genetic", "nsga2"];
const OBJECTIVES = ["cost", "energy", "makespan", "waiting", "priority", "balanced"];
const SOLVER_TIMEOUT_MS = 30000;

const app = express();
app.use(express.json({ limit: "1mb" }));
app.use(express.static(path.join(__dirname, "public")));

function solve(payload) {
  return new Promise((resolve, reject) => {
    const child = spawn(PYTHON, [RUNNER]);
    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => {
      child.kill();
      reject(new Error("solver timed out"));
    }, SOLVER_TIMEOUT_MS);

    child.stdout.on("data", (chunk) => (stdout += chunk));
    child.stderr.on("data", (chunk) => (stderr += chunk));
    child.on("error", (err) => {
      clearTimeout(timer);
      reject(new Error(`could not start ${PYTHON}: ${err.message}`));
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (code !== 0) {
        const lines = stderr.trim().split("\n");
        return reject(new Error(lines[lines.length - 1] || "solver failed"));
      }
      try {
        resolve(JSON.parse(stdout));
      } catch {
        reject(new Error("solver returned invalid output"));
      }
    });
    child.stdin.end(JSON.stringify(payload));
  });
}

function route(action, check) {
  app.post(`/api/${action}`, async (req, res) => {
    const body = req.body || {};
    if (!validSnapshot(body.snapshot) || !validOptions(body.options)) {
      return res.status(400).json({ error: snapshotIdError(body.snapshot) || "invalid jobs, servers, or options" });
    }
    const problem = check(body);
    if (problem) return res.status(400).json({ error: problem });
    try {
      res.json(await solve({ ...body, action }));
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });
}

const objectiveOk = (b) => (OBJECTIVES.includes(b.objective) ? null : "unknown objective");
route("run", (b) => (ALGORITHMS.includes(b.algo) ? objectiveOk(b) : "unknown algorithm"));
route("compare", objectiveOk);
route("transport", (b) => (validCapacity(b.capacity) ? null : "capacity must be a finite number greater than 0"));

app.listen(PORT, () => console.log(`GPU Scheduler running at http://localhost:${PORT}`));
