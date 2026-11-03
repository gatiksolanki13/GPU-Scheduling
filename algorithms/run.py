import json
import sys

from schedulers import ALGORITHMS, run_algorithm
from transport import solve_transport


def main():
    request = json.load(sys.stdin)
    action = request.get("action")
    snapshot = request.get("snapshot", {})
    objective = request.get("objective", "balanced")
    options = request.get("options", {})

    if action == "run":
        output = run_algorithm(request["algo"], snapshot, objective, options)
    elif action == "compare":
        output = [run_algorithm(key, snapshot, objective, options) for key in ALGORITHMS]
    elif action == "transport":
        output = solve_transport(snapshot, request.get("capacity", 60))
    else:
        raise ValueError(f"unknown action: {action}")

    json.dump(output, sys.stdout)


if __name__ == "__main__":
    main()
