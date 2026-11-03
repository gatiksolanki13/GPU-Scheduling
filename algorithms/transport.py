from metrics import js_round, number_text

BIG_M = 1e4


def money(value):
    return "\u20b9" + number_text(js_round(value * 100) / 100)


def hundredths(value):
    return number_text(js_round(value * 100) / 100)


def second_smallest_gap(values):
    ordered = sorted(values)
    return ordered[1] - ordered[0] if len(ordered) > 1 else ordered[0]


def solve_transport(snap, capacity):
    servers = [s for s in snap["servers"] if s["on"]]
    jobs = snap["jobs"]
    if not servers or not jobs:
        return None

    supply = [capacity for _ in servers]
    demand = [j["rt"] for j in jobs]
    row_names = [s["name"] for s in servers]
    col_names = [j["name"] for j in jobs]
    cost = [
        [s["cost"] / 60 if s["vram"] >= j["vram"] else BIG_M for j in jobs]
        for s in servers
    ]

    total_supply, total_demand = sum(supply), sum(demand)
    if total_supply > total_demand:
        demand.append(total_supply - total_demand)
        col_names.append("Dummy (unused capacity)")
        for row in cost:
            row.append(0)
    elif total_demand > total_supply:
        supply.append(total_demand - total_supply)
        row_names.append("Dummy (unmet demand)")
        cost.append([BIG_M for _ in jobs])

    m, n = len(supply), len(demand)
    alloc = [[0] * n for _ in range(m)]
    basis = {}
    trace = []
    row_left, col_left = list(supply), list(demand)
    open_rows, open_cols = list(range(m)), list(range(n))

    while open_rows and open_cols:
        best_penalty, best_line = -1, None
        for i in open_rows:
            penalty = second_smallest_gap([cost[i][j] for j in open_cols])
            if penalty > best_penalty:
                best_penalty, best_line = penalty, ("r", i)
        for j in open_cols:
            penalty = second_smallest_gap([cost[i][j] for i in open_rows])
            if penalty > best_penalty:
                best_penalty, best_line = penalty, ("c", j)

        if best_line[0] == "r":
            ci = best_line[1]
            cj = sorted(open_cols, key=lambda a: cost[ci][a])[0]
        else:
            cj = best_line[1]
            ci = sorted(open_rows, key=lambda a: cost[a][cj])[0]

        quantity = min(row_left[ci], col_left[cj])
        alloc[ci][cj] = quantity
        basis[(ci, cj)] = True
        row_left[ci] -= quantity
        col_left[cj] -= quantity
        line = (
            f"row {row_names[best_line[1]]}"
            if best_line[0] == "r"
            else f"column {col_names[best_line[1]]}"
        )
        trace.append(
            f"VAM: max penalty {hundredths(best_penalty)} on {line} \u2192 allocate "
            f"{number_text(quantity)} min to {row_names[ci]} / {col_names[cj]} "
            f"(cost {hundredths(cost[ci][cj])}/min)"
        )

        if (row_left[ci] == 0 and len(open_rows) > 1) or len(open_cols) == 1:
            open_rows.remove(ci)
        else:
            open_cols.remove(cj)
        if (
            len(open_rows) == 1
            and len(open_cols) == 1
            and row_left[open_rows[0]] == 0
            and col_left[open_cols[0]] == 0
        ):
            break

    parent = list(range(m + n))

    def find(a):
        if parent[a] != a:
            parent[a] = find(parent[a])
        return parent[a]

    for i, j in basis:
        parent[find(i)] = find(m + j)
    for i in range(m):
        if len(basis) >= m + n - 1:
            break
        for j in range(n):
            if len(basis) >= m + n - 1:
                break
            if (i, j) not in basis and find(i) != find(m + j):
                basis[(i, j)] = True
                parent[find(i)] = find(m + j)
                trace.append(
                    "Degeneracy: added zero-allocation basic cell "
                    f"{row_names[i]} / {col_names[j]}"
                )

    def total_cost():
        total = 0
        for i, row in enumerate(alloc):
            line_total = 0
            for j, value in enumerate(row):
                line_total += value * cost[i][j]
            total += line_total
        return total

    trace.append(f"VAM initial cost = {money(total_cost())}")

    for iteration in range(1, 51):
        u, v = [None] * m, [None] * n
        u[0] = 0
        for _ in range(m + n):
            for i, j in basis:
                if u[i] is not None and v[j] is None:
                    v[j] = cost[i][j] - u[i]
                elif v[j] is not None and u[i] is None:
                    u[i] = cost[i][j] - v[j]

        entering, lowest = None, -1e-9
        for i in range(m):
            for j in range(n):
                if (i, j) not in basis:
                    reduced = cost[i][j] - u[i] - v[j]
                    if reduced < lowest:
                        lowest, entering = reduced, (i, j)
        if entering is None:
            trace.append(
                f"MODI iteration {iteration}: all reduced costs \u2265 0 \u2192 "
                "optimal for the transportation model"
            )
            break

        ei, ej = entering

        def on_path(path, i, j):
            return any(a[0] == i and a[1] == j for a in path)

        def find_cycle(path, horizontal):
            r, c = path[-1]
            if horizontal:
                for k in range(n):
                    if k != c and (r, k) in basis and not on_path(path, r, k):
                        if k == ej and r != ei:
                            return path + [[r, k]]
                        found = find_cycle(path + [[r, k]], False)
                        if found:
                            return found
            else:
                for k in range(m):
                    if k != r and (k, c) in basis and not on_path(path, k, c):
                        found = find_cycle(path + [[k, c]], True)
                        if found:
                            return found
            return None

        cycle = find_cycle([[ei, ej]], True)
        if not cycle:
            trace.append("MODI: cycle not found, stopping")
            break

        shift, leaving = float("inf"), None
        for k, (i, j) in enumerate(cycle):
            if k % 2 and alloc[i][j] < shift:
                shift, leaving = alloc[i][j], (i, j)
        for k, (i, j) in enumerate(cycle):
            alloc[i][j] += -shift if k % 2 else shift
        basis[(ei, ej)] = True
        del basis[leaving]
        trace.append(
            f"MODI iteration {iteration}: enter {row_names[ei]} / {col_names[ej]} "
            f"(reduced cost {hundredths(lowest)}), shift {number_text(shift)} min, "
            f"cost now {money(total_cost())}"
        )

    return {
        "x": alloc,
        "c": cost,
        "rn": row_names,
        "cn": col_names,
        "tr": trace,
        "tot": total_cost(),
        "sup": supply,
        "dem": demand,
    }
