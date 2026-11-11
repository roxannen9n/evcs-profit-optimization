"""Single-objective EVCS planning from processed CSV inputs; PuLP/HiGHS."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
import pulp as pl

ROOT = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def accessible(stations, zones, radius):
    """Great-circle eligibility, in kilometres; never a road-distance estimate."""
    s = np.radians([[float(r["latitude"]), float(r["longitude"])] for r in stations])
    z = np.radians([[float(r["latitude"]), float(r["longitude"])] for r in zones])
    delta = s[:, None, :] - z[None, :, :]
    a = np.sin(delta[:, :, 0] / 2) ** 2 + np.cos(s[:, None, 0]) * np.cos(z[None, :, 0]) * np.sin(delta[:, :, 1] / 2) ** 2
    distances = 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return list(zip(*np.where(distances <= radius)))


def load_inputs(folder):
    candidates = read_csv(folder / "candidates.csv")
    zones = read_csv(folder / "demand_zones.csv")
    existing = read_csv(folder / "existing_stations.csv")
    p = json.loads((folder / "parameters.json").read_text(encoding="utf-8"))
    for rows, key in [(candidates, "candidate_id"), (zones, "zone_id"), (existing, "station_id")]:
        if not rows or len({r[key] for r in rows}) != len(rows):
            raise ValueError(f"Empty table or duplicate {key}")
        for r in rows:
            for coordinate, bound in [("latitude", 90), ("longitude", 180)]:
                value = float(r[coordinate])
                if not math.isfinite(value) or abs(value) > bound:
                    raise ValueError(f"Invalid {coordinate}")
    for rows, field in [(candidates, "commercial_price_adjusted"), (zones, "ev_charging_demand"), (existing, "daily_capacity")]:
        if any(not math.isfinite(float(r[field])) or float(r[field]) < 0 for r in rows):
            raise ValueError(f"Invalid {field}")
    for key in ["service_radius_km", "energy_kwh_per_session", "days_per_year", "installation_lifetime_years", "occupied_hours_per_day", "land_price_vnd_multiplier"]:
        if not math.isfinite(p[key]) or p[key] <= 0:
            raise ValueError(f"Invalid {key}")
    return candidates, zones, existing, p


def solve(data_dir, output_dir, time_limit=600, verify=False, relative_gap=0.0005):
    started = time.perf_counter()
    candidates, zones, existing, p = load_inputs(data_dir)
    demand = [float(r["ev_charging_demand"]) for r in zones]
    capacities = [float(r["daily_capacity"]) for r in existing]
    ce = accessible(candidates, zones, p["service_radius_km"])
    ee = accessible(existing, zones, p["service_radius_km"])
    configs = list(itertools.product(p["power_levels_kw"], p["port_counts"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    solver = pl.HiGHS(msg=True, timeLimit=time_limit, gapRel=relative_gap, gapAbs=1e-8,
                     threads=1, log_to_console=False, log_file=str(output_dir / "solver.log"))
    # Recompute the existing-only maximum coverage with identical eligibility.
    baseline = pl.LpProblem("Existing_network", pl.LpMaximize)
    base_flow = pl.LpVariable.dicts("base", ee, lowBound=0)
    by_existing = {e: [] for e in range(len(existing))}
    by_zone_existing = {j: [] for j in range(len(zones))}
    for e, j in ee:
        by_existing[e].append((e, j))
        by_zone_existing[j].append((e, j))
    for e, edges in by_existing.items():
        baseline += pl.lpSum(base_flow[edge] for edge in edges) <= capacities[e]
    for j, edges in by_zone_existing.items():
        baseline += pl.lpSum(base_flow[edge] for edge in edges) <= demand[j]
    baseline += pl.lpSum(base_flow.values())
    baseline.solve(solver)
    if baseline.sol_status != pl.LpSolutionOptimal:
        raise RuntimeError("Existing-only baseline was not proven optimal")
    base_served = float(pl.value(baseline.objective) or 0)
    model = pl.LpProblem("EVCS_incremental_daily_profit", pl.LpMaximize)
    x = {(i, k): pl.LpVariable(f"x_{i}_{k}", cat="Binary") for i in range(len(candidates)) for k in range(len(configs))}
    f = pl.LpVariable.dicts("new", ce, lowBound=0)
    g = pl.LpVariable.dicts("existing", ee, lowBound=0)
    u = pl.LpVariable.dicts("unserved", range(len(zones)), lowBound=0)
    by_candidate = {i: [] for i in range(len(candidates))}
    by_zone_candidate = {j: [] for j in range(len(zones))}
    for i, j in ce:
        by_candidate[i].append((i, j))
        by_zone_candidate[j].append((i, j))
    q = p["energy_kwh_per_session"]
    config_capacity = [p["occupied_hours_per_day"] * power * ports / q for power, ports in configs]
    fixed = {}
    dominated_configurations = 0
    for i, row in enumerate(candidates):
        model += pl.lpSum(x[i, k] for k in range(len(configs))) <= 1
        adjacent_demand = sum(demand[j] for _, j in by_candidate[i])
        effective = [min(cap, adjacent_demand) for cap in config_capacity]
        model += pl.lpSum(f[edge] for edge in by_candidate[i]) <= pl.lpSum(effective[k] * x[i, k] for k in range(len(configs)))
        for k, (power, ports) in enumerate(configs):
            rent = float(row["commercial_price_adjusted"]) * p["land_price_vnd_multiplier"] * p["land_area_m2_per_port"] * ports * p["annual_land_rent_fraction"] / p["days_per_year"]
            install = p["installation_vnd_per_port"][str(power)] * ports / (p["installation_lifetime_years"] * p["days_per_year"])
            fixed[i, k] = rent + install
        # Exact local dominance: no site can serve more than its adjacent demand.
        # Equal effective capacity with greater cost can never improve profit.
        for k in range(len(configs)):
            dominated = any(effective[l] >= effective[k] and (fixed[i, l] < fixed[i, k] or (fixed[i, l] == fixed[i, k] and l < k)) for l in range(len(configs)))
            if dominated:
                model += x[i, k] == 0
                dominated_configurations += 1
    for e, edges in by_existing.items():
        model += pl.lpSum(g[edge] for edge in edges) <= capacities[e]
    for j in range(len(zones)):
        model += pl.lpSum(f[edge] for edge in by_zone_candidate[j]) + pl.lpSum(g[edge] for edge in by_zone_existing[j]) + u[j] == demand[j]
    # This is incremental network revenue, not all revenue at the new stations.
    additional = sum(demand) - base_served - pl.lpSum(u.values())
    fixed_total = pl.lpSum(fixed[i, k] * var for (i, k), var in x.items())
    operating = p["electricity_price_vnd_per_kwh"] * q * pl.lpSum(f.values())
    model += (p["charging_price_vnd_per_kwh"] * q * additional - fixed_total - operating) / 1e6
    model.solve(solver)
    # Check solution status as well as problem status before exporting results.
    if model.sol_status != pl.LpSolutionOptimal:
        raise RuntimeError(f"Solution not proven optimal: {pl.LpSolution[model.sol_status]}; increase --time-limit")
    residuals = []
    for constraint in model.constraints.values():
        v = constraint.value()
        residuals.append(abs(v) if constraint.sense == 0 else max(0, v) if constraint.sense == -1 else max(0, -v))
    binary_error = max(abs(var.value() - round(var.value())) for var in x.values())
    if max(residuals, default=0) > 1e-4 or binary_error > 1e-5:
        raise RuntimeError("Independent primal feasibility check failed")
    selected = []
    for (i, k), var in x.items():
        if var.value() > 0.5:
            power, ports = configs[k]
            selected.append({"candidate_id": candidates[i]["candidate_id"], "latitude": candidates[i]["latitude"], "longitude": candidates[i]["longitude"], "power_kw_per_port": power, "ports": ports, "capacity_sessions_per_day": config_capacity[k], "served_sessions_per_day": sum(f[edge].value() for edge in by_candidate[i]), "fixed_cost_vnd_per_day": fixed[i, k]})
    final_unserved = sum(var.value() for var in u.values())
    profit = pl.value(model.objective) * 1e6
    info = model.solverModel.getInfo()
    profit_upper_bound = (-info.mip_dual_bound + model.objective.constant) * 1e6
    if profit < -1:
        raise RuntimeError("Negative profit contradicts feasible do-nothing baseline")
    summary = {"solver_status": "Optimal", "candidates": len(candidates), "demand_zones": len(zones), "existing_stations": len(existing), "configurations": len(configs), "candidate_zone_edges": len(ce), "existing_zone_edges": len(ee), "total_demand_sessions_per_day": sum(demand), "baseline_served_sessions_per_day": base_served, "final_served_sessions_per_day": sum(demand) - final_unserved, "additional_served_sessions_per_day": sum(demand) - base_served - final_unserved, "new_stations": len(selected), "incremental_revenue_vnd_per_day": p["charging_price_vnd_per_kwh"] * q * (sum(demand) - base_served - final_unserved), "fixed_cost_vnd_per_day": pl.value(fixed_total), "electricity_cost_vnd_per_day": pl.value(operating), "incremental_profit_vnd_per_day": profit, "maximum_constraint_violation": max(residuals, default=0), "runtime_seconds": time.perf_counter() - started}
    summary['dominated_configurations_removed'] = dominated_configurations
    summary['solver'] = 'HiGHS'
    summary['solver_relative_gap_tolerance'] = relative_gap
    summary['solver_reported_mip_gap'] = info.mip_gap
    summary['profit_upper_bound_vnd_per_day'] = profit_upper_bound
    summary['profit_optimality_gap_fraction'] = max(0, profit_upper_bound - profit) / max(1, abs(profit))
    summary['solver_status'] = 'Optimal within configured tolerance'
    if verify:
        reference = json.loads((ROOT / "results/reference/summary.json").read_text())
        if not math.isclose(base_served, reference['baseline_served_sessions_per_day'], rel_tol=1e-8, abs_tol=1e-4):
            raise AssertionError('Reference baseline mismatch')
        if not math.isclose(profit, reference['incremental_profit_vnd_per_day'], rel_tol=.005, abs_tol=1):
            raise AssertionError('Reference profit differs by more than 0.5%')
        if summary['profit_optimality_gap_fraction'] > .005:
            raise AssertionError('Profit optimality gap exceeds 0.5%')
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "parameters.json").write_text(json.dumps(p, indent=2), encoding="utf-8")
    write_csv(output_dir / "selected_stations.csv", selected, ["candidate_id", "latitude", "longitude", "power_kw_per_port", "ports", "capacity_sessions_per_day", "served_sessions_per_day", "fixed_cost_vnd_per_day"])
    zone_results = [{"zone_id": row["zone_id"], "demand": demand[j], "final_unserved": u[j].value()} for j, row in enumerate(zones)]
    write_csv(output_dir / "demand_balance.csv", zone_results, ["zone_id", "demand", "final_unserved"])
    flows = [{"station_type": kind, "station_id": rows[i][id_field], "zone_id": zones[j]["zone_id"], "sessions_per_day": var.value()} for kind, rows, id_field, flow in [("new", candidates, "candidate_id", f), ("existing", existing, "station_id", g)] for (i, j), var in flow.items() if var.value() > 1e-7]
    write_csv(output_dir / "allocations.csv", flows, ["station_type", "station_id", "zone_id", "sessions_per_day"])
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(data_dir.iterdir()) if path.is_file()}
    (output_dir / "input_sha256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/reproduced")
    parser.add_argument("--time-limit", type=float, default=600)
    parser.add_argument("--verify-reference", action="store_true")
    parser.add_argument("--relative-gap", type=float, default=0.0005, help="HiGHS relative MIP gap; use 0 for a stricter solve")
    args = parser.parse_args()
    solve(args.data_dir, args.output_dir, args.time_limit, args.verify_reference, args.relative_gap)
