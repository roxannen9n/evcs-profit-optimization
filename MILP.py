
from pathlib import Path
import numpy as np
import pandas as pd
import pulp as pl
import time
from env import (EVCS_env, configs,elec_price,install_cost,POI,demand_zones_gpd,charge_price,evcs_occupied_hours,
    energy_to_charge,existing_evcs, land_area_per_charger, haversine_distance)


#import env from RL:
env = EVCS_env(configs=configs,elec_price=elec_price,install_cost=install_cost,POI=POI,
    demand_zones=demand_zones_gpd,charging_price=charge_price,evcs_occupied_hours=evcs_occupied_hours,energy_to_charge=energy_to_charge,
    existing_evcs=existing_evcs,land_area_per_charger=land_area_per_charger)
#assign values
I = range(env.n_candidates)       # candidate EVCS
J = range(env.n_demand_zones)     # demand zones
K = range(env.n_configs)          # charger configurations
E = range(len(env.initial_existing_evcs))  # existing EVCS

#demand and uncovered demand; #NOTE - 📌 initial maximum flow network model to assign capacity-demand btw exisitng and demand zones is calculated in demand_zones_h3.py
demand = (env.initial_demand_zones["ev_charging_demand"].to_numpy(dtype=float))
initial_uncovered = (env.initial_uncovered_demand.astype(float))
initial_total_uncovered = float(initial_uncovered.sum())
initial_total_demand = float(demand.sum())
baseline_covered = (initial_total_demand- initial_total_uncovered)

print("=" * 70)
print("MILP DATA SUMMARY")
print("=" * 70)
print("Candidates:", env.n_candidates)
print("Demand zones:", env.n_demand_zones)
print("Configurations:", env.n_configs)
print("Existing EVCS:", len(E))
print("Total demand:", initial_total_demand)
print("Baseline covered:", baseline_covered)
print("Initial uncovered:", initial_total_uncovered)

# configs:incl. power and ports
config_capacity = {}
for k, (power, ports) in enumerate(configs):
    capacity = (evcs_occupied_hours* power* ports/ energy_to_charge)
    config_capacity[k] = float(capacity)

# 🌷candiate and demand zones accessability:
candidate_access = (env.candidate_zone_access.copy())
candidate_edges = []
for i in I:
    for j in J:
        if candidate_access[i, j]:
            candidate_edges.append((i, j))
print("Candidate-demand accessible edges:",len(candidate_edges))

# 🌷 existing evcs and demand zones accessability: #NOTE 📌 can't initiate maximum flow here because there is no capacity at candidate sites
existing_edges = []
for e in E:
    evcs = env.initial_existing_evcs.iloc[e]
    evcs_lat = float(evcs["Latitude"])
    evcs_lon = float(evcs["Longitude"])
    for j in J:
        zone = env.initial_demand_zones.iloc[j]
        zone_lat = float(zone.geometry.y)
        zone_lon = float(zone.geometry.x)
        distance = haversine_distance(evcs_lat,evcs_lon,zone_lat,zone_lon)
        if distance <= 1.0: #FIXME - service radius
            existing_edges.append((e, j))
print("Existing-EVCS-demand edges:",len(existing_edges))

# 🌷 load MILP model
model = pl.LpProblem("EVCS_Placement_MILP",pl.LpMaximize)

# 🌷 decision variables (pl.LpVariable creates a mathematical decision variable)

# x[i,k] = 1 if candidate i chooses configuration k
x = {(i, k):pl.LpVariable(f"x_{i}_{k}",cat="Binary")
    for i in I
    for k in K}

# Candidate flow to demand zone
flow_candidate = {(i, j):pl.LpVariable(f"flow_candidate_{i}_{j}",lowBound=0,cat="Continuous") 
                for (i, j) in candidate_edges}

# Existing EVCS flow to demand zones
flow_existing = {(e, j):pl.LpVariable(f"flow_existing_{e}_{j}",lowBound=0,cat="Continuous")
                for (e, j) in existing_edges}

# Final uncovered demand
uncovered = {j:pl.LpVariable(f"uncovered_{j}",lowBound=0,cat="Continuous")for j in J}

# 🤗 CONSTRAINT:
#     1. only 1 config per candidate
for i in I:
    model += (pl.lpSum(x[i, k] for k in K) <= 1), f"One_config_candidate_{i}"

#     2. candidate capacity constraint
for i in I:
    outgoing_flow = pl.lpSum(flow_candidate[i, j]for j in J if (i, j) in flow_candidate)
    installed_capacity = pl.lpSum(config_capacity[k]* x[i, k]for k in K)
    model += outgoing_flow<=installed_capacity, f"Candidate_capacity_{i}"

#     3. existing EVCS capacity constraint
for e in E:
    capacity_e = float(env.initial_existing_evcs.iloc[e]["daily_capacity"])
    if np.isnan(capacity_e):
        capacity_e = 0.0
    outgoing_flow = pl.lpSum(flow_existing[e, j]for j in J if (e, j) in flow_existing)
    model += outgoing_flow <= capacity_e, f"Existing_capacity_{e}"

#demand balance constraint: total served + uncovered = demand
for j in J:
    existing_service = pl.lpSum(flow_existing[e, j] for e in E
        if (e, j) in flow_existing)
    candidate_service = pl.lpSum(flow_candidate[i, j] for i in I
        if (i, j) in flow_candidate)
    model += (existing_service + candidate_service + uncovered[j] ==float(demand[j])), f"Demand_balance_{j}"

# 🤗 OBJECTIVE
# cost for each candidate:
fixed_cost = {}
for i in I:
    land_price = float(env.land_cost[i])
    for k, (power, ports) in enumerate(configs):
        # Same formula as current RL environment
        land_cost = (land_price* land_area_per_charger* ports)*0.03 #NOTE - 📌chỗ này có % phí thuê đất trên giá đất
        installation_cost = float(install_cost.loc[install_cost["Type"] == str(power),"Cost"].iloc[0])* ports/ (10 * 365)
        fixed_cost[i, k] = (land_cost+ installation_cost) 

# Final uncovered demand
final_total_uncovered = pl.lpSum(uncovered[j]for j in J)
# Additional coverage compared with existing network
additional_covered = (initial_total_uncovered-final_total_uncovered)
# Charging revenue caused by ADDITIONAL coverage
additional_revenue = (charge_price* energy_to_charge*additional_covered)

# Fixed candidate investment cost
total_fixed_cost = pl.lpSum(fixed_cost[i, k]* x[i, k]
    for i in I
    for k in K)

# Operating cost of newly installed EVCS
total_candidate_flow = pl.lpSum(flow_candidate[i, j]for (i, j)in candidate_edges)
total_operating_cost = (elec_price* energy_to_charge*total_candidate_flow)
total_cost = (total_fixed_cost+ total_operating_cost)

# Objective functions: maximize net profit across network
model += (additional_revenue-total_fixed_cost-total_operating_cost)

# 🤗 SOLVE MILP    
print("\n" + "=" * 70)
print("SOLVING MILP")
print("=" * 70)

solver = pl.PULP_CBC_CMD(msg=True,timeLimit=600)
start_time = time.perf_counter()
model.solve(solver)
end_time = time.perf_counter()
runtime_seconds = end_time - start_time
runtime_minutes = runtime_seconds / 60

# Status checking
print("\n" + "=" * 70)
print("MILP RESULT")
print("=" * 70)
print("Status:",pl.LpStatus[model.status])
print("Objective:",pl.value(model.objective))
print(
    "Runtime:",
    f"{runtime_seconds:.2f} seconds "
    f"({runtime_minutes:.2f} minutes)"
)
# selected EVCS list:
selected= []
for i in I:
    for k in K:
        if pl.value(x[i, k]) is not None:
            if pl.value(x[i, k]) > 0.5:
                power, ports = configs[k]
                used_capacity = sum(
                pl.value(flow_candidate[i, j]) or 0
                for j in J
                if (i, j) in flow_candidate)
                station_revenue = (charge_price* energy_to_charge* used_capacity)
                station_operating_cost = (elec_price* energy_to_charge* used_capacity)
                station_fixed_cost = fixed_cost[i, k]
                station_total_cost = (station_fixed_cost+ station_operating_cost)
                station_net_profit = (station_revenue- station_total_cost)
                selected.append({
                    "candidate_idx":i,
                    "longitude":float(env.candidate_x[i]),
                    "latitude":float(env.candidate_y[i]),
                    "land_cost":float(env.land_cost[i]),
                    "config_idx":k,
                    "power":power,
                    "ports":ports,
                    "daily_capacity":config_capacity[k],
                    "used_capacity":used_capacity,
                    "fixed_cost":fixed_cost[i, k],
                    "charging_revenue":station_revenue,
                    "operating_cost":station_operating_cost,
                    "total_cost":station_total_cost,
                    "net_profit":station_net_profit})


selected_df = pd.DataFrame(selected)

poi_lookup = POI.reset_index(drop=True).copy()

selected_df = selected_df.merge(
    poi_lookup[["road_name","parent_road","is_alley","commercial_price_adjusted","ward_name"]],
    left_on="candidate_idx",right_index=True,how="left")

# demand zones summary:
zone = []
for j in J:
    final_uncovered_j = pl.value(uncovered[j])
    final_served_j = (demand[j]- final_uncovered_j)
    zone.append({
        "zone_idx":j,
        "demand":demand[j],
        "initial_uncovered":initial_uncovered[j],
        "final_served":final_served_j,
        "final_uncovered":final_uncovered_j})
zone_df = pd.DataFrame(zone)

# metrics:
final_uncovered_value = float(zone_df["final_uncovered"].sum())
additional_coverage_value = (initial_total_uncovered-final_uncovered_value)
if initial_total_uncovered > 0:
    additional_coverage_ratio = (additional_coverage_value/initial_total_uncovered)
else:
    additional_coverage_ratio = 0.0

print("\nSelected new EVCS:",len(selected_df))
print("Initial uncovered demand:",initial_total_uncovered)
print("Final uncovered demand:",final_uncovered_value)
print("Additional covered demand:",additional_coverage_value)
print("Additional coverage ratio:",f"{additional_coverage_ratio:.2%}")

if len(selected_df) > 0:
    print("\nSELECTED EVCS")
    print(selected_df.to_string(index=False))

# ============================================================
# 20. EXPORT
# ============================================================

output_file = (Path(__file__).resolve().parent/ "milp_results.xlsx")


summary_df = pd.DataFrame([{
    "solver_status":pl.LpStatus[model.status],
    "objective_value":
        pl.value(model.objective),
    "initial_uncovered":initial_total_uncovered,
    "final_uncovered":final_uncovered_value,
    "additional_covered":additional_coverage_value,
    "additional_coverage_ratio":additional_coverage_ratio,
    "number_new_evcs":len(selected_df),
    "runtime_seconds":runtime_seconds,
    "runtime_minutes":runtime_minutes}])

with pd.ExcelWriter(output_file,engine="openpyxl") as writer:
    summary_df.to_excel(writer,sheet_name="Summary",index=False)
    selected_df.to_excel(writer,sheet_name="Selected_EVCS",index=False)
    zone_df.to_excel(writer,sheet_name="Demand_Zones",index=False)
print("\nResults saved to:",output_file)