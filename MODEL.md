# Single-objective formulation

Let I be candidate sites, E existing stations, J demand zones and K the Cartesian product of charger powers and port counts. Eligible pairs Aᶜ and Aᵉ have great-circle distance at most 1 km. Demand Dⱼ and flow are continuous expected charging sessions/day.

- xᵢₖ ∈ {0,1}: install configuration k at site i.
- fᵢⱼ ≥ 0 and gₑⱼ ≥ 0: sessions/day served by new and existing stations.
- uⱼ ≥ 0: unserved demand.

First maximize existing-only served demand under the same eligibility and capacity constraints. Let B be this optimum and U⁰ = ΣDⱼ − B. The packaged input's historical `covered_demand` and `uncovered_demand` columns are retained for provenance but are not used to fix allocations or define the baseline.

## Objective

$$\max\;pq\left(U^0-\sum_j u_j\right)-\sum_{i,k}F_{ik}x_{ik}-cq\sum_{(i,j)\in A^c}f_{ij}.$$

p and c are charging and electricity prices in VND/kWh; q is kWh/session. This is incremental network profit under the author's original accounting convention. Existing service can be reassigned; only the increase over B generates revenue. Electricity cost is charged on all flow at new stations. It is not the sum of standalone new-station profits.

## Constraints

$$\sum_k x_{ik}\le1\quad\forall i,$$
$$\sum_{j:(i,j)\in A^c}f_{ij}\le\sum_k C_kx_{ik}\quad\forall i,$$
$$\sum_{j:(e,j)\in A^e}g_{ej}\le C_e\quad\forall e,$$
$$\sum_{i:(i,j)\in A^c}f_{ij}+\sum_{e:(e,j)\in A^e}g_{ej}+u_j=D_j\quad\forall j.$$

No budget, service floor, grid capacity or travel-time objective is added. Opening no stations is feasible and has zero incremental profit when the existing-only optimum is reproduced.

## Capacity and cost units

For configuration k with power wₖ kW/port and Pₖ ports:

$$C_k=hw_kP_k/q.$$

With land price Lᵢ in million VND/m², area a m²/port, assumed annual rental fraction r, installation cost Hₖ in VND/port and lifetime Y years:

$$F_{ik}=\frac{10^6L_i aP_kr}{365}+\frac{H_kP_k}{365Y}.$$

The 3% rental fraction is a scenario assumption, not a verified lease quotation or legal rate. Straight-line installation allocation excludes discounting, maintenance, tax, queuing and grid upgrades. Input land-price scale is explicitly assumed to be million VND/m² and should be checked against the original schedule before investment use.

## Changes from the thesis script

The source is the author's `Code & Data/MILP.py` with parameters from `env.py`. This release keeps its objective structure, configurations, eight occupied hours/day and 1 km eligibility, while:

1. Converting installation values from million VND to VND and annual assumed rent to daily VND. The old mixed-unit objective is not a valid monetary reference.
2. Recomputing existing-only maximum coverage from packaged inputs instead of trusting saved rounded baseline columns.
3. Removing RL, Torch, Gym and GeoPandas dependencies; coordinates suffice for this distance model.
4. Requiring solver optimal status within the configured tolerance, checking primal residuals, supporting an empty selected-site table and exporting allocations plus input hashes.

The solver objective is scaled by 10⁶ for numerical conditioning; reports convert it back to VND/day. The default solver relative gap is 0.0005 (0.05%). Because solvers omit the constant baseline revenue, the summary separately converts the dual bound back into an upper bound on daily incremental profit and reports the corresponding profit gap. This is a tolerance-certified solution, not a claim of an exact zero-gap optimum. Different station sets or allocations can have equivalent objective values.

For exact preprocessing, each site's configuration capacity is tightened to the smaller of its physical capacity and the total demand in adjacent zones. A configuration is removed if another at the same site has at least as much effective capacity and a lower fixed cost (ties retain the earlier configuration). This preserves the integer optimum while strengthening the LP relaxation.
