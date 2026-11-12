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
