"""Validate the archived case and export the latest MILP tables."""
from pathlib import Path
from math import radians, sin, cos, asin, sqrt
import json
import hashlib
import pandas as pd
import geopandas as gpd
import networkx as nx

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results'
OUT.mkdir(exist_ok=True)
poi = pd.read_csv(BASE / 'Data/candidate_pois_with_land_cost.csv')
demand = pd.read_csv(BASE / 'Data/demand_zones_covered_uncovered.csv')
existing = pd.read_csv(BASE / 'Data/existing_evcs_hcmc.csv')
boundary = gpd.read_file(BASE / 'Data/study_boundary.geojson')
selected = pd.read_excel(BASE / 'milp_results.xlsx', sheet_name='Selected_EVCS')
zones = pd.read_excel(BASE / 'milp_results.xlsx', sheet_name='Demand_Zones')
summary = pd.read_excel(BASE / 'milp_results.xlsx', sheet_name='Summary')
manifest = json.loads((BASE/'Data/provenance.json').read_text(encoding='utf-8'))
for name, source in manifest['files'].items():
    assert hashlib.sha256((BASE/'Data'/name).read_bytes()).hexdigest() == source['sha256']

def distance(a, b, c, d):
    a,b,c,d = map(radians, (a,b,c,d))
    return 12742 * asin(sqrt(sin((c-a)/2)**2 + cos(a)*cos(c)*sin((d-b)/2)**2))

assert (len(poi), len(demand), len(existing)) == (402, 36, 24)
for frame, cols in [(poi,['latitude','longitude','commercial_price_adjusted']),
                    (demand,['latitude','longitude','ev_charging_demand','covered_demand','uncovered_demand']),
                    (existing,['Latitude','Longitude','daily_capacity'])]:
    assert frame[cols].notna().all().all()
assert ((demand.covered_demand + demand.uncovered_demand - demand.ev_charging_demand).abs() < 1e-6).all()
graph = nx.DiGraph()
for i, row in existing.iterrows():
    graph.add_edge('source', f'e{i}', capacity=float(row.daily_capacity))
    for j, zone in demand.iterrows():
        if distance(row.Latitude,row.Longitude,zone.latitude,zone.longitude) <= 1:
            graph.add_edge(f'e{i}',f'z{j}',capacity=float(zone.ev_charging_demand))
for j, zone in demand.iterrows():
    graph.add_edge(f'z{j}','sink',capacity=float(zone.ev_charging_demand))
baseline = nx.maximum_flow_value(graph,'source','sink')
assert abs(baseline - demand.covered_demand.sum()) < 1e-6, (baseline, demand.covered_demand.sum())
assert summary.solver_status.iloc[0] == 'Optimal'
assert ((zones.final_served + zones.final_uncovered - zones.demand).abs() < 1e-5).all()
assert (zones.final_uncovered >= -1e-6).all()
assert selected.candidate_idx.is_unique
assert (selected.used_capacity <= selected.daily_capacity + 1e-5).all()
assert abs(selected.used_capacity.sum() - summary.additional_covered.iloc[0]) < 1e-5
for sheet, name in [('Summary','summary'),('Selected_EVCS','selected-stations'),('Demand_Zones','demand-zones')]:
    pd.read_excel(BASE/'milp_results.xlsx',sheet_name=sheet).to_csv(OUT/f'{name}.csv',index=False)
(OUT/'validation.json').write_text(json.dumps({'candidates':len(poi),'demand_zones':len(demand),'existing_stations':len(existing),'baseline_max_flow':baseline,'total_demand':float(demand.ev_charging_demand.sum()),'checks':'Input completeness, demand balance, baseline maximum flow, optimal solver status, unique selected sites, selected capacity and final demand balance passed.'},indent=2),encoding='utf-8')

print("Validation passed; result tables exported. Study maps are the original figures from Model draft.pdf.")
