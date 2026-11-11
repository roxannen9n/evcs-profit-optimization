"""Independently check exported decisions, flows, costs and input hashes."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from run_model import ROOT, read_csv


def audit(folder):
    p = json.loads((folder / 'parameters.json').read_text())
    s = json.loads((folder / 'summary.json').read_text())
    zones = {r['zone_id']: r for r in read_csv(ROOT / 'data/demand_zones.csv')}
    candidates = {r['candidate_id']: r for r in read_csv(ROOT / 'data/candidates.csv')}
    existing = {r['station_id']: r for r in read_csv(ROOT / 'data/existing_stations.csv')}
    selected_rows = read_csv(folder / 'selected_stations.csv')
    selected = {r['candidate_id']: r for r in selected_rows}
    assert len(selected) == len(selected_rows), 'Multiple configurations at one candidate'
    hashes = json.loads((folder / 'input_sha256.json').read_text())
    for name, digest in hashes.items():
        assert hashlib.sha256((ROOT / 'data' / name).read_bytes()).hexdigest() == digest, name
    zone_flow = defaultdict(float)
    station_flow = defaultdict(float)
    for row in read_csv(folder / 'allocations.csv'):
        kind, identifier, zone_id = row['station_type'], row['station_id'], row['zone_id']
        value = float(row['sessions_per_day'])
        assert value >= 0
        station = candidates[identifier] if kind == 'new' else existing[identifier]
        if kind == 'new':
            assert identifier in selected, 'Flow at unopened station'
        a, b = station, zones[zone_id]
        lat1, lat2 = math.radians(float(a['latitude'])), math.radians(float(b['latitude']))
        delta_lat, delta_lon = lat1 - lat2, math.radians(float(a['longitude']) - float(b['longitude']))
        h = math.sin(delta_lat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(delta_lon/2)**2
        distance = 2*6371*math.asin(math.sqrt(min(1, max(0, h))))
        assert distance <= p['service_radius_km'] + 1e-8, 'Ineligible allocation'
        zone_flow[zone_id] += value
        station_flow[kind, identifier] += value
    q = p['energy_kwh_per_session']
    fixed = 0
    for identifier, row in selected.items():
        power, ports = int(row['power_kw_per_port']), int(row['ports'])
        assert power in p['power_levels_kw'] and ports in p['port_counts']
        capacity = power * ports * p['occupied_hours_per_day'] / q
        assert station_flow['new', identifier] <= capacity + 1e-4
        assert math.isclose(station_flow['new', identifier], float(row['served_sessions_per_day']), abs_tol=1e-4)
        land = float(candidates[identifier]['commercial_price_adjusted']) * p['land_price_vnd_multiplier'] * p['land_area_m2_per_port'] * ports * p['annual_land_rent_fraction'] / p['days_per_year']
        installation = p['installation_vnd_per_port'][str(power)] * ports / (p['installation_lifetime_years'] * p['days_per_year'])
        fixed += land + installation
    for identifier, row in existing.items():
        assert station_flow['existing', identifier] <= float(row['daily_capacity']) + 1e-4
    balances = read_csv(folder / 'demand_balance.csv')
    assert len(balances) == len(zones)
    for row in balances:
        assert math.isclose(zone_flow[row['zone_id']] + float(row['final_unserved']), float(zones[row['zone_id']]['ev_charging_demand']), abs_tol=1e-4)
    served = sum(zone_flow.values())
    additional = served - s['baseline_served_sessions_per_day']
    electricity = sum(v for (kind, _), v in station_flow.items() if kind == 'new') * q * p['electricity_price_vnd_per_kwh']
    revenue = additional * q * p['charging_price_vnd_per_kwh']
    for actual, expected in [(fixed, s['fixed_cost_vnd_per_day']), (electricity, s['electricity_cost_vnd_per_day']), (revenue, s['incremental_revenue_vnd_per_day']), (revenue - fixed - electricity, s['incremental_profit_vnd_per_day'])]:
        assert math.isclose(actual, expected, rel_tol=1e-8, abs_tol=.1), (actual, expected)
    print('Export audit passed: input hashes, site choices, eligibility, capacities, demand balance and daily profit.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results/reference')
    audit(parser.parse_args().results_dir)
