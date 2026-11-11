"""Tiny, analytically solvable instances exercise the economic model."""
import json
from pathlib import Path
import tempfile
import unittest

from run_model import ROOT, accessible, solve, write_csv


class ModelTests(unittest.TestCase):
    def instance(self, installation):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        data = root / 'data'
        data.mkdir()
        write_csv(data / 'candidates.csv', [{'candidate_id': 'C0', 'latitude': 10, 'longitude': 106, 'commercial_price_adjusted': 0}], ['candidate_id', 'latitude', 'longitude', 'commercial_price_adjusted'])
        write_csv(data / 'demand_zones.csv', [{'zone_id': 'Z0', 'latitude': 10, 'longitude': 106, 'ev_charging_demand': 10}], ['zone_id', 'latitude', 'longitude', 'ev_charging_demand'])
        write_csv(data / 'existing_stations.csv', [{'station_id': 'E0', 'latitude': 10, 'longitude': 106, 'daily_capacity': 4}], ['station_id', 'latitude', 'longitude', 'daily_capacity'])
        p = json.loads((ROOT / 'data/parameters.json').read_text())
        p.update(energy_kwh_per_session=1, charging_price_vnd_per_kwh=3, electricity_price_vnd_per_kwh=2, power_levels_kw=[20], port_counts=[3], installation_vnd_per_port={'20': installation})
        (data / 'parameters.json').write_text(json.dumps(p))
        return data, root / 'output'

    def test_incremental_profit_and_existing_capacity(self):
        data, out = self.instance(0)
        result = solve(data, out)
        self.assertAlmostEqual(result['baseline_served_sessions_per_day'], 4)
        self.assertAlmostEqual(result['additional_served_sessions_per_day'], 6)
        self.assertAlmostEqual(result['incremental_profit_vnd_per_day'], 6)
        self.assertEqual(result['new_stations'], 1)

    def test_do_nothing_is_feasible_when_installation_is_expensive(self):
        data, out = self.instance(1e9)
        result = solve(data, out)
        self.assertEqual(result['new_stations'], 0)
        self.assertAlmostEqual(result['additional_served_sessions_per_day'], 0)
        self.assertAlmostEqual(result['incremental_profit_vnd_per_day'], 0)

    def test_inaccessible_demand_is_unserved(self):
        data, out = self.instance(0)
        write_csv(data / 'demand_zones.csv', [{'zone_id': 'Z0', 'latitude': 11, 'longitude': 106, 'ev_charging_demand': 10}], ['zone_id', 'latitude', 'longitude', 'ev_charging_demand'])
        result = solve(data, out)
        self.assertAlmostEqual(result['final_served_sessions_per_day'], 0)
        self.assertEqual(result['new_stations'], 0)

    def test_radius_uses_kilometres(self):
        stations = [{'latitude': 0, 'longitude': 0}]
        zones = [{'latitude': 0, 'longitude': .008}, {'latitude': 0, 'longitude': .01}]
        self.assertEqual(accessible(stations, zones, 1), [(0, 0)])


if __name__ == '__main__':
    unittest.main()
