# Data preparation

The repository contains processed optimization inputs from the author's EV charging station planning project in Ho Chi Minh City. Raw GIS archives and intermediate files are excluded.

| Dataset | Preparation | Key fields |
|---|---|---|
| Candidate sites | OpenStreetMap points of interest joined to roads, wards and commercial land-price coefficients | Coordinates, road/ward names, land price, matching method and alley adjustment |
| Demand zones | H3 resolution-8 cells with aggregated population and estimated daily charging demand | Zone/H3 IDs, coordinates, population and sessions/day |
| Existing stations | Mapped charging station inventory with estimated capacity | Station ID, name, coordinates, charger description and sessions/day |

Demand is estimated as population × 0.065 car adoption × 0.10 EV share × 40/(330 × 0.60), rounded to three decimals. It represents a planning scenario. Land coefficients use the processed local 2026 land-price schedule; the model treats them as million VND/m² and assumes annual rent equal to 3% of land value. Candidate sites are planning alternatives rather than confirmed available plots.

The source files are `candidate_pois_with_land_cost.csv`, `demand_zones_covered_uncovered.csv` and `existing_evcs_hcmc.csv`. Packaging adds stable candidate/station IDs, normalizes coordinate headers and uses UTF-8. Numerical values and original row order are preserved. `data/provenance.json` records row counts, transformations and source SHA-256 checksums.

Historical coverage columns are retained for context; the model recomputes the existing-network baseline from station capacities and the same 1 km eligibility rule used for new sites. Coordinates are WGS84 decimal degrees.

References: [OpenStreetMap contributors](https://www.openstreetmap.org/copyright), [H3](https://h3geo.org/), and the author's processed population, station and land-price inputs. Original population-raster and station-KML versions are not recorded; upstream data rights apply.
