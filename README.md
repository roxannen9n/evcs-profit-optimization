# EVCS Profit Optimization

Mixed-integer linear programming model for electric vehicle charging station placement and profit optimization in Ho Chi Minh City.

## Files

- `MILP.py`: optimization model using PuLP and CBC.
- `env.py`: environment, input loading, accessibility, and cost calculations.
- `Data/`: candidate sites, existing charging stations, and demand zones.
- `results/`: study maps showing demand zones, candidate sites, and selected stations.

## Run

Install the dependencies with Python 3.12, then run the model:

```bash
pip install -r requirements.txt
python MILP.py
```

## Study maps


![Demand zones](results/demand-zones-heatmap.png)

![Existing and candidate stations](results/existing-vs-candidate.png)

![Existing and selected stations](results/existing-vs-selected.png)
