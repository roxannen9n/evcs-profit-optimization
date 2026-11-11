"""Render the saved reference solution without rerunning the solver."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_model import ROOT, read_csv


def main():
    result = ROOT / 'results/reference'
    summary = json.loads((result / 'summary.json').read_text())
    candidates = read_csv(ROOT / 'data/candidates.csv')
    zones = read_csv(ROOT / 'data/demand_zones.csv')
    existing = read_csv(ROOT / 'data/existing_stations.csv')
    selected = read_csv(result / 'selected_stations.csv')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': '#f7f9fc', 'axes.facecolor': '#f7f9fc', 'axes.titleweight': 'bold'})
    fig, ax = plt.subplots(figsize=(10, 8))
    demand = ax.scatter([float(r['longitude']) for r in zones], [float(r['latitude']) for r in zones], c=[float(r['ev_charging_demand']) for r in zones], s=5, cmap='Blues', alpha=.55, label='Demand-zone centroids', rasterized=True)
    ax.scatter([float(r['longitude']) for r in candidates], [float(r['latitude']) for r in candidates], s=8, c='#b5bdc9', alpha=.55, label='Candidate sites')
    ax.scatter([float(r['longitude']) for r in existing], [float(r['latitude']) for r in existing], s=16, c='#42536d', marker='+', alpha=.65, label='Existing stations')
    ax.scatter([float(r['longitude']) for r in selected], [float(r['latitude']) for r in selected], s=52, c='#e26b35', edgecolors='white', linewidths=.5, label=f"Selected new stations ({len(selected)})", zorder=5)
    ax.set(title='Profit-driven EV charging station expansion', xlabel='Longitude (WGS84)', ylabel='Latitude (WGS84)')
    ax.set_aspect(1 / .98)
    ax.set_xlim(min(float(r['longitude']) for r in candidates) - .025, max(float(r['longitude']) for r in candidates) + .025)
    ax.set_ylim(min(float(r['latitude']) for r in candidates) - .025, max(float(r['latitude']) for r in candidates) + .025)
    ax.legend(loc='upper left', fontsize=9, frameon=True)
    fig.colorbar(demand, ax=ax, shrink=.65, label='Estimated charging sessions/day per zone')
    fig.text(.09, .025, 'Candidate-site area; demand zones outside this view remain in the optimization.', fontsize=9, color='#516078')
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(result / 'selected_stations.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].bar(['Existing network', 'Expanded network'], [summary['baseline_served_sessions_per_day'], summary['final_served_sessions_per_day']], color=['#8294af', '#137d86'], width=.55)
    axes[0].set(title='Modeled daily service', ylabel='Charging sessions/day')
    values = [summary[k] / 1e6 for k in ['incremental_revenue_vnd_per_day', 'electricity_cost_vnd_per_day', 'fixed_cost_vnd_per_day', 'incremental_profit_vnd_per_day']]
    axes[1].bar(['Additional\nrevenue', 'Electricity\ncost', 'Fixed\ncost', 'Incremental\nprofit'], values, color=['#137d86', '#8294af', '#8294af', '#e26b35'], width=.6)
    axes[1].set(title='Incremental network economics', ylabel='Million VND/day')
    for ax in axes:
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
        for bar in ax.patches:
            ax.annotate(f'{bar.get_height():,.1f}', (bar.get_x() + bar.get_width()/2, bar.get_height()), ha='center', va='bottom', fontsize=9, xytext=(0, 4), textcoords='offset points')
        ax.margins(y=.18)
    fig.text(.07, .02, f"Scenario estimates with daily VND accounting; certified profit gap: {summary['profit_optimality_gap_fraction']:.2%}.", fontsize=9, color='#516078')
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(result / 'coverage_and_profit.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
