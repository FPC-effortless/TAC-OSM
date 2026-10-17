"""Standalone scientific plots from audited fixed-protocol records; no fitting."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tac_osm.development.adaptive_repair import LEVELS

LABELS = {'oracle': 'True-rate oracle', 'stationary_beta': 'Stationary Beta',
          'discounted_beta': 'Forgetting Beta', 'hmm_mixture_dp': 'Bayesian mixture DP',
          'gru128': 'GRU-128', 'mtsk': 'MTSK', 'prior': 'Fixed prior', 'gru32': 'GRU-32',
          'hmm_plugin': 'HMM plug-in', 'mtsk_frozen_kernel': 'Frozen kernel', 'mtsk_reset': 'Reset MTSK'}
COLORS = {'oracle': '#202020', 'stationary_beta': '#858585', 'discounted_beta': '#3182bd',
          'hmm_mixture_dp': '#7551a4', 'gru128': '#21966f', 'mtsk': '#d97721'}


def plot_review(folder):
    folder = Path(folder)
    supplement = json.loads((folder/'supplement.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    modes = ['intact', 'zeroed_probabilities', 'shuffled_forecast_rows']
    for ax, (family, label) in zip(axes, [('in_regime', 'In regime'), ('shifted', 'Shifted')]):
        values = [supplement['lesions'][family][mode]['mean_cost_per_entity'] for mode in modes]
        bars = ax.bar(['Intact', 'Zeroed', 'Shuffled'], values, color=['#21966f', '#d95959', '#d99b32'], width=.62)
        for bar, v in zip(bars, values): ax.text(bar.get_x()+bar.get_width()/2, v+.03, f'{v:.4f}', ha='center', fontsize=10)
        ax.set_title(label); ax.set_ylim(0, 2.8); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    axes[0].set_ylabel('Actual native cost per entity')
    fig.suptitle('DEV-053A: destructive forecast interventions change decisions and cost')
    fig.text(.5, .01, 'Frozen joint checkpoints · 256 paired worlds/cohort · 5 initializations reuse worlds · retrospective diagnostic', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .92))
    fig.savefig(folder/'forecast-lesions.png', dpi=180)
    fig.savefig(folder/'forecast-lesions.svg')
    plt.close(fig)


def plot_adaptation(folder):
    folder = Path(folder)
    audit = json.loads((folder/'audit.json').read_text())
    assert audit['status'] == 'AUDITED_DEVELOPMENT'
    common = json.loads((folder/'common.json').read_text())
    reports = [json.loads((folder/f'seed{s}/report.json').read_text()) for s in common['seeds']]
    shown = tuple(COLORS)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    total_fig, total_axes = plt.subplots(1, 2, figsize=(12, 4))
    csv_rows = []
    for col, family in enumerate(('markov', 'drift')):
        top, bottom = axes[0, col], axes[1, col]
        for arm in LABELS:
            cells = [r['arms'][arm]['evaluation'][family]['summary'] for r in reports] if arm in reports[0]['arms'] else [common['baselines'][arm][family]['summary']]
            costs = np.array([[c[str(n)]['cost_per_entity'] for n in LEVELS] for c in cells]).mean(0)
            totals = np.array([[c[str(n)]['total_cost_per_entity'] for n in LEVELS] for c in cells]).mean(0)
            faults = np.array([[c[str(n)]['residual_faults'] for n in LEVELS] for c in cells]).mean(0)
            for i, n in enumerate(LEVELS):
                stat = audit['statistics']['families'][family][arm]['by_level'][str(n)]
                csv_rows.append({'family': family, 'arm': arm, 'observed_episodes': n,
                    'cost_per_entity': costs[i], 'total_cost_per_entity': totals[i],
                    'residual_faults_mean_per_128_world_cohort': faults[i],
                    'fraction_oracle_gap_closed': stat['fraction_oracle_gap_closed'],
                    'fraction_ci_low': stat['fraction_95pct'][0] if stat['fraction_95pct'] else '',
                    'fraction_ci_high': stat['fraction_95pct'][1] if stat['fraction_95pct'] else '',
                    'bootstrap_invalid_denominator_rate': stat['fraction_invalid_denominator_rate']})
            if arm not in shown: continue
            style = '--' if arm == 'oracle' else '-'
            kw = {'color': COLORS[arm], 'label': LABELS[arm], 'linestyle': style, 'marker': 'o', 'markersize': 4}
            top.plot(LEVELS, costs, **kw)
            total_axes[col].plot(LEVELS, totals, **kw)
            by_level = audit['statistics']['families'][family][arm]['by_level']
            fractions = [by_level[str(n)]['fraction_oracle_gap_closed'] for n in LEVELS]
            bottom.plot(LEVELS, fractions, **kw)
            if arm in ('gru128', 'mtsk'):
                low = [by_level[str(n)]['fraction_95pct'][0] if by_level[str(n)]['fraction_95pct'] else np.nan for n in LEVELS]
                high = [by_level[str(n)]['fraction_95pct'][1] if by_level[str(n)]['fraction_95pct'] else np.nan for n in LEVELS]
                bottom.fill_between(LEVELS, low, high, color=COLORS[arm], alpha=.1)
        title = 'Matched switching process' if family == 'markov' else 'Unseen smooth drift'
        top.set_title(title); total_axes[col].set_title(title)
        bottom.axhline(0, color='#888888', linewidth=.7)
        bottom.axhline(1, color='#222222', linewidth=.7, linestyle='--')
        for ax in (top, bottom, total_axes[col]):
            ax.set_xticks(LEVELS); ax.set_xlabel('Identical public observation episodes available')
            ax.grid(alpha=.2); ax.set_axisbelow(True)
    axes[0, 0].set_ylabel('Maintenance-probe cost/entity')
    axes[1, 0].set_ylabel('Fraction of stationary-Beta → oracle gap closed')
    total_axes[0].set_ylabel('Total cost/entity including paid observations')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9, bbox_to_anchor=(.5, .025))
    total_fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9, bbox_to_anchor=(.5, .005))
    fig.suptitle('DEV-053B: adaptation at equal information', fontsize=14)
    fig.text(.5, .005, '128 stream clusters/family; five neural initializations reuse streams. Bands: crossed paired world/seed bootstrap; ratio intervals conditional on positive oracle-gap draws. All controls retained in CSV.', ha='center', fontsize=7)
    fig.tight_layout(rect=(0, .13, 1, .95)); total_fig.tight_layout(rect=(0, .2, 1, .98))
    fig.savefig(folder/'learning-curves.png', dpi=170); fig.savefig(folder/'learning-curves.svg')
    total_fig.savefig(folder/'total-cost-curves.png', dpi=170); total_fig.savefig(folder/'total-cost-curves.svg')
    plt.close(fig); plt.close(total_fig)
    with (folder/'learning-curves.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0]); writer.writeheader(); writer.writerows(csv_rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review'); p.add_argument('--adaptation')
    a = p.parse_args()
    if a.review: plot_review(a.review)
    if a.adaptation: plot_adaptation(a.adaptation)
