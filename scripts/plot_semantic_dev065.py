"""Display all seeds and both training phases; no checkpoint selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def run(directory, output):
    root = Path(directory); dest = Path(output); dest.mkdir(parents=True, exist_ok=True)
    assert json.loads((root/'summary.json').read_text())['status'] == 'COMPLETE'
    arms = json.loads((root/'specs.json').read_text())
    fig, axs = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, arm in zip(axs.flat, arms, strict=True):
        curves = [json.loads(p.read_text())['curves'] for p in root.glob(f'group-*/{arm}/report.json')]
        assert len(curves) == 20
        for curve in curves:
            ax.plot([c['update'] for c in curve], [c['validation']['kl'] for c in curve], alpha=.25)
        ax.axvline(1024, color='black', linestyle='--', linewidth=.8)
        ax.set_title(arm.replace('_', ' ')); ax.set_yscale('log'); ax.set_ylabel('Validation public-posterior KL'); ax.set_xlabel('Update')
    fig.suptitle('DEV065 — every seed, belief-only → joint phase; validation64 display only')
    fig.tight_layout(); fig.savefig(dest/'learning-curves.png', dpi=160); plt.close(fig)
    statistics = json.loads((dest/'statistics.json').read_text())
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for index, (arm, values) in enumerate(statistics['arms'].items()):
        rows = list(values['per_seed'].values())
        for ax, metric in zip(axs, ('completion', 'p99_KL', 'cost'), strict=True):
            y = [r['fixed']['p99_KL'] if metric == 'p99_KL' else r[metric] for r in rows]
            ax.scatter(np.full(len(y), index), y, alpha=.5, s=12)
            ax.set_ylabel(metric); ax.set_xticks(range(len(arms)), [a.replace('_', '\n') for a in arms], fontsize=7)
    axs[0].axhline(.95, color='black', linestyle='--'); axs[1].axhline(.1, color='black', linestyle='--')
    fig.tight_layout(); fig.savefig(dest/'per-seed.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--directory', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); run(a.directory, a.output)
