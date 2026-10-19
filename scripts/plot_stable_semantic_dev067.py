"""Display every fresh seed and both loss streams, without selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def run(directory, output):
    root = Path(directory); dest = Path(output); dest.mkdir(parents=True, exist_ok=True)
    summary = json.loads((root/'summary.json').read_text())
    assert summary['status'] == 'COMPLETE'
    arms = json.loads((root/'specs.json').read_text())
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, arm in zip(axes.flat, arms, strict=True):
        reports = [json.loads(p.read_text()) for p in root.glob(f'group-*/{arm}/report.json')]
        assert len(reports) == 20
        for report in reports:
            curve = report['curves']
            ax.plot([c['update'] for c in curve], [c['validation_KL'] for c in curve], alpha=.25)
        ax.set_title(arm.replace('_', ' ')); ax.set_yscale('log')
        ax.set_ylabel('Validation128 posterior KL'); ax.set_xlabel('Encoder update')
    fig.suptitle('DEV067 — all seeds; four TAC panels share the same encoders')
    fig.tight_layout(); fig.savefig(dest/'learning-curves.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, arm in zip(axes.flat, arms, strict=True):
        for path in root.glob(f'group-*/{arm}/report.json'):
            curve = json.loads(path.read_text())['actor_curves']
            ax.plot([c['actor_update'] for c in curve],
                    [c['validation_predicted']['policy'] for c in curve], alpha=.25)
        ax.set_title(arm.replace('_', ' ')); ax.set_ylabel('Validation128 policy CE')
        ax.set_xlabel('Actor update'); ax.set_yscale('log')
    fig.suptitle('DEV067 — all actor seeds; no best-curve selection')
    fig.tight_layout(); fig.savefig(dest/'actor-curves.png', dpi=160); plt.close(fig)
    statistics = json.loads((dest/'statistics.json').read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for index, (arm, values) in enumerate(statistics['arms'].items()):
        rows = list(values['per_seed'].values())
        for ax, metric in zip(axes, ('completion', 'p99_KL', 'cost'), strict=True):
            y = [r['fixed']['p99_KL'] if metric == 'p99_KL' else r[metric] for r in rows]
            ax.scatter(np.full(len(y), index), y, alpha=.5, s=12)
            ax.set_ylabel(metric)
            ax.set_xticks(range(len(arms)), [a.replace('_', '\n') for a in arms], fontsize=7)
    axes[0].axhline(.95, color='black', linestyle='--')
    axes[1].axhline(.1, color='black', linestyle='--')
    fig.tight_layout(); fig.savefig(dest/'per-seed.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--directory', required=True)
    parser.add_argument('--output', required=True); args = parser.parse_args()
    run(args.directory, args.output)
