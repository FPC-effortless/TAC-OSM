"""Expose every fresh seed's curves and heldout outcome distribution."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(directory, statistics, destination):
    root, out = Path(directory), Path(destination)
    assert json.loads((root / 'summary.json').read_text())['status'] == 'COMPLETE'
    data = json.loads(Path(statistics).read_text())
    arms = list(data['arms'])
    fig, axes = plt.subplots(len(arms), 3, figsize=(15, 2.5 * len(arms)))
    for i, arm in enumerate(arms):
        for seed in range(93401, 93421):
            report = json.loads((root / f'group-{seed}' / arm / 'report.json').read_text())
            curves = report['curves']
            for j, (split, key) in enumerate([('train', 'kl'), ('validation', 'kl'), ('validation', 'policy')]):
                axes[i, j].plot([v['update'] for v in curves],
                                [max(v[split][key], 1e-8) for v in curves],
                                alpha=.5, linewidth=.8)
        for j, title in enumerate(['Minibatch joint KL', 'Validation joint KL', 'Validation policy NLL']):
            ax = axes[i, j]
            ax.set_title(arm + ': ' + title, fontsize=9)
            ax.set_yscale('log')
            ax.axvline(1024, linestyle='--', color='black', alpha=.6)
            ax.set_xlabel('Updates; dashed line ends belief warmup')
            ax.grid(alpha=.2)
    fig.suptitle('DEV063: all twenty fresh seeds per arm. Validation curves use the fixed 64-record diagnostic.', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .98))
    fig.savefig(out / 'learning-curves.png', dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(17, 7))
    primary = data['arms'][data['primary']]['per_seed']
    completion = np.array([[v['completion'] for v in primary[str(s)]['strata']]
                           for s in range(93401, 93421)])
    image = axes[0].imshow(completion, aspect='auto', vmin=0, vmax=1, cmap='viridis')
    axes[0].set_xticks(range(4), ['H32', 'H128', 'H512', 'H1024'])
    axes[0].set_yticks(range(20), range(93401, 93421))
    axes[0].set_title('Primary completion by seed and history')
    fig.colorbar(image, ax=axes[0], label='Completion fraction')
    for i, arm in enumerate(arms):
        rows = data['arms'][arm]['per_seed'].values()
        axes[1].scatter([i] * 20, [max(v['fixed']['p99_KL'], 1e-9) for v in rows], s=16, alpha=.6)
        axes[2].scatter([i] * 20, [v['cost'] for v in rows], s=16, alpha=.6)
    axes[1].axhline(.1, color='black', linestyle='--')
    axes[1].set_yscale('log')
    axes[1].set_title('Every seed: fixed-query p99 joint KL')
    axes[2].set_title('Every seed: mean primary world cost')
    for ax in axes[1:]:
        ax.set_xticks(range(len(arms)), arms, rotation=70, fontsize=8)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Shared physical worlds across seeds/histories; seed replicates are not independent real machines.')
    fig.tight_layout()
    fig.savefig(out / 'per-seed.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    parser.add_argument('--statistics', required=True)
    parser.add_argument('--destination', required=True)
    args = parser.parse_args()
    plot(args.directory, args.statistics, args.destination)
