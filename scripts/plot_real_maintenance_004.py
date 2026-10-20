"""Publication-style pilot results; no invented test or20seed evidence."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run():
    root = Path('results/real004-main'); destination = Path('research/real-maint-004')
    refs = json.loads((root/'references.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for arm, color in [('prestate', '#3883bd'), ('repair', '#cc7755')]:
        for index in range(4):
            reports = [json.loads(p.read_text()) for p in sorted(root.glob(f'pilot-{arm}-{index}-*/report.json'))]
            values = [r['validation_MAE_seconds'] for r in reports]
            x = index+(.12 if arm == 'repair' else -.12)
            axes[0].scatter([x]*len(values), values, color=color, alpha=.7, label=arm if index == 0 else None)
            axes[0].plot([x-.06, x+.06], [np.mean(values)]*2, color=color, linewidth=3)
    axes[0].axhline(refs['prestatus']['validation_MAE_seconds'], color='#333333', label='locked prestatus median')
    axes[0].axhline(.95*refs['prestatus']['validation_MAE_seconds'], color='#555555', linestyle='--', label='entry threshold')
    axes[0].set(xticks=range(4), xlabel='Registered recipe index', ylabel='Validation MAE (seconds per20L)', title='Three pilot initializations per recipe')
    axes[0].legend(fontsize=8)
    for arm, color in [('prestate', '#3883bd'), ('repair', '#cc7755')]:
        for i, path in enumerate(sorted(root.glob(f'pilot-{arm}-3-*/report.json'))):
            curve = json.loads(path.read_text())['curve']
            axes[1].plot([c['update'] for c in curve], [c['validation_MAE_seconds'] for c in curve],
                color=color, alpha=.5, label=arm if i == 0 else None)
    axes[1].axhline(refs['prestatus']['validation_MAE_seconds'], color='#333333')
    axes[1].set(xlabel='Update', ylabel='Validation MAE (seconds per20L)', title='Validation-selected recipe; final EMA only')
    axes[1].legend(fontsize=8)
    fig.suptitle('REAL-MAINT-004: validation rejection; test cohort remains unopened', fontsize=12)
    fig.savefig(destination/'pilot-validation.png', dpi=180); plt.close(fig)


if __name__ == '__main__': run()
