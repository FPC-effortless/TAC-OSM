"""Per-seed reliability and actual learning curves; test worlds stay closed."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run():
    root = Path('results/dev068-main'); destination = Path('research/dev068-public-outcome')
    arms = [('tac_outcome', 'TAC'), ('gru_outcome', 'GRU'), ('no_casm_outcome', 'No CASM'), ('tac_teacher_belief', 'Teacher control')]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    for column, (arm, label) in enumerate(arms):
        reports = [json.loads(p.read_text()) for p in sorted(root.glob(f'group-*/{arm}-report.json'))]
        completion = [min(c['completion'] for c in r['validation']) for r in reports]
        passing = [all(c['completion'] >= .95 and c['faults'] == 0 for c in r['validation']) for r in reports]
        axes[0].scatter(column+np.linspace(-.12, .12, len(reports)), completion,
            c=['#267f7b' if value else '#bc5a4d' for value in passing], s=32, alpha=.8)
        axes[1].bar(column, sum(passing), color='#52899b')
    axes[0].axhline(.95, color='#333333', linestyle='--')
    axes[0].set(xticks=range(4), xticklabels=[a[1] for a in arms], ylim=(.6,1.02),
        ylabel='Minimum validation-stratum completion', title='20 parent initializations; green also has zero faults')
    axes[1].axhline(18, color='#333333', linestyle='--', label='Registered entry requirement')
    axes[1].set(xticks=range(4), xticklabels=[a[1] for a in arms], ylim=(0,20),
        ylabel='Passing training seeds /20', title='Completion AND fault guard')
    axes[1].legend(fontsize=8)
    fig.suptitle('DEV068: all arms fail seed-reliability entry; sealed test worlds remain unopened')
    fig.savefig(destination/'per-seed-validation.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for arm, label in arms:
        reports = [json.loads(p.read_text()) for p in sorted(root.glob(f'group-*/{arm}-report.json'))]
        for axis, stage, key, title in zip(axes, ('encoder','world','value'),
                ('validation_measurement_NLL','NLL','Bellman_MSE'),
                ('Held-out public measurement NLL','Factual world training NLL','Sampled training Bellman MSE'), strict=True):
            curves = [[row for row in r['curves'] if row['stage'] == stage] for r in reports]
            updates = [c['update'] for c in curves[0]]
            values = np.array([[c[key] for c in curve] for curve in curves])
            line, = axis.plot(updates, values.mean(0), label=label)
            axis.fill_between(updates, np.quantile(values,.1,axis=0), np.quantile(values,.9,axis=0), color=line.get_color(), alpha=.1)
            axis.set(xlabel='Update', title=title)
    axes[0].legend(fontsize=8)
    fig.suptitle('Mean and10–90% range over20initializations; fixed budgets, no convergence claim')
    fig.savefig(destination/'learning-curves.png', dpi=180); plt.close(fig)


if __name__ == '__main__': run()
