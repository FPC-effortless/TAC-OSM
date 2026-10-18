"""Standalone figures for all registered retrieval seeds; no model selection."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('results/dev060-memory');out=Path('research/dev060');s=json.loads((root/'summary.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(10,4))
lengths=[8,16,32,64]
for arm in ('content','temporal'):
    values=np.array([[s['arms'][arm][str(seed)][str(length)]['exact'] for length in lengths] for seed in range(80601,80611)])*100
    axes[0].errorbar(lengths,values.mean(0),yerr=values.std(0,ddof=1),marker='o',capsize=3,label=arm)
    stale=np.array([[s['arms'][arm][str(seed)][str(length)]['stale_same_identity'] for length in lengths] for seed in range(80601,80611)])*100
    axes[1].plot(lengths,stale.mean(0),marker='o',label=arm)
axes[0].axhline(100,color='gray',linestyle='--',label='Exact latest-record reference')
axes[0].set_ylabel('Exact retrieval (%)');axes[1].set_ylabel('Stale same-identity choices (%)')
for ax in axes:ax.set_xscale('log',base=2);ax.set_xticks(lengths,lengths);ax.set_xlabel('Records per query');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Ten seeds per arm; same public time features; no established temporal advantage')
fig.tight_layout();fig.savefig(out/'retrieval-results.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4),sharex=True)
for arm in ('content','temporal'):
    curves=[json.loads((root/f'{arm}-{seed}/fit.json').read_text())['curves'] for seed in range(80601,80611)]
    x=[r['update'] for r in curves[0]]
    for c in curves:
        axes[0].plot(x,[r['loss'] for r in c],alpha=.25)
        axes[1].plot(x,[np.mean(r['validation']) for r in c],alpha=.25)
    axes[0].plot(x,np.mean([[r['loss'] for r in c] for c in curves],0),label=arm)
    axes[1].plot(x,np.mean([[np.mean(r['validation']) for r in c] for c in curves],0),label=arm)
for ax in axes:ax.set_xlabel('Training updates');ax.grid(alpha=.2);ax.legend()
axes[0].set_ylabel('Training cross entropy');axes[1].set_ylabel('Validation exact retrieval')
fig.suptitle('All twenty isolated retrieval fits; convergence is not established')
fig.tight_layout();fig.savefig(out/'retrieval-learning-curves.png',dpi=160)
