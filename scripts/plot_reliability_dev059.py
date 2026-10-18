"""Export all ten seed curves and endpoints; no checkpoint selection."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path('results/dev059-main');out=Path('research/dev059');out.mkdir(exist_ok=True)
stats=json.loads(Path('results/dev059-statistics.json').read_text())
arms=list(stats['arms']);fig,axes=plt.subplots(2,5,figsize=(17,7),sharex='row',sharey='row')
for col,arm in enumerate(arms):
    curves=[json.loads((root/f'{arm}-{s}/progress.json').read_text())['curves'] for s in stats['seeds']]
    x=[c['update'] for c in curves[0]]
    for row,key in enumerate(('policy_ce','progress_accuracy')):
        values=np.array([[c['validation'][key] for c in curve] for curve in curves])
        for y in values:axes[row,col].plot(x,y,alpha=.3,lw=1)
        axes[row,col].plot(x,values.mean(0),color='black',lw=2,label='Ten-seed mean')
        axes[row,col].grid(alpha=.2)
    axes[0,col].set_title(arm);axes[1,col].set_xlabel('Controller updates')
axes[0,0].set_ylabel('Validation policy cross entropy');axes[1,0].set_ylabel('Validation teacher-action accuracy')
axes[0,0].legend();fig.suptitle('DEV-059: every initialization; snapshot imitation curves, not closed-loop convergence')
fig.tight_layout();fig.savefig(out/'learning-curves.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,5))
for i,arm in enumerate(arms):
    y=np.array(stats['arms'][arm]['per_seed']['completion'])*100
    ax.scatter(np.full(10,i)+np.linspace(-.14,.14,10),y,s=32)
    ax.plot([i-.2,i+.2],[y.mean()]*2,color='black',lw=2)
ax.set_xticks(range(5),arms);ax.set_ylabel('Closed-loop completion (%)');ax.set_ylim(0,103)
ax.set_title('Ten fresh seeds per arm on the same 576 held-out worlds');ax.grid(axis='y',alpha=.2)
fig.tight_layout();fig.savefig(out/'seed-completions.png',dpi=160)
