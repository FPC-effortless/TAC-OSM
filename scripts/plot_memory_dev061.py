"""Standalone research figures; all seeds remain visible."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def run(directory,statistics,output):
    root=Path(directory);target=Path(output);target.mkdir(parents=True,exist_ok=True);s=json.loads(Path(statistics).read_text());arms=list(s['arms']);seeds=list(range(9101,9111))
    fig,axes=plt.subplots(2,4,figsize=(16,7),sharex=True,constrained_layout=True)
    for ax,arm in zip(axes.flat,arms):
        for seed in seeds:
            r=json.loads((root/f'{arm}-{seed}/report.json').read_text());x=[v['update'] for v in r['curves']]
            ax.plot(x,[v['validation']['loss'] for v in r['curves']],alpha=.7,lw=1,label=str(seed))
        ax.axvspan(0,256,color='gray',alpha=.12);ax.set_title(arm);ax.set_ylabel('Validation combined loss');ax.set_xlabel('Update')
    axes.flat[-1].axis('off');axes.flat[-1].text(0,.8,'Ten fresh seeds per arm\nShading: belief/read warmup\nActor-only MLP objective differs\nFixed updates do not prove convergence')
    fig.savefig(target/'learning-curves.png',dpi=160);plt.close(fig)
    values=np.array([[s['arms'][a]['per_seed'][str(seed)]['completion']*100 for a in arms] for seed in seeds])
    fig,ax=plt.subplots(figsize=(11,5),constrained_layout=True);im=ax.imshow(values,vmin=0,vmax=100,cmap='viridis',aspect='auto')
    for i in range(10):
        for j in range(7):ax.text(j,i,f'{values[i,j]:.1f}',ha='center',va='center',color='black' if values[i,j]>55 else 'white',fontsize=8)
    ax.set_xticks(range(7),arms,rotation=25,ha='right');ax.set_yticks(range(10),seeds);ax.set_ylabel('Fresh training seed');ax.set_title('Clean completion (%) — shared 384 worlds per seed/arm')
    fig.colorbar(im,ax=ax,label='Completion (%)');fig.savefig(target/'per-seed-completion.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    for j,arm in enumerate(arms):
        costs=[s['arms'][arm]['per_seed'][str(seed)]['economic_cost'] for seed in seeds];ax.scatter(np.full(10,j),costs,alpha=.65,s=25)
    ax.axhline(s['references']['public']['economic_cost'],c='black',ls='--',label='Public rule; same clean worlds')
    ax.set_xticks(range(7),arms,rotation=25,ha='right');ax.set_ylabel('Economic cost/world (logical writes separate)');ax.legend();fig.savefig(target/'per-seed-cost.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--directory',required=True);a.add_argument('--statistics',required=True);a.add_argument('--output',required=True);args=a.parse_args();run(args.directory,args.statistics,args.output)
