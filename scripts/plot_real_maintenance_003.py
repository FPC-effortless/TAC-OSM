"""All seeds/cells and all first-day rooms, with correct published Fahrenheit units."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from run_real_maintenance_003 import make
from data_real_maintenance_003 import ROOMS


@torch.no_grad()
def run(directory,output):
    torch.set_num_threads(1);root=Path(directory);dest=Path(output)
    stats=json.loads((dest/'statistics.json').read_text());assert stats['status']=='COMPLETE'
    specs=json.loads((root/'specs.json').read_text());candidate=stats['candidate']
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for ax,arm in zip(axes.flat,specs,strict=True):
        reports=[json.loads(p.read_text()) for p in root.glob(f'group-*/{arm}/report.json')]
        assert len(reports)==20
        for report in reports:
            ax.plot([c['update'] for c in report['curves']],
                [c['validation']['mean_relative_MAE'] for c in report['curves']],alpha=.25)
        ax.axhline(1,color='black',linestyle='--');ax.set_title(arm)
        ax.set_xlabel('Update');ax.set_ylabel('Validation mean relative MAE')
    fig.suptitle('REAL-MAINT-003 — all 20 seeds; final EMA only')
    fig.tight_layout();fig.savefig(dest/'learning-curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,sharey=True)
    for ax,(arm,values) in zip(axes.flat,stats['arms'].items(),strict=True):
        rows=list(values['per_seed'].values())
        ratios=np.mean([r['relative_MAE'] for r in rows],axis=0)
        display=ax.imshow(ratios,vmin=.5,vmax=1.5,cmap='coolwarm',aspect='auto')
        ax.set_title(arm);ax.set_yticks([0,1],['30 min','60 min'])
        ax.set_xticks(range(10),[n.split('_')[2] for n in ROOMS])
        for i in range(2):
            for j in range(10):ax.text(j,i,f'{ratios[i,j]:.2f}',ha='center',va='center',fontsize=8)
    fig.colorbar(display,ax=axes.ravel().tolist(),label='MAE / persistence MAE',shrink=.6)
    fig.suptitle('All 10 rooms and 2 horizons; average across 20 seeds; 2019 measured experiments')
    fig.savefig(dest/'per-channel.png',dpi=160,bbox_inches='tight');plt.close(fig)
    data=torch.load(root/'_datasets/test.pt',weights_only=True)
    normalization=torch.load(root/'_datasets/normalization.pt',weights_only=True)
    report=json.loads((root/'group-97301'/candidate/'report.json').read_text())
    model=make(report['spec'],97301,normalization).double().eval()
    model.load_state_dict(torch.load(root/'group-97301'/candidate/'agent.pt',weights_only=True))
    indices=(data['day']==data['day'].min()).nonzero(as_tuple=True)[0]
    x=data['x'][indices];prediction=model(x,data['minutes'][indices]);truth=data['y'][indices]
    hours=(data['origin_seconds'][indices].numpy()%86400)/3600+1
    fig,axes=plt.subplots(5,2,figsize=(13,12))
    for room,(ax,name) in enumerate(zip(axes.flat,ROOMS,strict=True)):
        ax.plot(hours,truth[:,1,room],label='Measured',color='black')
        ax.plot(hours,prediction[:,1,room],label='MLP RevIN, seed 97301')
        ax.plot(hours,x[:,-1,room],label='Persistence',alpha=.6)
        ax.set_title(name);ax.set_ylabel('Temperature (°F)');ax.set_xlabel('Target logger hour')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('First held-out CSV day, all rooms, 60 min horizon, predeclared first seed')
    fig.tight_layout();fig.savefig(dest/'first-day-forecasts.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.directory,args.output)
