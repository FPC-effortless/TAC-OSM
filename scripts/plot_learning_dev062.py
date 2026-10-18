"""Plot every retained seed and median validation curves, without refitting."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(study):
    root=Path('results')/(study+'-main');real=study=='real-maint-001';identity=study=='dev0615';arms=sorted({p.parent.name for p in root.glob('group-*/*/report.json')});fig,axes=plt.subplots(1,2,figsize=(12,4.5));colors=plt.get_cmap('tab10')
    keys=('policy','current') if identity else ('Gaussian_NLL','normalized_MSE') if real else ('policy','kl')
    for i,arm in enumerate(arms):
        reports=[json.loads(p.read_text()) for p in sorted(root.glob(f'group-*/{arm}/report.json'))];traces=[r['curves'] for r in reports];x=[r['update'] for r in traces[0]]
        assert all([r['update'] for r in t]==x for t in traces)
        for axis,key in zip(axes,keys,strict=True):
            values=np.array([[r['validation'][key] for r in t] for t in traces]);axis.plot(x,values.T,color=colors(i),alpha=.13,lw=.6);axis.plot(x,np.median(values,axis=0),color=colors(i),lw=2,label=arm)
    for axis,key in zip(axes,keys,strict=True):axis.set_xlabel('Optimizer updates');axis.set_ylabel('Validation '+key);axis.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.suptitle(study.upper()+': thin lines=all20 seeds; bold=median; validation only');fig.tight_layout();fig.savefig(Path('research')/study/'learning-curves.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',choices=['dev0615','dev062','dev062-r1','real-maint-001'],required=True);main(p.parse_args().study)
