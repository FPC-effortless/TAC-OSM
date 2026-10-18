"""All paired seeds on the correction's fresh worlds; no cross-namespace pooling."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run(statistics,directory):
    result=json.loads(Path(statistics).read_text());root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    for metric,label in [('completion','Completion (%)'),('economic_cost','Economic cost/world')]:
        fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
        for ax,(arm,record) in zip(axes,result['arms'].items()):
            values=record['per_seed']
            for seed in sorted(values['original'],key=int):
                pair=[values[c][seed][metric] for c in ('original','corrected')]
                if metric=='completion':pair=[100*v for v in pair]
                ax.plot([0,1],pair,marker='o',alpha=.7,label=seed)
            if metric=='completion':ax.axhline(95,c='gray',ls='--');ax.set_ylim(0,101)
            else:ax.axhline(result['public_reference']['economic_cost'],c='black',ls='--',label='Public rule')
            ax.set_xticks([0,1],['Original arithmetic','Exact forward gate']);ax.set_ylabel(label);ax.set_title(arm)
        axes[-1].legend(fontsize=8,ncol=2);fig.suptitle('DEV-061-R1 — ten paired seeds, fresh shared clean worlds')
        fig.savefig(root/f'paired-{metric}.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--statistics',required=True);a.add_argument('--directory',required=True);args=a.parse_args();run(args.statistics,args.directory)
