"""Standalone per-training-seed figures; repeated worlds are not replications."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(study):
    root=Path('research')/study;data=json.loads((root/'statistics.json').read_text());real=study=='real-maint-001';fig,axes=plt.subplots(1,2,figsize=(12,4.2));colors=plt.get_cmap('tab10');lines=[]
    for i,(arm,values) in enumerate(data['arms'].items()):
        rows=[v for _,v in sorted(values['per_seed'].items(),key=lambda r:int(r[0]))];x=np.arange(1,len(rows)+1);color=colors(i)
        left=[v['normalized_MSE'] if real else 100*v['completion'] for v in rows];right=[v['anomalies']['outside_reported_windows_alert_fraction'] if real else v['cost'] for v in rows]
        line,=axes[0].plot(x,left,'o-',markersize=3,alpha=.85,color=color,label=arm);axes[1].plot(x,right,'o-',markersize=3,alpha=.85,color=color);lines.append(line)
    if real:
        axes[0].axhline(data['baselines']['persistence_normalized_MSE'],color='black',ls='--',label='persistence');axes[0].set_yscale('log');axes[0].set_ylabel('Held-out normalized sensor MSE (log)');axes[1].set_ylabel('Alerts outside reported fault windows\n(health unknown; this is not false-positive rate)')
    else:
        axes[0].axhline(95,color='black',ls='--',lw=1);axes[0].set_ylim(0,102);axes[0].set_ylabel('Held-out completion (%)');axes[1].set_ylabel('Mean total synthetic economic cost/world')
        reference=data['reference']['cost'] if study=='dev0615' else data['references']['public']['cost'];axes[1].axhline(reference,color='black',ls='--',label='public reference')
    for axis in axes:axis.set_xlabel('Training seed index');axis.set_xticks([1,5,10,15,20]);axis.grid(alpha=.2)
    fig.suptitle(study.upper()+': all twenty training initializations; shared held-out data',fontsize=12);fig.legend(handles=lines,loc='lower center',ncol=4,frameon=False,fontsize=8);fig.tight_layout(rect=(0,.16,1,.95));fig.savefig(root/'per-seed.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',choices=['dev0615','dev062','real-maint-001'],required=True);a=p.parse_args();plot(a.study)
