"""Render fixed endpoint results and all registered learning curves."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    src=Path('results/me030');dest=Path('research/mathematical-engineering/me030')
    r=json.loads((src/'results.json').read_text())
    metrics={}
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for arm in r['parameters']:
        rows=[x for x in r['rows'] if x['arm']==arm]
        metrics[arm]={k:float(np.mean([x['metrics'][k] for x in rows]))
                      for k in ('Brier','log_loss','ROC_AUC','ECE10')}
        metrics[arm].update(parameters=r['parameters'][arm],
            CPU_seconds=float(np.mean([x['charged_train_cpu_seconds'] for x in rows])),
            inference_B128_ms=float(np.mean([x['inference_B128_median_seconds'] for x in rows]))*1000)
        steps=[x['step'] for x in rows[0]['curve']]
        mean=np.mean([[x['metrics']['Brier'] for x in row['curve']] for row in rows],axis=0)
        cpu=np.mean([[x['charged_train_cpu_seconds'] for x in row['curve']] for row in rows],axis=0)
        for ax,x in zip(axes,(steps,cpu)):ax.plot(x,mean,marker='.',label=arm)
    axes[0].set_xlabel('Optimizer updates');axes[1].set_xlabel('Charged training CPU seconds')
    for ax in axes:ax.set_ylabel('Historical DEV Brier (lower better)');ax.grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.suptitle('ME030: three-seed means, reused KuaiRec DEV; fixed300 primary')
    fig.tight_layout();fig.savefig(dest/'learning-curves.png',dpi=160);fig.savefig(dest/'learning-curves.svg');plt.close(fig)
    (dest/'summary.json').write_text(json.dumps({'mean_metrics':metrics,'gates':r['gates'],'contrasts':r['contrasts']},indent=2)+'\n')
    for f in ('results.json','replay.json','artifact-audit.json'):
        (dest/f).write_bytes((src/f).read_bytes())


if __name__=='__main__':main()
