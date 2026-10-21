"""Render registered cross-fit endpoints without selection or refitting."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run():
    src=Path('results/me031');dest=Path('research/mathematical-engineering/me031')
    r=json.loads((src/'results.json').read_text());assert r['complete']
    e=r['evaluation'];arms=('prior','base','common_residual','state_residual')
    means={a:{k:float(np.mean([row[a][k] for row in e['pooled_per_seed'].values()]))
              for k in ('Brier','log_loss','ROC_AUC','ECE10')} for a in arms}
    costs={}
    for arm in ('common_residual','state_residual'):
        rows=[]
        for cell in r['cells']:
            pr=next(p for p in cell['probes'] if p['arm']==arm)
            rows.append(cell['shared_charged_cpu_seconds']+pr['charged_cpu_seconds'])
        costs[arm]={'mean_charged_pipeline_cpu_seconds':float(np.mean(rows)),
                    'per_cell_charged_cpu_seconds':rows}
    summary={'mean_metrics':means,'charged_pipeline_resources':costs,
             'state_minus_common_Brier':e['state_minus_common_mean_Brier'],
             'state_minus_common_per_seed':e['state_minus_common_per_seed_Brier'],
             'user_cluster_CI95':e['user_cluster_CI95'],'gates':e['gates']}
    (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['| Predictor | Brier ↓ | Log loss ↓ | AUC ↑ | ECE10 ↓ |',
           '|---|---:|---:|---:|---:|']
    for a in arms:lines.append('| '+a+' | '+' | '.join(f'{means[a][k]:.9f}' for k in ('Brier','log_loss','ROC_AUC','ECE10'))+' |')
    (dest/'metrics-table.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    axes[0].bar(range(4),[means[a]['Brier'] for a in arms])
    axes[0].set_xticks(range(4),['prior','frozen base','common residual','state residual'],rotation=20,ha='right')
    axes[0].set_ylabel('Mean Brier (lower better)')
    axes[1].bar(['14001','14002','14003'],e['state_minus_common_per_seed_Brier'])
    axes[1].axhline(0,color='black',linewidth=.8)
    axes[1].axhline(-.0005,color='red',linestyle='--',label='registered mean-gain threshold')
    axes[1].set_ylabel('State minus common Brier');axes[1].set_xlabel('Training seed')
    axes[1].legend(fontsize=7)
    fig.suptitle('ME031: user-separated evaluation within previously used KuaiRec FIT')
    fig.tight_layout();fig.savefig(dest/'crossfit-results.png',dpi=160);fig.savefig(dest/'crossfit-results.svg');plt.close(fig)
    for name in ('results.json','roles.json','replay.json'):
        if (src/name).exists():(dest/name).write_bytes((src/name).read_bytes())
    print(json.dumps(summary,indent=2))


if __name__=='__main__':run()
