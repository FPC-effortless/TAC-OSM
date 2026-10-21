"""Render ME032 frozen endpoints without selecting models or refitting."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run():
    source=Path('results/me032');dest=Path('research/mathematical-engineering/me032')
    r=json.loads((source/'results.json').read_text());assert r['complete']
    e=r['evaluation'];arms=('prior','teacher','conditional_native','joint_native','conditional_pool')
    means={a:{k:float(np.mean([seed[a][k] for seed in e['pooled_per_seed'].values()]))
              for k in ('Brier','log_loss','ROC_AUC','ECE10')} for a in arms}
    costs={}
    for a in arms[2:]:
        rows=[];s_only=[]
        for cell in r['cells']:
            student=next(s for s in cell['students'] if s['arm']==a)
            rows.append(cell['shared_charged_cpu_seconds']+student['charged_cpu_seconds'])
            s_only.append(student['charged_cpu_seconds'])
        costs[a]={'mean_pipeline_charged_cpu_seconds':float(np.mean(rows)),
                  'mean_S_charged_cpu_seconds':float(np.mean(s_only)),
                  'per_cell_pipeline_charged_cpu_seconds':rows}
    summary={'mean_metrics':means,'charged_resources':costs,'contrasts':e['contrasts'],
             'gates':e['gates'],'new_study_physical_CPU_seconds':r['new_study_physical_CPU_seconds'],
             'new_study_wall_seconds':r['new_study_wall_seconds']}
    (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['| Predictor | Brier ↓ | Log loss ↓ | AUC ↑ | ECE10 ↓ |','|---|---:|---:|---:|---:|']
    for a in arms:lines.append('| '+a+' | '+' | '.join(f'{means[a][k]:.9f}' for k in ('Brier','log_loss','ROC_AUC','ECE10'))+' |')
    (dest/'metrics-table.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].bar(range(5),[means[a]['Brier'] for a in arms]);axes[0].set_xticks(range(5),arms,rotation=30,ha='right')
    axes[0].set_ylabel('Mean FIT-user-held-out Brier (lower better)')
    controls=('joint_native','conditional_pool','teacher')
    points=np.array([e['contrasts'][c]['mean_Brier'] for c in controls])
    lo=np.array([e['contrasts'][c]['user_cluster_CI95'][0] for c in controls])
    hi=np.array([e['contrasts'][c]['user_cluster_CI95'][1] for c in controls])
    axes[1].errorbar(range(3),points,yerr=np.stack((points-lo,hi-points)),fmt='o',capsize=5)
    axes[1].axhline(0,color='black',linewidth=.8);axes[1].axhline(-.0005,color='red',linestyle='--')
    axes[1].set_xticks(range(3),controls,rotation=20,ha='right');axes[1].set_ylabel('Conditional native minus control Brier')
    fig.suptitle('ME032 centered memory training: reused FIT, fixed3seeds; conditional user-cluster intervals')
    fig.tight_layout();fig.savefig(dest/'conditional-training-results.png',dpi=160);fig.savefig(dest/'conditional-training-results.svg');plt.close(fig)
    for name in ('results.json','roles.json','replay.json'):
        if (source/name).exists():(dest/name).write_bytes((source/name).read_bytes())
    print(json.dumps(summary,indent=2))


if __name__=='__main__':run()
