"""Render only the registered ME033 endpoints and predeclared contrasts."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run():
    src=Path('results/me033');dst=Path('research/mathematical-engineering/me033')
    r=json.loads((src/'results.json').read_text());assert r['complete']
    e=r['evaluation'];arms=('teacher','common_adapt','history_frozen','history_adapt','query_frozen','query_adapt','native_adapt')
    means={a:{k:float(np.mean([row[a][k] for row in e['pooled_per_seed'].values()]))
              for k in ('Brier','log_loss','ROC_AUC','ECE10')} for a in arms}
    costs={}
    for a in arms[1:]:
        ss=[];full=[]
        for c in r['cells']:
            st=next(s for s in c['students'] if s['arm']==a)
            ss.append(st['charged_cpu_seconds']);full.append(c['shared_charged_cpu_seconds']+st['charged_cpu_seconds'])
        costs[a]={'mean_S_CPU_seconds':float(np.mean(ss)),'mean_charged_pipeline_CPU_seconds':float(np.mean(full)),
                  'per_cell_charged_pipeline_CPU_seconds':full}
    summary={'mean_metrics':means,'costs':costs,'contrasts':e['contrasts'],
        'history_gates':e['history_gates'],'native_gates':e['native_gates'],
        'adaptation_specific_interaction':e['adaptation_specific_interaction'],
        'new_physical_CPU_seconds':r['new_physical_CPU_seconds'],'new_physical_wall_seconds':r['new_physical_wall_seconds']}
    (dst/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['| Predictor | Brier ↓ | Log loss ↓ | AUC ↑ | ECE10 ↓ |','|---|---:|---:|---:|---:|']
    for a in arms:lines.append('| '+a+' | '+' | '.join(f'{means[a][k]:.9f}' for k in ('Brier','log_loss','ROC_AUC','ECE10'))+' |')
    (dst/'metrics-table.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for path in ('history','query'):
        vals=[means[path+'_'+mode]['Brier'] for mode in ('frozen','adapt')]
        axes[0].plot(['Frozen embedding','S-adapted embedding'],vals,'o-',label=path)
    axes[0].axhline(means['common_adapt']['Brier'],ls='--',label='common-only S adaptation',color='gray')
    axes[0].set_ylabel('Three-seed mean Brier (lower better)');axes[0].legend(fontsize=8)
    names=('history_vs_query_adapt','history_adaptation','interaction','native_vs_history')
    p=np.array([e['contrasts'][n]['mean_Brier'] for n in names])
    low=np.array([e['contrasts'][n]['user_cluster_CI95'][0] for n in names])
    high=np.array([e['contrasts'][n]['user_cluster_CI95'][1] for n in names])
    axes[1].errorbar(range(4),p,yerr=np.stack((p-low,high-p)),fmt='o',capsize=4)
    axes[1].axhline(0,color='black',lw=.8);axes[1].set_xticks(range(4),names,rotation=25,ha='right')
    axes[1].set_ylabel('Paired Brier contrast; conditional user intervals')
    fig.suptitle('ME033: history/query identity pathway x embedding adaptation; reused FIT')
    fig.tight_layout();fig.savefig(dst/'semantic-factorial-results.png',dpi=160);fig.savefig(dst/'semantic-factorial-results.svg');plt.close(fig)
    for n in ('results.json','roles.json','replay.json'):
        if (src/n).exists():(dst/n).write_bytes((src/n).read_bytes())
    print(json.dumps(summary,indent=2))


if __name__=='__main__':run()
