"""Render registered ME034 outcomes without selecting or fitting anything."""
import json
import shutil
from pathlib import Path
import numpy as np
from scripts.me034_stable_research import ROOT,ARMS,SEEDS,write_json,file_sha


def run():
    out=ROOT/'results/me034';report=ROOT/'research/mathematical-engineering/me034'
    r=json.loads((out/'results.json').read_text());a=json.loads((out/'replay.json').read_text())
    assert r['complete'] and a['status']=='PASS'
    e=r['final_evaluation'];order=sorted(e['means'],key=lambda arm:e['means'][arm]['Brier'])
    rows=[]
    for arm in order:
        models=[c['tree'] if arm=='boosted' else next(s for s in c['students'] if s['arm']==arm) for c in r['cells']]
        rows.append({'arm':arm,**e['means'][arm],
            'active_downstream_parameters':None if arm=='boosted' else models[0]['trainable_parameters'],
            'mean_S_CPU_seconds':float(np.mean([m['charged_cpu_seconds'] for m in models])),
            'mean_charged_pipeline_CPU_seconds':float(np.mean([c['shared_charged_cpu_seconds']+m['charged_cpu_seconds'] for c,m in zip(r['cells'],models)])),
            'new_fits':len(models),'fully_matched_architecture_claim':False})
    summary={'study':r['study'],'rows':e['rows_per_seed'],'users':e['users'],
        'source_freeze_commit':r['source_commit'],'protocol_sha256':r['protocol_sha256'],
        'source_zip_sha256':r['final_source']['zip_sha256'],'final_snapshot_sha256':file_sha(out/'final-data.pt'),
        'source_user_remainder':6,'final_accesses':1,'new_neural_fits':63,'new_tree_fits':9,
        'metrics':rows,'primary':e['primary'],'audit_counts':{k:v for k,v in a.items() if isinstance(v,(int,float,bool))},
        'candidate_promoted':False,'reserved_remainder6_now_exposed':True,
        'native_KuaiRec_positive_memory_claims_closed':e['primary']['stop_native_KuaiRec_recipe']}
    write_json(report/'summary.json',summary)
    for name in ('results.json','replay.json','frozen-payloads.json','pipeline-exports.json','final-open-receipt.json'):
        shutil.copyfile(out/name,report/name)
    table='| Model | Brier ↓ | AUC ↑ | ECE10 ↓ | Active S weights | Mean S CPU s |\n|---|---:|---:|---:|---:|---:|\n'
    for row in rows:
        table+=f"| {row['arm']} | {row['Brier']:.6f} | {row['ROC_AUC']:.6f} | {row['ECE10']:.6f} | {row['active_downstream_parameters'] if row['active_downstream_parameters'] is not None else 'tree reference'} | {row['mean_S_CPU_seconds']:.4f} |\n"
    (report/'metrics-table.md').write_text(table)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=[x.replace('_','\n') for x in order]
    colors=['#0b8b72' if x=='boosted' else '#e55b46' if x=='native_frozen' else '#2766b0' if x=='common_adapt' else '#a1acbb' for x in order]
    fig,axes=plt.subplots(1,2,figsize=(13,5.4))
    axes[0].scatter(range(len(order)),[e['means'][x]['Brier'] for x in order],color=colors,s=65,zorder=3)
    for j,x in enumerate(order):axes[0].scatter([j-.15,j,j+.15],[e['pooled_per_seed'][str(s)][x]['Brier'] for s in SEEDS],s=12,color='#192837',zorder=4)
    axes[0].set_ylim(.15,.185);axes[0].set_ylabel('Brier score (lower is better)')
    axes[0].set_title('One new-user evaluation; dots show fitted seeds')
    axes[1].bar(range(len(order)),[x['mean_charged_pipeline_CPU_seconds']*3 for x in rows],color=colors)
    axes[1].set_ylabel('Charged training CPU seconds / 3-model seed ensemble')
    axes[1].set_title('Includes inherited R pretraining and teacher charges')
    for ax in axes:
        ax.set_xticks(range(len(order)),labels,fontsize=8);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('ME034 • stable-input recurrence • 438 users / 5,841 genuine later observations',fontsize=13)
    fig.text(.02,.015,'Three rotation probabilities averaged within each of three seeds. Practical references have unequal capacity/exposure; full compute parity failed.',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.94))
    fig.savefig(report/'final-results.png',dpi=180);fig.savefig(report/'final-results.svg');plt.close(fig)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':run()
