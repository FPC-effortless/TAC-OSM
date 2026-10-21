"""Fixed ME027 contrasts and descriptive resource curves; never selects weights."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.me027_state_research import C
from scripts.me027_audit import selected
from scripts.me022_native_readout import save_json


def summarize(root,out):
    root,out=Path(root),Path(out);out.mkdir(parents=True,exist_ok=True)
    r=json.loads((root/'results.json').read_text());replay=json.loads((root/'replay.json').read_text())
    rows={(m['arm'],m['seed']):m for m in r['models']};assert len(rows)==12
    label=f'cpu{C["primary_cpu_budget_seconds"]}'
    points={(a,s):selected(rows[a,s],label) for a in C['arms'] for s in C['seeds']}
    deltas=np.array([points['tac',s]['dev']['Brier']-points['mlp',s]['dev']['Brier'] for s in C['seeds']])
    memory=[q['Brier_delta'] for q in replay['state_replacement_diagnostics'] if q['arm']=='tac']
    ece=float(np.mean([points['tac',s]['dev']['ECE10']-points['mlp',s]['dev']['ECE10'] for s in C['seeds']]))
    cpu_ok=all(20<=q['charged_cpu_seconds']<=20.2 for q in points.values())
    gates={'all_primary_seed_deltas_negative':bool((deltas<0).all()),'primary_mean_at_most_minus_0005':bool(deltas.mean()<=-.0005),
        'all_TAC_feature_mean_deltas_at_least_0002':all(x>=.0002 for x in memory),'mean_ECE_guard':ece<=.002,
        'all_primary_spends_within_one_percent':cpu_ok}
    summary={'source_commit':r['source_commit'],'primary_paired_Brier_deltas':deltas.tolist(),'primary_mean_Brier_delta':float(deltas.mean()),
        'TAC_feature_mean_deltas':memory,'mean_TAC_minus_MLP_ECE_delta':ece,'gates':gates,'engineering_gate_passed':all(gates.values()),
        'lanes':{},'primary_bootstrap':json.loads((root/'primary-user-bootstrap.json').read_text()),
        'overall_simultaneous_resource_parity':'FAIL','strict_architecture_superiority_allowed':False,'independent_validation':False,
        'primary_charged_CPU_range_seconds':[min(x['charged_cpu_seconds'] for x in points.values()),max(x['charged_cpu_seconds'] for x in points.values())],
        'actual_final_training_steps':{a:[rows[a,s]['actual_final_steps'] for s in C['seeds']] for a in C['arms']}}
    lines=['| Lane / checkpoint | Model | Seed 14001 Brier | Seed 14002 | Seed 14003 | Mean Brier | Mean ECE | Mean charged CPU s | Mean updates |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for lane,labels in [('equal_exposure',[f'step{n}' for n in C['exposure_steps']]),('charged_CPU',[f'cpu{n}' for n in C['cpu_budgets_seconds']])]:
        summary['lanes'][lane]={}
        for l in labels:
            summary['lanes'][lane][l]={}
            for a in C['arms']:
                q=[selected(rows[a,s],l) for s in C['seeds']]
                mean={k:float(np.mean([v['dev'][k] for v in q])) for k in ['Brier','log_loss','ROC_AUC','ECE10']}
                mean['charged_CPU_seconds']=float(np.mean([v['charged_cpu_seconds'] for v in q]));mean['updates']=float(np.mean([v['step'] for v in q]))
                summary['lanes'][lane][l][a]=mean
                lines.append(f'| {lane} / {l} | {a} | '+' | '.join(f'{v["dev"]["Brier"]:.6f}' for v in q)+f' | {mean["Brier"]:.6f} | {mean["ECE10"]:.6f} | {mean["charged_CPU_seconds"]:.3f} | {mean["updates"]:.1f} |')
    summary['parity']={'parameters':'PASS','shared_source_features':'PASS','tuning_selection_privileges':'PASS',
        'exposure_lane_data_presentations':'PASS','exposure_lane_actual_compute':'UNMATCHED',
        'CPU_lane_primary_charged_budget':'PASS' if cpu_ok else 'FAIL','CPU_lane_data_presentations':'UNMATCHED',
        'complete_FLOPs_energy':'UNESTABLISHED','test':'REUSED_DEVELOPMENT_ONLY'}
    save_json(out/'summary.json',summary);(out/'table.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for i,a in enumerate(C['arms']):
        for s in C['seeds']:
            t=rows[a,s]['trace'];axes[0].plot([v['step'] for v in t if v['step']<=300],[v['dev']['Brier'] for v in t if v['step']<=300],color=f'C{i}',alpha=.3)
            axes[1].plot([v['charged_cpu_seconds'] for v in t if v['charged_cpu_seconds']<=20.2],[v['dev']['Brier'] for v in t if v['charged_cpu_seconds']<=20.2],color=f'C{i}',alpha=.3)
        for ax,lane,labels in [(axes[0],'equal_exposure',[f'step{x}' for x in C['exposure_steps']]),(axes[1],'charged_CPU',[f'cpu{x}' for x in C['cpu_budgets_seconds']])]:
            vals=[summary['lanes'][lane][l][a] for l in labels]
            ax.plot([v['updates'] if lane=='equal_exposure' else v['charged_CPU_seconds'] for v in vals],[v['Brier'] for v in vals],color=f'C{i}',marker='o',label=a,lw=2)
    for ax in axes:ax.set_ylabel('Historical-development Brier ↓');ax.grid(alpha=.2)
    axes[0].set_xlabel('Optimizer updates (exposure lane)');axes[1].set_xlabel('Charged training CPU seconds');axes[1].legend()
    fig.suptitle('Shared-feature learned-state study · all three seeds · separate resource lanes')
    fig.savefig(out/'resource-curves.svg');fig.savefig(out/'resource-curves.png',dpi=160)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['lanes','actual_final_training_steps']}));return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/me027');p.add_argument('--out',default='research/mathematical-engineering/me027')
    a=p.parse_args();summarize(a.root,a.out)
