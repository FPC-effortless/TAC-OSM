"""Fixed ME024 contrasts; descriptive output, no selection or training."""
import argparse
import json
from pathlib import Path
import numpy as np


def summarize(root,out):
    r=json.loads((Path(root)/'results.json').read_text()); c=r['contract']
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    rows={(m['arm'],m['seed']):m for m in r['models']}; assert len(rows)==12
    def delta(a,b): return [rows[a,s]['dev']['Brier']-rows[b,s]['dev']['Brier'] for s in c['seeds']]
    primary=np.array(delta('coupled_rms','rms'))
    memory=np.array([rows['coupled_rms',s]['feature_mean_brier_delta'] for s in c['seeds']])
    ece=float(np.mean([rows['coupled_rms',s]['dev']['ECE10']-rows['rms',s]['dev']['ECE10'] for s in c['seeds']]))
    gates={'all_primary_negative':bool((primary<0).all()),'primary_mean_material':bool(primary.mean()<=-.0005),
           'all_feature_mean_deltas_at_least_0002':bool((memory>=.0002).all()),'mean_ece_guard':bool(ece<=.002)}
    summary={'source_commit':r['source_commit'],'primary_paired_deltas':primary.tolist(),
        'primary_mean_delta':float(primary.mean()),'coupled_rms_feature_mean_deltas':memory.tolist(),
        'mean_ece_delta':ece,'gates':gates,'engineering_gate_passed':all(gates.values()),
        'secondary_paired_deltas':{'coupling_without_rms':delta('coupled','plain'),
            'rms_without_coupling':delta('rms','plain'),'rms_with_coupling':delta('coupled_rms','coupled')},
        'training_process_cpu_seconds':sum(m['train_cpu_seconds'] for m in rows.values()),
        'arm_means':{},'independent_test_opened':False,'architectural_superiority_claim':False}
    lines=['| Native variant | Seed 14001 Brier | Seed 14002 | Seed 14003 | Mean Brier | Mean ECE | Feature-mean Brier delta | CPU s |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for a in c['arms']:
        arm=[rows[a,s] for s in c['seeds']]
        mean={k:float(np.mean([q['dev'][k] for q in arm])) for k in ['Brier','log_loss','ROC_AUC','ECE10']}
        mean['feature_mean_brier_delta']=float(np.mean([q['feature_mean_brier_delta'] for q in arm]))
        mean['train_cpu_seconds']=float(np.mean([q['train_cpu_seconds'] for q in arm]))
        mean['feedback_disabled_brier_delta']=float(np.mean([q['feedback_disabled_dev']['Brier']-q['dev']['Brier'] for q in arm]))
        summary['arm_means'][a]=mean
        lines.append('| '+a+' | '+' | '.join(f'{q["dev"]["Brier"]:.6f}' for q in arm)+
            f' | {mean["Brier"]:.6f} | {mean["ECE10"]:.6f} | {mean["feature_mean_brier_delta"]:+.6f} | {mean["train_cpu_seconds"]:.2f} |')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'table.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    for i,a in enumerate(c['arms']):
        traces=[rows[a,s]['trace'] for s in c['seeds']]
        for ax,key in zip(axes,['step','train_cpu_seconds']):
            for tr in traces: ax.plot([v[key] for v in tr],[v['dev']['Brier'] for v in tr],color=f'C{i}',alpha=.25)
            ax.plot(np.mean([[v[key] for v in tr] for tr in traces],axis=0),
                np.mean([[v['dev']['Brier'] for v in tr] for tr in traces],axis=0),color=f'C{i}',label=a,lw=2)
    for ax in axes: ax.set_ylabel('Historical-development Brier ↓'); ax.grid(alpha=.2)
    axes[0].set_xlabel('Optimizer updates'); axes[1].set_xlabel('Measured training CPU seconds'); axes[1].legend()
    fig.suptitle('Native TAC: tied slow feedback × split RMS readout\nAuthenticated KuaiRec development only; thin lines show seeds')
    fig.savefig(out/'learning-curves.png',dpi=160); fig.savefig(out/'learning-curves.svg')
    print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',default='results/me024')
    p.add_argument('--out',default='research/mathematical-engineering/me024')
    a=p.parse_args(); summarize(a.root,a.out)
