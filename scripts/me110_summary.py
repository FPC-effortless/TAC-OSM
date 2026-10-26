"""Descriptive all-seed original ME110 results and fixed-budget FIT curves."""
import argparse
import json
import numpy as np
from scripts import me110_typed_training as run

OUT=run.OUT/'SUMMARY.json'
KEYS=('pm_mae','raw_density_nll','coverage_by_channel','mae_by_channel','interval_width_by_channel','interval_score_by_channel')


def average(rows):
    result={}
    for key in KEYS:
        if key in rows[0]:result[key]=np.asarray([r[key] for r in rows],dtype=np.float64).mean(axis=0).tolist()
    return result


def execute(replay=False):
    if OUT.exists() and not replay:raise RuntimeError('original summary exists')
    index=run.budget_audit();scores=json.loads((run.OUT/'RESULTS.json').read_text())
    means={name:average([r['scores'][name] for r in scores['models']]) for name in ('gap','general')}
    curves={}
    for step in (0,*run.CHECKPOINTS):
        rows=[json.loads((run.OUT/run.ARCH/f"seed-{s}"/f'MONITOR-{step}.json').read_text()) for s in run.SEEDS]
        curves[str(step)]={name:average([r['cohorts'][name] for r in rows]) for name in ('gap','general')}
    curves_per_seed=[]
    for seed in run.SEEDS:
        stages={str(step):json.loads((run.OUT/run.ARCH/f'seed-{seed}'/f'MONITOR-{step}.json').read_text())['cohorts'] for step in (0,*run.CHECKPOINTS)}
        curves_per_seed.append(dict(seed=seed,stages=stages))
    counts={key:sum(r['gates'][key] for r in scores['model_gates']) for key in scores['model_gates'][0]['gates']}
    checks=sum(int(v) for row in index['models'] for key,v in row['guard_aggregate'].items() if key.endswith('.checks'))
    fallbacks=sum(int(v) for row in index['models'] for key,v in row['guard_aggregate'].items() if key.endswith('.fallbacks'))
    guard_max=max(v for row in index['models'] for key,v in row['guard_aggregate'].items() if key.endswith('.max_certified_upper_dot'))
    result=dict(study='ME110',final_report_means=means,model_gates=scores['model_gates'],gate_counts=counts,
        full_gate_seeds=sum(r['full_gate'] for r in scores['model_gates']),all_seed_gate=scores['all_seed_gate'],
        monitor_scope=run.sampling.MONITOR_SCOPE,monitor_means=curves,monitor_per_seed=curves_per_seed,
        fixed_budget_no_selection=True,updates=5*run.STEPS,examples=5*65536,
        fit_pipeline_cpu_seconds=sum(r['optimizer_cpu_seconds'] for r in index['models']),
        monitor_cpu_seconds=sum(r['monitor_cpu_seconds'] for r in index['models']),
        partial_training_flops=sum(r['training_flops'] for r in index['models']),
        flops_scope='partial forward/routed-VJP, excludes optimizer and unsupported operations',
        actual_block_certificates=checks,conditional_certificate_fallbacks=fallbacks,
        max_conditional_upper_dot=guard_max,independent_qualification=False,
        architecture_superiority_eligible=False,protected_scored=False)
    if replay:run._compare_numeric(result,json.loads(OUT.read_text()))
    else:run.write(OUT,result)
    print(json.dumps(dict(report=means,gate_counts=counts,full_gate_seeds=result['full_gate_seeds'],monitor=curves),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--replay',action='store_true');execute(p.parse_args().replay)
