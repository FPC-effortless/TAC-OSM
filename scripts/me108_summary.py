"""Publish all five seeds without selecting a winner or changing any gate."""
import json
from pathlib import Path
import numpy as np
from scripts import me108_role_training as run


def summary():
    d=json.loads((run.OUT/'RESULTS.json').read_text())
    parent=json.loads((run.PARENT/'RESULTS.json').read_text())
    forecasts=np.load(run.OUT/'FORECASTS.npz')
    old_forecasts=np.load(run.PARENT/'FORECASTS.npz')
    result=dict(full_gate='PASS' if d['all_seed_gate'] else 'FAIL',all_seed_gate=d['all_seed_gate'],
                independent_qualification=False,protected_test_scored=False,cohorts={})
    for cohort in ('gap','general'):
        actual=d['cohorts'][cohort]['models']
        prior=[r for r in parent['cohorts'][cohort]['models'] if r['recipe']=='strong_reconstruction']
        means=lambda rows:dict(pm_mae=float(np.mean([r['pm_mae'] for r in rows])),
            six_channel_nll=float(np.mean([r['raw_density_nll'] for r in rows])),
            pm90coverage=float(np.mean([r['coverage_by_channel'][0] for r in rows])),
            pm_interval_score=float(np.mean([r['interval_score_by_channel'][0] for r in rows])))
        a,b=means(actual),means(prior)
        a['pm90width']=float(np.mean([np.mean(forecasts[f'{cohort}_role_guarded_{s}'][:,0,2]-forecasts[f'{cohort}_role_guarded_{s}'][:,0,0]) for s in run.SEEDS]))
        b['pm90width']=float(np.mean([np.mean(old_forecasts[f'{cohort}_strong_reconstruction_{s}'][:,0,2]-old_forecasts[f'{cohort}_strong_reconstruction_{s}'][:,0,0]) for s in run.PARENT_SEEDS]))
        refs={name:float(np.mean(np.abs(forecasts[f'{cohort}_{name}'][:,0]-forecasts[f'{cohort}_truth'][:,0]))) for name in ('locf','ridge')}
        result['cohorts'][cohort]=dict(candidate_five_seed_mean=a,parent_five_seed_mean=b,
            relative_change_percent={key:100*(a[key]/b[key]-1) for key in ('pm_mae','pm_interval_score','pm90width','six_channel_nll')},
            unmatched_practical_references=refs,
            seeds_better_than_each_reference={name:sum(r['pm_mae']<score for r in actual) for name,score in refs.items()})
    result['gate_counts']={name:sum(r['gates'][name] for r in d['model_gates']) for name in d['model_gates'][0]['gates']}
    result['full_seed_passes']=sum(r['full_gate'] for r in d['model_gates'])
    index=json.loads((run.OUT/'FIT_INDEX.json').read_text())
    result['training']=dict(models=len(index['models']),updates_each=4096,total_updates=20480,
        guarded_block_checks=20480*13,
        max_computed_applied_point_dot=max(v for r in index['models'] for k,v in r['guard_aggregate'].items() if k.endswith('.max_applied_dot')),
        projected_blocks=sum(v for r in index['models'] for k,v in r['guard_aggregate'].items() if k.endswith('.projected')),
        actual_rounding_rejected_blocks=sum(v for r in index['models'] for k,v in r['guard_aggregate'].items() if k.endswith('.applied_rounding_fallback')),
        new_training_cpu_seconds=sum(r['optimizer_cpu_seconds'] for r in index['models']))
    return result


if __name__=='__main__':
    path=run.OUT/'SUMMARY.json';result=summary()
    if path.exists():
        run._compare_numeric(result,json.loads(path.read_text()))
        print('ME108_SUMMARY_REPLAY_PASS')
    else:
        run.write(path,result);print('ME108_SUMMARY_CREATED')
