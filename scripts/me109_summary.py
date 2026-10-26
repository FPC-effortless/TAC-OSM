"""Publish all five seeds without selecting a winner or changing any gate."""
import json
from pathlib import Path
import numpy as np
from scripts import me109_residual_training as run


def summary():
    d=json.loads((run.OUT/'RESULTS.json').read_text())
    parent=json.loads((run.PARENT/'RESULTS.json').read_text())
    original=json.loads((run.ROOT/'research/replay-bundles/ME101/RESULTS.json').read_text())
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
            pm_interval_score=float(np.mean([r['interval_score_by_channel'][0] for r in rows])),
            mae_by_channel=np.mean([r['mae_by_channel'] for r in rows],axis=0).tolist(),
            coverage_by_channel=np.mean([r['coverage_by_channel'] for r in rows],axis=0).tolist())
        a,b=means(actual),means(prior)
        a['pm90width']=float(np.mean([np.mean(forecasts[f'{cohort}_residual_guarded_{s}'][:,0,2]-forecasts[f'{cohort}_residual_guarded_{s}'][:,0,0]) for s in run.SEEDS]))
        b['pm90width']=float(np.mean([np.mean(old_forecasts[f'{cohort}_strong_reconstruction_{s}'][:,0,2]-old_forecasts[f'{cohort}_strong_reconstruction_{s}'][:,0,0]) for s in run.PARENT_SEEDS]))
        refs={name:float(np.mean(np.abs(forecasts[f'{cohort}_{name}'][:,0]-forecasts[f'{cohort}_truth'][:,0]))) for name in ('locf','ridge')}
        result['cohorts'][cohort]=dict(candidate_five_seed_mean=a,parent_five_seed_mean=b,
            relative_change_percent={key:100*(a[key]/b[key]-1) for key in ('pm_mae','pm_interval_score','pm90width','six_channel_nll')},
            unmatched_practical_references=refs,
            seeds_better_than_each_reference={name:sum(r['pm_mae']<score for r in actual) for name,score in refs.items()})
    # Exact original limits, with signed exceedance and nonnegative failure size.
    old_gap={r['seed']:r for r in original['cohorts']['gap']['models'] if r.get('context')==256}
    parents={cohort:{r['seed']:r for r in parent['cohorts'][cohort]['models']
                    if r['recipe']=='strong_reconstruction'} for cohort in ('gap','general')}
    by_seed={cohort:{r['seed']:r for r in d['cohorts'][cohort]['models']}
             for cohort in ('gap','general')}
    recorded={r['seed']:r for r in d['model_gates']}
    result['per_seed_gate_thresholds']=[]
    daily_locf=result['cohorts']['general']['unmatched_practical_references']['locf']
    for seed in run.SEEDS:
        old=old_gap[seed-8000];pg=parents['gap'][seed-7000];pd=parents['general'][seed-7000]
        g=by_seed['gap'][seed];n=by_seed['general'][seed]
        limits=dict(gap_point=min(1.05*old['pm_mae'],1.05*pg['pm_mae']),
            daily_point=min(1.05*daily_locf,1.05*pd['pm_mae']),
            gap_nll=min(6.234504261016846,1.05*pg['raw_density_nll']),
            interval_score=min(1.05*old['interval_score_by_channel'][0],.90*pg['interval_score_by_channel'][0]))
        actual=dict(gap_point=g['pm_mae'],daily_point=n['pm_mae'],
                    gap_nll=g['raw_density_nll'],interval_score=g['interval_score_by_channel'][0])
        checks={key:dict(actual=actual[key],maximum=limit,signed_overrun=actual[key]-limit,
                        overrun=max(0.,actual[key]-limit),passed=actual[key]<=limit)
                for key,limit in limits.items()}
        coverage=g['coverage_by_channel'][0]
        checks['coverage']=dict(actual=coverage,minimum=.90,maximum=.98,
            below_minimum=max(0.,.90-coverage),above_maximum=max(0.,coverage-.98),
            overrun=max(0.,.90-coverage,coverage-.98),passed=.90<=coverage<=.98)
        if {key:row['passed'] for key,row in checks.items()} != recorded[seed]['gates']:
            raise RuntimeError('summary gates differ from frozen original score')
        result['per_seed_gate_thresholds'].append(dict(seed=seed,parent_seed=seed-7000,
            me101_reference_seed=seed-8000,gates=checks,full_gate=recorded[seed]['full_gate']))
    result['gate_counts']={name:sum(r['gates'][name] for r in d['model_gates']) for name in d['model_gates'][0]['gates']}
    result['full_seed_passes']=sum(r['full_gate'] for r in d['model_gates'])
    index=json.loads((run.OUT/'FIT_INDEX.json').read_text())
    if {r['seed'] for r in index['models']} != set(run.SEEDS) or len(index['models']) != 5:
        raise RuntimeError('summary requires all five registered seeds')
    if any(r['actual_steps'] != 4096 or r['trainable_parameters'] != 104663 for r in index['models']):
        raise RuntimeError('summary training budget differs from frozen protocol')
    result['training']=dict(models=len(index['models']),trainable_parameters_each=104663,updates_each=4096,total_updates=20480,
        guarded_block_checks=20480*13,
        max_computed_applied_point_dot=max(v for r in index['models'] for k,v in r['guard_aggregate'].items() if k.endswith('.max_applied_dot')),
        projected_blocks=sum(v for r in index['models'] for k,v in r['guard_aggregate'].items() if k.endswith('.projected')),
        proposal_rounding_rejected_blocks=sum(v for r in index['models'] for k,v in r['guard_aggregate'].items() if k.endswith('.rounding_fallback')),
        actual_rounding_rejected_blocks=sum(v for r in index['models'] for k,v in r['guard_aggregate'].items() if k.endswith('.applied_rounding_fallback')),
        new_training_cpu_seconds=sum(r['optimizer_cpu_seconds'] for r in index['models']))
    result['training']['guard_counts_by_block']={
        block:{key:sum(row['guard_aggregate'][f'{block}.{key}'] for row in index['models'])
               for key in ('projected','rounding_fallback','applied_rounding_fallback')}
        for block in run.ROLE_BLOCKS}
    result['training']['guard_count_denominator']=20480*13
    return result


if __name__=='__main__':
    path=run.OUT/'SUMMARY.json';result=summary()
    if path.exists():
        run._compare_numeric(result,json.loads(path.read_text()))
        print('ME109_SUMMARY_REPLAY_PASS')
    else:
        run.write(path,result);print('ME109_SUMMARY_CREATED')
