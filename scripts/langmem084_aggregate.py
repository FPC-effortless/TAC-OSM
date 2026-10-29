#!/usr/bin/env python3
"""Strict original-source paired no-fit factorial aggregation, all arms preserved."""
import argparse
import json
from pathlib import Path
AGES=('8','32','128','256')

def aggregate(directory):
    records=[json.loads(p.read_text()) for p in sorted(directory.rglob('FACTORIAL_RESULTS.json'))]
    if len(records)!=3 or sorted(r['seed'] for r in records)!=[1701,1702,1703]:raise ValueError('three seeds required')
    for r in records:
        if r['study']!='LANGMEM084' or not r['weights_unchanged'] or not r['no_new_fits'] or r['optimizer_updates'] or r['new_trainable_parameters']:raise ValueError('frozen model contract')
        if r['historical_cells_per_seed']!=956 or r['repeated_arm_rows']!=7648 or len(r['rows'])!=7648:raise ValueError('registered cohort')
        if r['max_actual_native_bank_error']>1e-5 or r['original_prefix_error']>3e-5:raise ValueError('actual source parity')
        if any(r[k]!=2 for k in ('actual_source_future_poison_pairs','source_write_key_current_byte_invariance_pairs','bitwise_source_sham_pairs')):raise ValueError('actual source gates')
        if r['bank_tensor_bytes_each']!=256 or r['gpu_minutes'] or r['protected_TEST']!='NOT_ACCESSED':raise ValueError('resource boundary')
    if len({r['original_checkpoint_sha256'] for r in records})!=3:raise ValueError('distinct original checkpoints')
    if len({r['source_historical_targets_sha256'] for r in records})!=1:raise ValueError('identical real targets')
    if len({json.dumps(r['source_documents_sha256'],sort_keys=True) for r in records})!=1:raise ValueError('same documents')
    arms=tuple(records[0]['metrics'])
    if len(arms)!=8 or any(set(r['metrics'])!=set(arms) for r in records):raise ValueError('eight identical registered arms')
    mean={}
    for arm in arms:
        mean[arm]={}
        for age in AGES:
            xs=[r['metrics'][arm][age] for r in records]
            if len({x['count'] for x in xs})!=1:raise ValueError('same age denominators')
            result={k:sum(x[k] for x in xs)/3 for k in xs[0] if k not in ('unique_context','conflicting_context','count')}
            result['count_per_seed']=xs[0]['count']
            result['correct_minus_uniform_wrong']=result['correct']-result['wrong_uniform_correct']
            result['correct_minus_hard_wrong']=result['correct']-result['wrong_hard_correct']
            for subgroup in ('unique_context','conflicting_context'):
                if len({x[subgroup]['count'] for x in xs})!=1:raise ValueError('subgroup denominators')
                n=xs[0][subgroup]['count']
                result[subgroup]={'count_per_seed':n,**{k:(sum(x[subgroup][k] for x in xs)/3 if n else None) for k in ('correct','wrong_hard_correct','wrong_uniform_correct')}}
            mean[arm][age]=result
    # Explicit paired row identities, not merely caller-supplied hashes.
    cohort=lambda r:tuple((x['owner'],x['clock'],x['anchor'],x['age'],x['arm'],x['historical_byte']) for x in r['rows'])
    if len({cohort(r) for r in records})!=1:raise ValueError('actual cohort rows differ')
    return {'study':'LANGMEM084','seed_count':3,'historical_cells_per_seed':956,'arm_rows_per_seed':7648,
        'source_historical_targets_sha256':records[0]['source_historical_targets_sha256'],
        'source_documents_sha256':records[0]['source_documents_sha256'],'source_role':records[0]['source_role'],
        'no_new_fits':True,'gpu_minutes':0,'protected_TEST':'NOT_ACCESSED','independently_qualified':False,
        'metrics':mean,'per_seed':[{k:r[k] for k in ('seed','original_checkpoint_sha256','key_geometry','metrics','cpu_process_seconds','bank_update_wall_seconds','max_actual_native_bank_error','original_prefix_error')} for r in records],
        'limits':records[0]['limits']+['Repeated labels across arms and seeds are not independent samples; no cell-wise confidence intervals.']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable aggregate exists')
    r=aggregate(a.results);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(r,indent=2,sort_keys=True))
if __name__=='__main__':main()
