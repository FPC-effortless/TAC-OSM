#!/usr/bin/env python3
"""Require three original seed checkpoint/cohort identities; no pseudo replication."""
import argparse,json,math
from pathlib import Path

def aggregate(directory):
    records=[json.loads(p.read_text()) for p in sorted(directory.rglob('RECOVER_RESULTS.json'))]
    if len(records)!=3 or sorted(r['seed'] for r in records)!=[1701,1702,1703]:raise ValueError('three original seeds required')
    for r in records:
        if r['study']!='LANGMEM082' or len(r['future_rows'])!=768 or len(r['history_rows'])!=2528:raise ValueError('cohort grid')
        if not r['checkpoint_unchanged'] or not r['no_new_fits'] or r['gpu_minutes'] or r['protected_TEST']!='NOT_ACCESSED':raise ValueError('read-only integrity')
        if r['actual_source_future_poison_pairs']!=4 or r['bitwise_sham_pairs']!=4 or r['original_prefix_error']>3e-5 or r['max_actual_source_affine_error']>1e-5:raise ValueError('actual source gates')
    for field in ('actual_source_nextbyte_sha256','historical_targets_sha256'):
        if len({r[field] for r in records})!=1:raise ValueError('same real source targets required')
    if len({r['checkpoint_sha256'] for r in records})!=3:raise ValueError('distinct original checkpoints')
    if len({tuple((x['owner'],x['clock'],x['anchor'],x['historical_byte']) for x in r['history_rows']) for r in records})!=1:raise ValueError('historical cohort mismatch')
    if len({tuple((x['owner'],x['clock'],x['target']) for x in r['future_rows']) for r in records})!=1:raise ValueError('future cohort mismatch')
    return {'study':'LANGMEM082','seed_count':3,'actual_source_nextbyte_sha256':records[0]['actual_source_nextbyte_sha256'],
        'unique_nextbyte_targets_per_seed':768,'historical_query_cells_per_seed':2528,
        'independently_qualified':False,'no_new_fits':True,'gpu_minutes':0,'protected_TEST':'NOT_ACCESSED',
        'history_metrics':{age:{key:sum(r['history_metrics'][age][key] for r in records)/3 for key in records[0]['history_metrics'][age]} for age in ('8','32','128','256')},
        'future_metrics':{key:sum(r['future_metrics'][key] for r in records)/3 for key in records[0]['future_metrics']},
        'paired_event_metrics':{age:{
            'pairs_per_seed':[sum(x['age']==int(age) and x['wrong_content_donor'] is not None for x in r['history_rows']) for r in records],
            **{key:sum(sum(x[key] for x in r['history_rows'] if x['age']==int(age) and x['wrong_content_donor'] is not None)/sum(x['age']==int(age) and x['wrong_content_donor'] is not None for x in r['history_rows']) for r in records)/3 for key in ('matrix_correct','norm_matched_wrong_event_correct','event_removed_correct','wrong_event_minus_full_nextbyte_nll')}
        } for age in ('8','32','128','256')},
        'per_seed':[{k:r[k] for k in ('seed','checkpoint_sha256','history_metrics','future_metrics','timing','cpu_process_seconds')} for r in records],
        'limits':records[0]['limits']+['Repeated same source targets across ages/seeds are not independent examples; no confidence interval over cells.']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable output exists')
    r=aggregate(a.results);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(r,indent=2))
if __name__=='__main__':main()
