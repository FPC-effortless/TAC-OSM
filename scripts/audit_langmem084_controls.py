#!/usr/bin/env python3
"""POST-HOC read-only paired audit. No source/model evaluation or new prediction.

Structural exclusion uses only anchor/decoy indices, never prediction outcome.
A decoy at b has context[b-4:b]; it includes target position a iff a<b<=a+4.
Both wrong arms and correct arm are scored on the SAME retained rows per seed.
This is not a new preregistered result and does not erase the original control.
"""
import argparse
import hashlib
import json
from pathlib import Path

def target_in_decoy_context(a,b):return a<b<=a+4

def audit(directory):
    records=[json.loads(p.read_text()) for p in sorted(directory.rglob('FACTORIAL_RESULTS.json'))]
    if len(records)!=3 or sorted(x['seed'] for x in records)!=[1701,1702,1703]:raise ValueError('three original artifacts')
    result={};identities=[]
    for record in records:
        if record['study']!='LANGMEM084' or len(record['rows'])!=7648:raise ValueError('original source artifact')
        seed=str(record['seed']);result[seed]={}
        arms=tuple(record['metrics'])
        for arm in arms:
            result[seed][arm]={}
            for age in (8,32,128,256):
                rows=[r for r in record['rows'] if r['arm']==arm and r['age']==age]
                hard=[r for r in rows if target_in_decoy_context(r['anchor'],r['wrong_hard_anchor'])]
                uniform=[r for r in rows if target_in_decoy_context(r['anchor'],r['wrong_uniform_anchor'])]
                clean=[r for r in rows if not target_in_decoy_context(r['anchor'],r['wrong_hard_anchor']) and not target_in_decoy_context(r['anchor'],r['wrong_uniform_anchor'])]
                result[seed][arm][str(age)]={'original_count':len(rows),'hard_overlap':len(hard),'uniform_overlap':len(uniform),
                    'paired_retained_count':len(clean),**{k:(sum(r[k] for r in clean)/len(clean) if clean else None) for k in ('correct','wrong_hard_correct','wrong_uniform_correct')}}
                for r in clean:identities.append((seed,arm,r['owner'],r['clock'],r['anchor']))
    means={}
    for arm in records[0]['metrics']:
        means[arm]={}
        for age in ('8','32','128','256'):
            xs=[result[str(s)][arm][age] for s in (1701,1702,1703)]
            means[arm][age]={'paired_counts_per_seed':[x['paired_retained_count'] for x in xs],
                **{k:sum(x[k] for x in xs)/3 for k in ('correct','wrong_hard_correct','wrong_uniform_correct')}}
    return {'study':'LANGMEM084','analysis':'POST_HOC_STRUCTURAL_DECOY_TARGET_OVERLAP_AUDIT',
        'preregistered_primary_results_replaced':False,'new_model_evaluations':0,'new_fits':0,'new_DEV_reads':0,
        'rule':'Exclude both decoy contexts that contain source index a: a<b<=a+4; paired correct and both wrong scores on retained identical rows.',
        'risk':'Decoy context can contain queried historical byte index; all bytes still observed by query time, so this is not future-byte leakage.',
        'all_original_and_excluded_rows_preserved':True,'means':means,'per_seed':result,
        'retained_row_identity_sha256':hashlib.sha256(json.dumps(identities,separators=(',',':')).encode()).hexdigest(),
        'limits':['Post-hoc structural audit, not a prospectively confirmed control or new unseen validation.',
                  'Controls still differ in historical writer age; no full temporal matching claim.',
                  'Read vectors intentionally contain historical content; audit removes direct target-position routes through the decoy key context only.']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable audit exists')
    r=audit(a.results);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(r['means'],indent=2))
if __name__=='__main__':main()
