#!/usr/bin/env python3
"""Read original CODEVALUE003 saved generations; no model inference or fitting."""
import argparse
import hashlib
import json
from pathlib import Path


def analyze(root):
    rows=[(p,json.loads(p.read_text())) for p in sorted((root/'scores').glob('*.json'))]
    transfer=[(p,r) for p,r in rows if r['source']['partition']=='TRANSFER_DEV']
    baseline={r['source']['file']:r for p,r in transfer if r['predictor']=='wb6'}
    result=[];hashes={}
    for seed in (1701,1702,1703):
        joint={'both_correct':0,'tac_only_correct':0,'wb6_only_correct':0,'both_wrong':0}
        per_publisher={}
        for p,r in transfer:
            if r['predictor']!='tac' or r['seed']!=seed or r['step']!=10240:continue
            b=baseline[r['source']['file']]
            known={o['position']:o for o in b['offers']}
            own=per_publisher.setdefault(r['source']['group'],dict.fromkeys(joint,0))
            for offer in r['offers']:
                t=bool(offer['exact']['4']);w=bool(known[offer['position']]['exact']['4'])
                key='both_correct' if t and w else 'tac_only_correct' if t else 'wb6_only_correct' if w else 'both_wrong'
                joint[key]+=1;own[key]+=1
            for path in (p,next(p2 for p2,r2 in transfer if r2 is b)):
                hashes[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
        if sum(joint.values())!=5120:raise ValueError('missing paired offer matrix')
        result.append({'seed':seed,**joint,'by_publisher':per_publisher,
            'oracle_union_exact4':5120-joint['both_wrong'],
            'oracle_max_additional_correct_over_wb6':joint['tac_only_correct'],
            'oracle_is_deployable':False})
    return {'study':'CODEVALUE004_PREPARATION','source_study':'CODEVALUE003',
        'scope':'EXPOSED_DEV_SAVED_PREDICTIONS_ONLY','optimizer_updates':0,'new_predictions':0,
        'seeds':result,'original_score_sha256':hashes,
        'interpretation':'Oracle union measures error complementarity only. Selecting the correct predictor requires a real causal gate and adds costs. It is not measured fusion performance or customer benefit.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.write_text(json.dumps(analyze(a.source),indent=2)+'\n')
