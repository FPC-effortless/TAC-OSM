#!/usr/bin/env python3
"""ME004 independent recorded-only replay. No training or post-day-25 evaluation.
Restore original GitHub artifact 12235023980 first; compare recorded scores.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import torch
from scripts.me004_native_real_forecast import acquire,prepare,make_model,score

def compare(a,b,key):
    if isinstance(a,dict):
        if set(a)!=set(b):raise AssertionError(f'{key}: metric keys changed')
        for k in a:compare(a[k],b[k],key+'.'+k)
    elif isinstance(a,list):
        if len(a)!=len(b):raise AssertionError(f'{key}: metric count')
        for i,(v,w) in enumerate(zip(a,b)):compare(v,w,key+str(i))
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        if not math.isclose(float(a),float(b),rel_tol=3e-6,abs_tol=3e-6):
            raise AssertionError(f'{key}: {a} != {b}')
    elif a!=b:raise AssertionError(f'{key}: {a} != {b}')

def replay(original,output):
    root=Path(original);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    recorded=json.loads((root/'results.json').read_text())
    frozen_source=json.loads((root/'source.json').read_text())
    if len(recorded['fits'])!=6 or recorded['study']!='ME004':raise ValueError('unexpected manifest')
    assert recorded['source']==frozen_source
    assert recorded['official_future_test_unopened'] and frozen_source['future_after_day25_not_read']
    hashes={}
    for fit in recorded['fits']:
        p=root/fit['checkpoint']
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        if digest!=fit['checkpoint_sha256']:raise ValueError('checkpoint digest mismatch')
        hashes[f"{fit['model']}-{fit['seed']}"]=digest
    data,times,src=acquire(output)
    if src!=frozen_source:raise ValueError('source provenance differs')
    training,validation,threshold,cohort=prepare(data,times)
    compare(cohort,recorded['cohort'],'cohort')
    persistence=score(torch.zeros_like(validation[1]),validation[1],threshold)
    compare(persistence,recorded['baseline_persistence'],'persistence')
    results=[]
    torch.set_num_threads(1)
    for fit in recorded['fits']:
        ck=torch.load(root/fit['checkpoint'],weights_only=True,map_location='cpu')
        if ck['kind']!=fit['model'] or ck['seed']!=fit['seed'] or ck['steps']!=450:
            raise ValueError('checkpoint identity/steps mismatch')
        model=make_model(fit['model']);model.load_state_dict(ck['state_dict'],strict=True);model.eval()
        x,y=validation
        with torch.inference_mode():
            pred=torch.cat([model(x[i:i+256]) for i in range(0,len(x),256)])
        s=score(pred,y,threshold)
        compare(s,fit['validation'],f"score:{fit['model']}:{fit['seed']}")
        results.append({'model':fit['model'],'seed':fit['seed'],'mae':s['macro_mae'],
                        'replay_matches_original':True})
    out={'study':'ME004_RECORDED_ONLY_AUDIT','zero_refits':True,
         'source_reproducible':True,'source_sha256':src['publisher_archive_sha256'],
         'checkpoint_hashes':hashes,'fits_replayed':len(results),
         'metrics_reproduced':True,'scores':results,
         'read_future_after_day25':False,'architecture_promoted':False}
    (output/'replay.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':'PASS','replayed':len(results),'refits':0},sort_keys=True))
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--original',default='results/me004')
    p.add_argument('--output',default='results/me004-audit')
    a=p.parse_args();replay(a.original,a.output)
