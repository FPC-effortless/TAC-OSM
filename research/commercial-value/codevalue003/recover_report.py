#!/usr/bin/env python3
"""Recover a report from original outputs, without prediction or retraining."""
import argparse
import hashlib
import json
from pathlib import Path

from core import ARMS,MILESTONES,SEEDS,atomic_json,read_sources,sha
from evaluate_run import summarize
from scripts.validate_research_parity import check_manifest


def recover(out,data):
    seal=json.loads((out/'SEAL.json').read_text())
    if len(seal['checkpoints'])!=45:raise ValueError('incomplete checkpoint seal')
    for name,digest in seal['checkpoints'].items():
        if sha(out/name)!=digest:raise ValueError('original checkpoint changed: '+name)
    check_manifest(json.loads((out/'FINAL_MANIFEST.json').read_text()),'final')
    _,sources=read_sources(data,{'FIT','DEV','TRANSFER_DEV'})
    lookup={r['file']:(r,raw) for r,raw in sources}
    paths=sorted((out/'scores').glob('*.json'))
    expected={(arm,seed,step,r['file']) for r,raw in sources
              if r['partition']=='DEV' for arm in ARMS for seed in SEEDS for step in MILESTONES}
    expected|={(arm,seed,10240,r['file']) for r,raw in sources
               if r['partition']=='TRANSFER_DEV' for arm in ARMS for seed in SEEDS}
    expected|={(arm,None,None,r['file']) for r,raw in sources
               if r['partition']!='FIT' for arm in ('p3','wb6')}
    actual=set();offers_checked=0;score_hashes={}
    for path in paths:
        row=json.loads(path.read_text());r,raw=lookup[row['source']['file']]
        if row['source']['sha256']!=r['sha256']:raise ValueError('score source mismatch')
        identity=(row['predictor'],row.get('seed'),row.get('step'),r['file'])
        if identity in actual:raise ValueError('duplicate score record')
        actual.add(identity)
        if row['targets']!=len(raw)-64:raise ValueError('wrong full-file target denominator')
        if not 0<=row['top1']<=row['top5']<=row['targets']:raise ValueError('invalid accuracy counts')
        completion=row['predictor'] in ('p3','wb6') or row.get('step')==10240
        if completion:
            if len(row['offers'])!=512 or len({o['position'] for o in row['offers']})!=512:
                raise ValueError('incorrect unique offer denominator')
            for o in row['offers']:
                generated=bytes.fromhex(o['generated_hex'])
                gold=raw[o['position']:o['position']+32]
                if len(generated)!=32 or len(gold)!=32:raise ValueError('truncated suggestion')
                longest=0
                for a,b in zip(generated,gold):
                    if a!=b:break
                    longest+=1
                if longest!=o['longest_correct_prefix']:raise ValueError('prefix score mismatch')
                for n,value in o['exact'].items():
                    if value!=int(longest>=int(n)):raise ValueError('exact completion score mismatch')
                offers_checked+=1
        score_hashes[path.name]=sha(path)
    if actual!=expected:raise ValueError(f'incomplete/extra score matrix: actual{len(actual)} expected{len(expected)}')
    summarize(out)
    receipt={'study':'CODEVALUE003','recovery':'SUMMARY_SERIALIZATION_ONLY',
             'original_remote_status':json.loads((out/'RUNTIME.json').read_text())['status'],
             'original_failure':'NumPy int64 in publisher/seed win counts was not JSON serializable',
             'fix':'Convert only the two derived win counts to Python int before JSON output',
             'retraining_updates':0,'new_predictions':0,'additional_GPU_minutes':0,
             'verified_original_checkpoints':45,'verified_score_files':len(paths),
             'recomputed_saved_suggestions':offers_checked,
             'score_sha256':score_hashes,'results_sha256':sha(out/'RESULTS.json'),
             'status':'RECOVERED_INTERNAL_RESULTS_ORIGINAL_RUNTIME_FAILURE_PRESERVED'}
    atomic_json(out/'RECOVERY.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='score_sha256'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);a=p.parse_args();recover(a.out,a.data)
