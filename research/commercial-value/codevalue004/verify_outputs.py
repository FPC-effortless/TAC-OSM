#!/usr/bin/env python3
"""Read-only replay audit of completed outputs; no new prediction or fitting."""
import argparse
import json
from pathlib import Path
from protocol004 import atomic_json,read_sources,select_checkpoint,sha


def verify(out,data):
    seal=json.loads((out/'SELECTION_SEAL.json').read_text())
    runtime=json.loads((out/'RUNTIME.json').read_text())
    if runtime['status']!='COMPLETE_INTERNAL_RESEARCH':raise ValueError('remote study incomplete')
    if len(seal['checkpoints'])!=120 or len(seal['selections'])!=24:raise ValueError('incomplete sealed matrix')
    for path,digest in seal['checkpoints'].items():
        if sha(out/path)!=digest:raise ValueError('original checkpoint changed: '+path)
    for name,selection in seal['selections'].items():
        if select_checkpoint(selection['candidate_curve'])!=selection['selected_step']:raise ValueError('selection differs from registered DEV rule')
        path=out/name/f'MODEL-{selection["selected_step"]:06d}.pt'
        if sha(path)!=selection['selected_checkpoint_sha256']:raise ValueError('chosen checkpoint mismatch')
    _,sources=read_sources(data,{'TRANSFER_DEV'});lookup={r['file']:(r,b) for r,b in sources}
    paths=sorted((out/'scores').glob('*.json'))
    if len(paths)!=280:raise ValueError('missing original score files')
    offers=0;identities=set();hashes={}
    for path in paths:
        row=json.loads(path.read_text());source,raw=lookup[row['source']['file']]
        if row['source']['sha256']!=source['sha256'] or row['targets']!=len(raw)-64:raise ValueError('source/target denominator mismatch')
        identity=(row['predictor'],row.get('cell',row['corpus']),row.get('seed'),source['file'])
        if identity in identities:raise ValueError('duplicate score record')
        identities.add(identity)
        if len(row['offers'])!=512 or len({o['position'] for o in row['offers']})!=512:raise ValueError('missing unique offers')
        for suggestion in row['offers']:
            generated=bytes.fromhex(suggestion['generated_hex']);gold=raw[suggestion['position']:suggestion['position']+8]
            if len(generated)!=8 or len(gold)!=8:raise ValueError('truncated generated/gold bytes')
            prefix=0
            for a,b in zip(generated,gold):
                if a!=b:break
                prefix+=1
            if prefix!=suggestion['longest_correct_prefix']:raise ValueError('prefix scoring mismatch')
            for n,correct in suggestion['exact'].items():
                if correct!=int(prefix>=int(n)):raise ValueError('exact completion scoring mismatch')
            offers+=1
        hashes[path.name]=sha(path)
    result={'study':'CODEVALUE004','status':'PASS_ORIGINAL_ARTIFACT_REPLAY',
        'checkpoints_verified':120,'selections_verified':24,'original_score_files_verified':280,
        'saved_suggestions_recomputed':offers,'new_predictions':0,'optimizer_updates':0,'additional_gpu_minutes':0,
        'physical_GPU_count':runtime['gpu_count'],'score_sha256':hashes,'results_sha256':sha(out/'RESULTS.json'),
        'independently_qualified':False,'protected_TEST':'NOT_OPENED'}
    atomic_json(out/'VERIFICATION.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='score_sha256'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);a=p.parse_args();verify(a.out,a.data)
