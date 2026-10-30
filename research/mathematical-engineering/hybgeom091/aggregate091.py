#!/usr/bin/env python3
"""Read-only complete matrix + original checkpoint/optimizer/source audit.

No training or protected evaluation. Failure is fail-closed and no arm may be
silently omitted. This is internal exploratory evidence, NOT comparative
scientific qualification because training FLOPs are deliberately unmatched.
"""
from __future__ import annotations
import argparse,hashlib,json,math,statistics,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'research/mathematical-engineering/hybgeom091'))
from model091 import ARMS,EXPECTED,HybridGeometryTAC091
from run091 import SEEDS,STEPS,BATCH,LENGTH,DEV_OWNERS,state_dict_hash,source_hash

def digest(raw):return hashlib.sha256(raw).hexdigest()

def verify(location: Path):
    report=[];sha_per_seed={}
    torch.set_num_threads(2)
    for seed in SEEDS:
        for arm in ARMS:
            path=location/f'{arm}-{seed}'
            data=json.loads((path/'RESULTS.json').read_text())
            raw=(path/'MODEL.pt').read_bytes()
            if data['checkpoint_sha256']!=digest(raw):raise ValueError('checkpoint SHA '+str(path))
            cp=torch.load(path/'MODEL.pt',map_location='cpu',weights_only=True)
            if data['study_id']!='HYBGEOM091' or data['arm']!=arm or data['seed']!=seed or cp['arm']!=arm or cp['seed']!=seed or cp['steps']!=STEPS:
                raise ValueError('arm/seed/checkpoint mismatch')
            if cp['model_source_sha256']!=source_hash() or data['model_source_sha256']!=source_hash():
                raise ValueError('code/source altered')
            if cp['schedule']['sample_order_sha256']!=data['actual_sample_order_sha256'] or cp['schedule']['sample_bytes_sha256']!=data['actual_sample_bytes_sha256']:
                raise ValueError('actual schedule mismatch')
            if cp['schedule']['targets']!=STEPS*BATCH*LENGTH or data['training_targets']!=STEPS*BATCH*LENGTH:
                raise ValueError('fit budget mismatch')
            if seed not in sha_per_seed:sha_per_seed[seed]=cp['schedule']
            if cp['schedule']!=sha_per_seed[seed]:raise ValueError('per-seed arm sampler parity failed')
            model=HybridGeometryTAC091(arm).eval();model.load_state_dict(cp['state'],strict=True)
            if model.counts()!={'trainable':EXPECTED[arm],'deployed':EXPECTED[arm]} or model.counts()!=data['counts']:
                raise ValueError('original 100k parameters changed')
            if state_dict_hash(model)!=data['weights_hash']:raise ValueError('checkpoint weights hash')
            if len(cp['optimizer']['state'])!=len(list(model.parameters())):
                raise ValueError('optimizer missing parameter states')
            if not all(int(v['step'])==STEPS for v in cp['optimizer']['state'].values()):
                raise ValueError('optimizer step count mismatch')
            if len(data['exposed_DEV'])!=2 or {x['owner'] for x in data['exposed_DEV']}!=set(DEV_OWNERS):
                raise ValueError('DEV owner identity')
            for d in data['exposed_DEV']:
                if len(d['tokens'])!=224 or any(t['ordinal']!=i+32 for i,t in enumerate(d['tokens'])):
                    raise ValueError('original prequential prediction stream')
                if not math.isclose(sum(t['nll'] for t in d['tokens'])/224,d['nll'],abs_tol=1e-7):
                    raise ValueError('DEV NLL aggregate incorrect')
            report.append(data)
    paired={}
    baseline={z['seed']:statistics.mean(d['nll'] for d in z['exposed_DEV']) for z in report if z['arm']=='original'}
    for arm in ARMS:
        cases=[z for z in report if z['arm']==arm]
        if len(cases)!=5:raise ValueError('incomplete five seeds')
        diffs=[statistics.mean(d['nll'] for d in z['exposed_DEV'])-baseline[z['seed']] for z in cases]
        mu=statistics.mean(diffs)
        sd=statistics.stdev(diffs) if len(diffs)>1 else 0.
        halfwidth=2.776445*sd/math.sqrt(len(diffs))
        paired[arm]={'mean_nll':statistics.mean(statistics.mean(d['nll'] for d in z['exposed_DEV']) for z in cases),
            'mean_accuracy':statistics.mean(statistics.mean(d['accuracy'] for d in z['exposed_DEV']) for z in cases),
            'delta_nll_vs_original':mu,'seed_paired_95pct_t_interval':[mu-halfwidth,mu+halfwidth],
            'wins_vs_original':sum(x<0 for x in diffs),'seed_deltas':diffs,
            'fit_cpu_seconds_mean':statistics.mean(z['fit_process_seconds'] for z in cases),
            'dev_cpu_ms_per_byte_mean':statistics.mean(z['dev_latency']['cpu_mean_ms_per_step'] for z in cases)}
    return {'study':'HYBGEOM091','status':'INTERNAL_EXPLORATORY_TRAINED_MATCHED_EXPOSURE_NOT_MATCHED_TRAINING_FLOPS',
        'checkpoint_count':len(report),'seed_count':len(SEEDS),'source':'published original CPython genuine FIT and exposed DEV',
        'dev_targets_per_arm':len(SEEDS)*len(DEV_OWNERS)*224,
        'note':'two DEV owners reused in each seed; t interval treats 5 seeds as independent only',
        'replay_certification':'source_code_weights_optimizer_and_metrics_integrity_checked; no original independent full inference replay',
        'gpu_minutes':0,'protected_TEST':'UNOPENED',
        'actual_elapsed_timestamps':False,'context_specific_human_semantics':False,
        'matched_training_FLOPs':False,'full_PNDS_proof':False,'paired':paired}

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable summary destination')
    result=verify(a.directory)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({a:{k:v for k,v in x.items() if k in ('mean_nll','delta_nll_vs_original')} for a,x in result['paired'].items()},sort_keys=True))
if __name__=='__main__':main()
