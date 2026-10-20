#!/usr/bin/env python3
"""ME005-C: recorded-only ablation of real-trained integrated TAC adapter.
Original train/validation source only; exactly ZERO fits and NO later TEST.
Ablation of learned modules after training is a diagnostic, not causal attribution.
"""
from __future__ import annotations
import json,hashlib,copy,time
from pathlib import Path
import torch
from scripts.me004_native_real_forecast import acquire,prepare,score
from scripts.me005_unified_native_tac import UnifiedNativeTAC,load_me004_backbone

SWITCHES=('has_graph','has_plastic','has_operators','has_rhythm','has_reasoning')

def full_prediction(net,data):
    net.eval()
    x,y=data
    with torch.inference_mode():
        pred=torch.cat([net(x[i:i+128]) for i in range(0,len(x),128)],dim=0)
    return pred

def main(base='results/me004',adapter='results/me005',out='results/me005-attribution'):
    torch.set_num_threads(1)
    base,adapter,out=Path(base),Path(adapter),Path(out)
    out.mkdir(parents=True,exist_ok=True)
    prior=json.loads((base/'results.json').read_text())
    record=json.loads((adapter/'results.json').read_text())
    if record['study']!='ME005' or len(record['fits'])!=4:
        raise ValueError('not frozen ME005 original artifact')
    if prior['study']!='ME004':raise ValueError('not frozen ME004')
    real,t,source=acquire(out)
    if source!=prior['source'] or source!=record['source']:
        raise ValueError('authentic source mismatch')
    train,val,threshold,cohort=prepare(real,t)
    if cohort!=record['cohort']:raise ValueError('exploratory source cohorts differ')
    summary={'study':'ME005_RECORDED_ONLY_POSTTRAIN_ABLATION','refits':0,
         'official_future_test_read':False,'authentic_source':source,
         'architecture_promoted':False,'limitations':'Post-training flag disabling is not causal training-attribution proof.',
         'fits':[]}
    for fit in record['fits']:
        if fit['arm']!='unified_all':continue
        cp=adapter/fit['checkpoint']
        if hashlib.sha256(cp.read_bytes()).hexdigest()!=fit['checkpoint_sha256']:
            raise ValueError('original full adapter checkpoint digest mismatch')
        saved=torch.load(cp,weights_only=True,map_location='cpu')
        if saved['seed']!=fit['seed'] or saved['arm']!='unified_all':
            raise ValueError('checkpoint metadata mismatch')
        backbone=load_me004_backbone(base/'checkpoints'/f"tac_two_timescale-{fit['base_seed']}.pt",
                                     base/'results.json',fit['base_seed'])
        model=UnifiedNativeTAC(backbone,graph=True,plastic=True,operators=True,
                               rhythms=True,reasoning=True,freeze_base=True)
        model.load_state_dict(saved['weights'],strict=True)
        baseline=full_prediction(model,val)
        measured=score(baseline,val[1],threshold)
        if abs(measured['macro_mae']-fit['validation']['macro_mae'])>3e-6:
            raise AssertionError('frozen source/checkpoint fails to replay original score')
        result={'seed':fit['seed'],'base_seed':fit['base_seed'],
             'original_full_macro_mae':measured['macro_mae'],
             'original_checkpoint_sha256':fit['checkpoint_sha256'],
             'ablations':{}}
        for field in SWITCHES:
            setattr(model,field,False)
            candidate=full_prediction(model,val)
            sc=score(candidate,val[1],threshold)
            diff=(candidate-baseline).abs()
            result['ablations'][field]={
                'macro_mae':sc['macro_mae'],
                'delta_mae_vs_full':sc['macro_mae']-measured['macro_mae'],
                'changed_mae':sc['changed_mae'],
                'max_absolute_prediction_delta':float(diff.max()),
                'mean_absolute_prediction_delta':float(diff.mean())}
            setattr(model,field,True)
        summary['fits'].append(result)
        print(json.dumps({'seed':fit['seed'],'ablations':result['ablations']},sort_keys=True),flush=True)
    if len(summary['fits'])!=2:raise ValueError('wrong full-model count')
    (out/'ablation.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':'POSTTRAIN_ABLATION_COMPLETE','models':len(summary['fits']),
                      'refits':0,'future_opened':False}),flush=True)

if __name__=='__main__':main()
