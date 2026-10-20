#!/usr/bin/env python3
"""ME005 authentic MetroPT adapter training; zero original ME004 backbone refits.
Diagnostic TRAIN/validation split was previously exposed; not scientific TEST.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from scripts.me004_native_real_forecast import acquire,prepare,score
from scripts.me005_unified_native_tac import UnifiedNativeTAC,load_me004_backbone

ADAPTER_SEEDS=(505001,505002)
BACKBONE_SEEDS=(240401,240402)
ARMS=('residual_only','unified_all')
UPDATES=240
BATCH=64

def evaluate(model,cohort,threshold):
    model.eval();x,y=cohort
    with torch.inference_mode():
        pred=torch.cat([model(x[i:i+128]) for i in range(0,len(x),128)])
    return score(pred,y,threshold)

def train_one(arm,seed,bseed,train,val,threshold,root,original,updates):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    ck=original/'checkpoints'/f'tac_two_timescale-{bseed}.pt'
    base=load_me004_backbone(ck,original/'results.json',bseed)
    active=(arm=='unified_all')
    net=UnifiedNativeTAC(base,graph=active,plastic=active,operators=active,
                         rhythms=active,reasoning=active,freeze_base=True)
    initial=evaluate(net,val,threshold)
    recorded=json.loads((original/'results.json').read_text())
    expected=next(f for f in recorded['fits'] if f['model']=='tac_two_timescale' and f['seed']==bseed)
    if abs(initial['macro_mae']-expected['validation']['macro_mae'])>3e-6:
        raise AssertionError('original pretrained forecast changed before training')
    trainable=[p for p in net.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=5e-4,weight_decay=1e-4)
    gen=torch.Generator().manual_seed(seed+100)
    xx,yy=train;history=[];start=time.perf_counter()
    for step in range(updates):
        net.train()
        ix=torch.randint(len(xx),(BATCH,),generator=gen)
        output=net(xx[ix],details=True)
        prediction=output['prediction']
        huber=F.smooth_l1_loss(prediction,yy[ix],beta=.2)
        variance=output['logvariance']
        error=(prediction-yy[ix]).detach() if False else prediction-yy[ix]
        nll=(.5*(error.square()*torch.exp(-variance)+variance)).mean()
        compute=output['compute_cost']
        loss=huber+.008*nll+.0005*compute
        if not bool(torch.isfinite(loss)):raise ArithmeticError('nonfinite loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm=torch.nn.utils.clip_grad_norm_(trainable,1.0)
        if not bool(torch.isfinite(gnorm)):raise ArithmeticError('nonfinite gradient')
        optimizer.step()
        if step+1 in (1,updates//2,updates):
            history.append({'step':step+1,'loss':float(loss.detach()),
                            'huber':float(huber.detach()),'nll':float(nll.detach()),
                            'compute':float(compute.detach())})
    measured=evaluate(net,val,threshold)
    output=root/'checkpoints'/f'{arm}-{seed}.pt'
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'arm':arm,'seed':seed,'base_seed':bseed,'updates':updates,
        'weights':net.state_dict(),'base_sha256':hashlib.sha256(ck.read_bytes()).hexdigest(),
        'baseline_original':expected['validation']},output)
    return {'arm':arm,'seed':seed,'base_seed':bseed,
        'trainable_params':sum(p.numel() for p in trainable),
        'total_params':sum(p.numel() for p in net.parameters()),
        'original_pretrained_mae':initial['macro_mae'],
        'validation':measured,'loss_samples':history,
        'wall_seconds':round(time.perf_counter()-start,2),
        'checkpoint':str(output.relative_to(root)),
        'checkpoint_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'original_pretrained_checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest()}

def run(output,original,updates):
    output=Path(output);original=Path(original);output.mkdir(parents=True,exist_ok=True)
    prior=json.loads((original/'results.json').read_text())
    if prior['study']!='ME004' or len(prior['fits'])!=6:raise ValueError('must use genuine ME004 source')
    for f in prior['fits']:
        ck=original/f['checkpoint']
        if hashlib.sha256(ck.read_bytes()).hexdigest()!=f['checkpoint_sha256']:
            raise ValueError('pretrained original artifact altered')
    vals,t,source=acquire(output)
    if source!=prior['source']:raise ValueError('authentic source provenance changed')
    train,val,threshold,cohort=prepare(vals,t)
    # Existing original real split ONLY — qualification cannot rely on its repeated use.
    if cohort!=prior['cohort']:raise ValueError('previously registered source split changed')
    output_data={'study':'ME005','evidence':'EXPLORATORY_EXISTING_REAL_TRAIN_SPLIT',
       'source':source,'cohort':cohort,'adapter_seeds':list(ADAPTER_SEEDS),
       'backbone_seeds':list(BACKBONE_SEEDS),'fits':[],
       'future_reserved_test_read':False,'pretrained_original_refits':0,
       'real_action_outcome_training':False,'architecture_promoted':False}
    for seed,bseed in zip(ADAPTER_SEEDS,BACKBONE_SEEDS):
        for arm in ARMS:
            fit=train_one(arm,seed,bseed,train,val,threshold,output,original,updates)
            output_data['fits'].append(fit)
            print(json.dumps({'arm':arm,'seed':seed,'metrics':fit['validation'],
                              'wall_seconds':fit['wall_seconds']},sort_keys=True),flush=True)
    (output/'results.json').write_text(json.dumps(output_data,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':'REAL_EXPLORATORY_DONE','original_refits':0,
        'adapter_fits':len(output_data['fits']),'promotion':False}),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='results/me005')
    p.add_argument('--original',default='results/me004')
    p.add_argument('--steps',type=int,default=UPDATES)
    a=p.parse_args();run(a.out,a.original,a.steps)
if __name__=='__main__':main()
