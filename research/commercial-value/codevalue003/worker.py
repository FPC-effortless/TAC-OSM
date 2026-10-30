#!/usr/bin/env python3
"""One physical-T4 worker; original-schedule paired from-scratch fits."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import resource
import time

import numpy as np
import torch
from torch.nn import functional as F

from core import (ARMS,MILESTONES,ROOT,REG,atomic_json,checkpoint_path,cpu_tree,
                  make,planned_digests,read_sources,sampler,sha,update_digest)

def fit(arm,seed,data,out,device,manifest):
    from scripts.validate_research_parity import check_manifest
    gate=check_manifest(manifest,'preflight')
    if gate['eligible_for_architecture_superiority']:
        raise ValueError('exploratory experiment cannot qualify superiority')
    receipt,rows=read_sources(data,{'FIT'})
    if receipt['cohort_sha256']!=manifest['real_world_source']['sha256']:
        raise ValueError('source receipt differs from frozen manifest')
    docs={r['repo']+':'+r['path']:raw for r,raw in rows}
    expected=planned_digests(docs,seed)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device.startswith('cuda'):torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    model=make(arm,device).train()
    params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    registered=next(x for x in manifest['models'] if x['name']==arm and x['seed']==seed)
    if params!=registered['trainable_parameters']:raise ValueError('live model count mismatch')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    folder=out/f'{arm}-{seed}'
    if folder.exists():raise FileExistsError('refuse to overwrite original run')
    folder.mkdir(parents=True)
    h=hashlib.sha256(); start=time.monotonic();cpu_start=time.process_time()
    losses=[]
    for step,batch,sampler_state in sampler(docs,seed):
        update_digest(h,batch)
        z=torch.tensor([list(raw) for _,_,raw in batch],dtype=torch.long,device=device)
        logits=model(z[:,:-1])
        loss=F.cross_entropy(logits.reshape(-1,256),z[:,1:].reshape(-1))
        optimizer.zero_grad(set_to_none=True);loss.backward()
        missing=[n for n,p in model.named_parameters() if p.grad is None]
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        if missing or not torch.isfinite(loss):raise ValueError('invalid loss or missing gradients')
        optimizer.step();losses.append(float(loss.detach()))
        done=step+1
        if done%80==0:
            print(json.dumps({'arm':arm,'seed':seed,'updates':done,
                              'last80_loss':sum(losses[-80:])/80,
                              'elapsed_s':round(time.monotonic()-start,2)}),flush=True)
        if done not in MILESTONES:continue
        if h.hexdigest()!=expected[done]:raise ValueError('sample replay mismatch')
        numpy_rng=np.random.get_state()
        cp={'study':'CODEVALUE003','arm':arm,'seed':seed,'steps':done,'params':params,
            'source_cohort_sha256':receipt['cohort_sha256'],'sample_sha256':h.hexdigest(),
            'register_sha256':sha(Path(__file__).parent/'REGISTER.json'),
            'model':cpu_tree(model.state_dict()),'optimizer':cpu_tree(optimizer.state_dict()),
            'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if device.startswith('cuda') else [],
            'python_rng':random.getstate(),
            'numpy_rng':(numpy_rng[0],numpy_rng[1].tolist(),*numpy_rng[2:]),
            'sampler_state':sampler_state,'next_update':done,
            'loss_history':losses.copy()}
        dest=checkpoint_path(out,arm,seed,done)
        torch.save(cp,dest.with_suffix('.tmp'));dest.with_suffix('.tmp').replace(dest)
        # Cold load on same device; the constructor consumes global RNG, so
        # restore CPU/CUDA RNG after replay to leave training unchanged.
        cpu_rng=torch.get_rng_state()
        cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else []
        loaded=torch.load(dest,map_location='cpu',weights_only=True)
        cold=make(arm,device).eval();cold.load_state_dict(loaded['model'],strict=True)
        model.eval()
        with torch.no_grad():
            replay_delta=float((model(z[:1,:32])-cold(z[:1,:32])).abs().max())
        if replay_delta>1e-6:raise ValueError('cold checkpoint prediction replay failed')
        del cold,loaded
        torch.set_rng_state(cpu_rng)
        if cuda_rng:torch.cuda.set_rng_state_all(cuda_rng)
        model.train()
        actual_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
        if actual_params!=params:raise ValueError('postfit model count changed')
        rec={'study':'CODEVALUE003','arm':arm,'seed':seed,'steps':done,'targets':done*1024,
             'trainable_parameters':params,'deployed_parameters':sum(p.numel() for p in model.parameters()),
             'checkpoint_sha256':sha(dest),'sample_sha256':h.hexdigest(),
             'source_cohort_sha256':receipt['cohort_sha256'],
             'cold_replay_max_abs_difference':replay_delta,
             'elapsed_seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu_start,
             'peak_rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated() if cuda_rng else 0,
             'training_flops':'UNMATCHED_INCOMPLETE_OPERATOR_COVERAGE',
             'training_loss_last80':sum(losses[-80:])/min(80,len(losses)),
             'protected_TEST':'NOT_OPENED','qualification':'INTERNAL_NOT_INDEPENDENTLY_QUALIFIED'}
        atomic_json(dest.with_suffix('.json'),rec)
    return rec

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--arms',nargs='+',default=ARMS)
    a=p.parse_args()
    manifest=json.loads((ROOT/'research/mathematical-engineering/codevalue003/FAIR_COMPARISON.json').read_text())
    for seed in a.seeds:
        for arm in a.arms:fit(arm,seed,a.data,a.out,a.device,manifest)

if __name__=='__main__':main()
