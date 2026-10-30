#!/usr/bin/env python3
"""Original from-scratch fits; fixed DEV-only checkpoint selection, no transfer reads."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import resource
import time

import numpy as np
import torch
from torch.nn import functional as F

from protocol004 import (ARMS,CELLS,SEEDS,HERE,ROOT,MILESTONES,PARAMS,STEPS,atomic_json,batches,corpus_docs,cpu_tree,
                         folder_name,learning_rate,make,path_for,planned_hashes,read_sources,
                         scoring,select_checkpoint,sha,update_hash)

def fit(arm,cell,seed,data,out,device='cuda:0'):
    if arm not in ARMS or cell not in CELLS or seed not in SEEDS:raise ValueError("unregistered fit")
    from scripts.validate_research_parity import check_manifest
    manifest=json.loads((ROOT/'research/mathematical-engineering/codevalue004/FAIR_COMPARISON.json').read_text())
    check_manifest(manifest,'preflight')
    receipt,docs=corpus_docs(data,cell)
    if receipt['cohort_sha256']!=manifest['real_world_source']['sha256']:raise ValueError('source cohort changed')
    _,dev=read_sources(data,{'SELECT_DEV'})
    expected=planned_hashes(docs,seed)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device.startswith('cuda'):torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    model=make(arm,device).train();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    folder=out/folder_name(arm,cell,seed)
    if folder.exists():raise FileExistsError('refuse to overwrite original fit')
    folder.mkdir(parents=True)
    if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    order=hashlib.sha256();losses=[];curve=[];started=time.monotonic();cpu_started=time.process_time()
    for step,rows,sampler_state in batches(docs,seed):
        lr=learning_rate(cell,step)
        for group in optimizer.param_groups:group['lr']=lr
        update_hash(order,rows)
        z=torch.tensor([list(b) for _,_,b in rows],dtype=torch.long,device=device)
        logits=model(z[:,:-1]);loss=F.cross_entropy(logits.reshape(-1,256),z[:,1:].reshape(-1))
        optimizer.zero_grad(set_to_none=True);loss.backward()
        if any(p.grad is None for p in model.parameters()):raise ValueError('disconnected trained parameters')
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        if not torch.isfinite(loss):raise ValueError('nonfinite FIT loss')
        optimizer.step();losses.append(float(loss.detach()));done=step+1
        if done%160==0:print(json.dumps({'arm':arm,'cell':cell,'seed':seed,'step':done,
              'last160_fit_loss':float(np.mean(losses[-160:])),'lr':lr,
              'elapsed_s':time.monotonic()-started}),flush=True)
        if done not in MILESTONES:continue
        if order.hexdigest()!=expected[done]:raise ValueError('paired sample replay mismatch')
        numpy_rng=np.random.get_state()
        cp={'study':'CODEVALUE004','arm':arm,'cell':cell,'seed':seed,'steps':done,'params':PARAMS[arm],
            'source_cohort_sha256':receipt['cohort_sha256'],'sample_sha256':order.hexdigest(),
            'register_sha256':sha(HERE/'REGISTER.json'),'model':cpu_tree(model.state_dict()),
            'optimizer':cpu_tree(optimizer.state_dict()),'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if device.startswith('cuda') else [],
            'python_rng':random.getstate(),'numpy_rng':(numpy_rng[0],numpy_rng[1].tolist(),*numpy_rng[2:]),
            'sampler_state':sampler_state,'next_update':done,'loss_history':losses.copy(),'last_lr':lr}
        dest=path_for(out,arm,cell,seed,done)
        torch.save(cp,dest.with_suffix('.tmp'));dest.with_suffix('.tmp').replace(dest)
        saved_cpu=torch.get_rng_state();saved_cuda=torch.cuda.get_rng_state_all() if device.startswith('cuda') else []
        cold=make(arm,device).eval();loaded=torch.load(dest,map_location='cpu',weights_only=True)
        cold.load_state_dict(loaded['model'],strict=True);model.eval()
        with torch.inference_mode():
            delta=float((cold(z[:1,:32])-model(z[:1,:32])).abs().max())
            poisoned=z[:1,:32].clone();poisoned[:,16:]=(poisoned[:,16:]+127)%256
            future_delta=float((model(z[:1,:32])[:,:16]-model(poisoned)[:,:16]).abs().max())
        if not np.isfinite(delta) or not np.isfinite(future_delta) or delta>1e-6 or future_delta>1e-5:raise ValueError('cold replay or causal prefix failure')
        del cold,loaded
        scores=[]
        for row,raw in dev:
            score=scoring.neural_file(model,raw,device,completion=False)
            score.update(source=row,arm=arm,cell=cell,seed=seed,step=done)
            atomic_json(folder/f'DEV-{done:06d}-{row["file"]}.json',score);scores.append(score)
        targets=sum(r['targets'] for r in scores)
        metric={'step':done,'nll':sum(r['nll_sum'] for r in scores)/targets,
                'accuracy':sum(r['top1'] for r in scores)/targets,'targets':targets}
        curve.append(metric)
        torch.set_rng_state(saved_cpu)
        if saved_cuda:torch.cuda.set_rng_state_all(saved_cuda)
        model.train()
        rec={'study':'CODEVALUE004','arm':arm,'cell':cell,'seed':seed,'steps':done,'targets':done*1024,
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'deployed_parameters':sum(p.numel() for p in model.parameters()),
            'sample_sha256':order.hexdigest(),'checkpoint_sha256':sha(dest),
            'cold_replay_max_abs_difference':delta,'future_poison_max_abs_difference':future_delta,
            'source_cohort_sha256':receipt['cohort_sha256'],'elapsed_seconds':time.monotonic()-started,
            'cpu_seconds':time.process_time()-cpu_started,'last_lr':lr,
            'peak_rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated() if saved_cuda else 0,
            'training_loss_last160':float(np.mean(losses[-160:])),
            'training_flops':'UNMATCHED_INCOMPLETE_OPERATOR_COVERAGE','protected_TEST':'NOT_OPENED'}
        atomic_json(dest.with_suffix('.json'),rec)
        atomic_json(folder/'DEV_CURVE.json',curve)
        print(json.dumps({'stage':'DEV_checkpoint','arm':arm,'cell':cell,'seed':seed,**metric}),flush=True)
    chosen=select_checkpoint(curve)
    selection={'arm':arm,'cell':cell,'seed':seed,'selected_step':chosen,
        'selected_checkpoint_sha256':sha(path_for(out,arm,cell,seed,chosen)),
        'selection_rule':'minimum full SELECT_DEV target-weighted NLL; tie goes to earlier update',
        'transfer_scoring_performed':False,'candidate_curve':curve}
    atomic_json(folder/'SELECTION.json',selection)
    return selection

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--tasks',type=Path,required=True)
    a=p.parse_args()
    for task in json.loads(a.tasks.read_text()):fit(*task,a.data,a.out)
