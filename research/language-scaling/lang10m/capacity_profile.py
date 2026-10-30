#!/usr/bin/env python3
"""Bounded two-T4 capacity measurements on authenticated FIT token shapes.

No optimizer is constructed or stepped. Random initial weights are unchanged.
Loss values are numerical diagnostics, not trained language-quality results.
"""
import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from profile_no_fit import physical_t4s,load_model

HERE=Path(__file__).resolve().parent
SHAPES=((1,128),(2,128),(4,128),(8,128),(2,256),(4,256),(2,512),(4,512),(1,1024),(2,1024))
PRECISIONS=('fp32','amp_fp16')
WARMUPS=2
REPEATS=5


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def json_safe(value):
    if isinstance(value,np.generic):return json_safe(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
    return value

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(json_safe(value),indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(path)

def state_hash(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode()+b'\0'+value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def shards():
    loads=[0,0];jobs=[[],[]]
    for batch,context in sorted(SHAPES,key=lambda s:-(s[0]*s[1])):
        gpu=min(range(2),key=lambda i:loads[i])
        jobs[gpu].extend([{'batch':batch,'context':context,'precision':p,'seed':540001+gpu} for p in PRECISIONS])
        loads[gpu]+=batch*context
    return jobs

def sample(arrays,batch,context,seed=540001):
    rng=np.random.default_rng(seed+batch*context)
    names=sorted(k for k,v in arrays.items() if len(v)>=context+1)
    if not names:raise ValueError('no FIT document supports registered context')
    rows=[];receipt=[]
    for _ in range(batch):
        name=names[int(rng.integers(len(names)))];a=arrays[name]
        offset=int(rng.integers(len(a)-context))
        row=a[offset:offset+context+1].astype(np.int64)
        if ((row<0)|(row>=4096)).any():raise ValueError('token IDs outside4096 vocabulary')
        rows.append(row);receipt.append({'array':name,'offset':offset,'tokens_sha256':hashlib.sha256(row.tobytes()).hexdigest()})
    z=np.stack(rows)
    return torch.from_numpy(z[:,:-1].copy()),torch.from_numpy(z[:,1:].copy()),receipt

def one_case(task,arrays):
    batch,context,precision=task['batch'],task['context'],task['precision']
    if (batch,context) not in SHAPES or precision not in PRECISIONS:raise ValueError('unregistered shape/mode')
    seed=int(task["seed"])
    if seed not in (540001,540002):raise ValueError("unregistered profile seed")
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(2)
    torch.cuda.set_device(0);torch.cuda.empty_cache()
    model=load_model()().to('cuda:0').train()
    params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if params!=10001188 or sum(p.numel() for p in model.parameters())!=params:raise ValueError('actual10M count changed')
    before=state_hash(model)
    x,y,owners=sample(arrays,batch,context,seed);x=x.cuda();y=y.cuda()
    def autocast():return torch.autocast('cuda',dtype=torch.float16) if precision=='amp_fp16' else contextlib.nullcontext()
    # Compare precision on the same device/weights/input before timing; release
    # reference logits so they do not inflate the forward/backward memory peak.
    with torch.no_grad():
        reference=model(x);reference_loss=F.cross_entropy(reference.reshape(-1,4096),y.reshape(-1))
        with autocast():
            candidate=model(x);candidate_loss=F.cross_entropy(candidate.reshape(-1,4096),y.reshape(-1))
        mean_delta=float((reference-candidate.float()).abs().mean())
        max_delta=float((reference-candidate.float()).abs().max())
        relative_loss_delta=abs(float(candidate_loss)-float(reference_loss))/max(abs(float(reference_loss)),1e-12)
        dtype=str(candidate.dtype)
    del reference,candidate,reference_loss,candidate_loss
    def backward():
        model.zero_grad(set_to_none=True)
        with autocast():
            logits=model(x);loss=F.cross_entropy(logits.reshape(-1,4096),y.reshape(-1))
        loss.backward()
        return loss
    for _ in range(WARMUPS):
        loss=backward();del loss
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    elapsed=[];losses=[]
    for _ in range(REPEATS):
        torch.cuda.synchronize();started=time.perf_counter();loss=backward();torch.cuda.synchronize()
        elapsed.append(time.perf_counter()-started);losses.append(float(loss.detach()));del loss
    missing=[n for n,p in model.named_parameters() if p.grad is None]
    finite=not missing and bool(torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all())
    zeros=int(torch.stack([(p.grad==0).all() for p in model.parameters() if p.grad is not None]).sum())
    allocated=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
    after=state_hash(model)
    if before!=after:raise RuntimeError('no-update profile changed model weights')
    parity=bool(np.isfinite(mean_delta) and np.isfinite(relative_loss_delta) and mean_delta<=.02 and relative_loss_delta<=.02)
    ok=finite and all(np.isfinite(losses)) and parity
    result={**task,'status':'PASS_NO_UPDATE_PROFILE' if ok else 'FAIL_NUMERICAL_OR_PRECISION_GATE',
        'trainable_parameters':params,'deployed_parameters':params,'weight_sha256_before':before,
        'weight_sha256_after':after,'weights_unchanged':True,'optimizer_steps':0,'amp_loss_scaling':'NONE_NO_OPTIMIZER_PROFILE','nonfinite_numeric_fields_serialized_as_null':True,'warmups':WARMUPS,'measured_repeats':REPEATS,
        'elapsed_forward_backward_seconds':elapsed,
        'median_tokens_per_second_forward_backward':float(batch*context/np.median(elapsed)),
        'peak_cuda_allocated_bytes':int(allocated),'peak_cuda_reserved_bytes':int(reserved),
        'additional_AdamW_moment_bytes_estimate_NOT_ALLOCATED':int(params*8),
        'observed_logit_dtype':dtype,'missing_gradients':missing,'all_gradients_finite':bool(finite),
        'zero_gradient_parameter_tensors_on_fixture':zeros,
        'mean_logit_abs_precision_difference':mean_delta,'max_logit_abs_precision_difference':max_delta,
        'relative_loss_precision_difference':relative_loss_delta,'precision_gate':parity,
        'untrained_FIT_loss_numerical_diagnostic_NOT_LANGUAGE_QUALITY':losses,
        'source_windows':owners,'full_model_streaming':False}
    del model,x,y;gc.collect();torch.cuda.empty_cache()
    return result

def worker(tasks,tokens,output):
    if torch.cuda.device_count()!=1 or 'T4' not in torch.cuda.get_device_name(0):raise RuntimeError('one isolated physical T4 required per worker')
    with np.load(tokens,allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
    results=[]
    for task in json.loads(tasks.read_text()):
        try:result=one_case(task,arrays)
        except (RuntimeError,ValueError) as exc:
            result={**task,'status':'FAIL','exception':type(exc).__name__,'message':str(exc)[:600],'optimizer_steps':0}
            gc.collect();torch.cuda.empty_cache()
        results.append(result)
        atomic(output,{'status':'WORKER_PROGRESS','physical_slot':os.environ['LANG10M_PHYSICAL_SLOT'],
            'results':results,'optimizer_steps':0})
        print(json.dumps({'stage':'case_completed',**{k:v for k,v in result.items() if k in ('batch','context','precision','status','median_tokens_per_second_forward_backward','peak_cuda_allocated_bytes')}}),flush=True)
    atomic(output,{'status':'WORKER_MATRIX_COMPLETE','physical_slot':os.environ['LANG10M_PHYSICAL_SLOT'],
        'results':results,'optimizer_steps':0})

def stop(child):
    if child.poll() is not None:return
    child.terminate()
    try:child.wait(timeout=8)
    except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)

def run(tokens,output,max_seconds=300):
    if output.exists():raise FileExistsError('do not overwrite original GPU evidence')
    output.mkdir(parents=True)
    physical=physical_t4s();jobs=shards();started=time.monotonic();children=[];streams=[]
    recipe=json.loads((tokens.parent/'TOKENS_RECEIPT.json').read_text())
    if recipe['partition']!='FIT' or sha(tokens)!=recipe['arrays_sha256']:raise ValueError('FIT token arrays changed')
    report={'study':'LANG10M001_CAPACITY','status':'RUNNING_NO_UPDATE_PROFILE','evidence_level':'UNTRAINED_ENGINEERING_ONLY',
        'physical_T4s':physical,'torch_version':str(torch.__version__),'cuda_version':str(torch.version.cuda),
        'model_sha256':sha(HERE/'model.py'),'token_arrays_sha256':sha(tokens),
        'tokenizer_sha256':recipe['tokenizer_sha256'],'FIT_manifest_sha256':recipe['manifest_sha256'],
        'global_deadline_seconds':max_seconds,'shapes':SHAPES,'precisions':PRECISIONS,
        'optimizer_steps':0,'heldout_inputs':0,'automatic_retry':False,'trained_language_quality':'NOT_MEASURED'}
    atomic(output/'CAPACITY.json',report)
    try:
        for gpu,tasks in enumerate(jobs):
            path=output/f'tasks-{gpu}.json';atomic(path,tasks)
            log=(output/f'worker-{gpu}.log').open('w');streams.append(log)
            cmd=[sys.executable,str(HERE/'capacity_profile.py'),'--worker','--tasks',str(path),'--tokens',str(tokens),'--output',str(output/f'gpu-{gpu}.json')]
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,
                CUDA_VISIBLE_DEVICES=str(gpu),LANG10M_PHYSICAL_SLOT=str(gpu)))
            children.append(child)
        while any(c.poll() is None for c in children):
            if time.monotonic()-started>=max_seconds:raise TimeoutError('capacity profile deadline reached')
            if any(c.poll() not in (None,0) for c in children):raise RuntimeError('capacity worker failed; original logs retained')
            time.sleep(1)
        if any(c.returncode for c in children):raise RuntimeError('capacity worker failed')
        results=[]
        for gpu in range(2):
            data=json.loads((output/f'gpu-{gpu}.json').read_text())
            if data['status']!='WORKER_MATRIX_COMPLETE':raise RuntimeError('partial worker matrix')
            results.extend(dict(r,physical_slot=gpu,physical_uuid=physical[gpu][0]) for r in data['results'])
        if len(results)!=20:raise RuntimeError('incomplete20-case profile')
        report['results']=results
        report['status']='PASS_CAPACITY_MATRIX' if all(r['status']=='PASS_NO_UPDATE_PROFILE' for r in results) else 'MATRIX_COMPLETE_WITH_FAILED_CASES'
    except BaseException as exc:
        report.update(status='INCOMPLETE_TIMEOUT' if isinstance(exc,TimeoutError) else 'FAILED',error=str(exc))
        for c in children:stop(c)
        raise
    finally:
        for f in streams:f.close()
        report.update(elapsed_seconds=time.monotonic()-started,allocated_GPU_minutes_upper_bound=2*(time.monotonic()-started)/60,
            worker_exit_codes=[c.poll() for c in children])
        atomic(output/'CAPACITY.json',report)
        print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tokens',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--tasks',type=Path)
    a=p.parse_args();worker(a.tasks,a.tokens,a.output) if a.worker else run(a.tokens,a.output)
