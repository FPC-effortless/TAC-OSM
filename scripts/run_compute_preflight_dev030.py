#!/usr/bin/env python3
"""Kaggle CPU/CUDA preflight on engineering fixtures, not research outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import torch.nn.functional as F
from tac_osm.development.compute_runtime import DeviceBoundary
from tac_osm.development.optimization_program import ARMS
from tac_osm.development.schema_maintenance import make_pair,ROLES
from tac_osm.development.delayed_maintenance import Action
from run_optimization_pilot_dev029 import build
from run_learned_schema_dev import public_stream_tensors,public_queries,labels


def fixture(index,history):
    cases=make_pair(303000+index,history,32,ROLES[index%4],'at',domain='dev030-engineering-fixture')
    inputs=public_stream_tensors([c.observations for c in cases])
    query=public_queries([c.tick for c in cases],[c.horizon for c in cases])
    receipts=public_stream_tensors([(c.world.act(Action('inspect',(index+b)%8)),) for b,c in enumerate(cases)])
    return inputs,query,receipts,labels(cases),[c.fingerprint for c in cases]


def update(model,optimizer,index,history,capture=False):
    inputs,query,receipts,target,_=fixture(index,history)
    target=target.to(model.device);model.train();optimizer.zero_grad(set_to_none=True)
    state=model.encode(*inputs);before=model.decode(state,query)
    after=model.decode(model.update(state,*(x[:,0] for x in receipts)),query)
    loss=(F.cross_entropy(before.reshape(-1,2),target.reshape(-1))+
          F.cross_entropy(after.reshape(-1,2),target.reshape(-1)))/2
    if not torch.isfinite(loss):raise AssertionError('nonfinite fixture loss')
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    if capture:
        with torch.no_grad():return model.decode(model.encode(*inputs),query).cpu()


def initialized(arm,device,seed=30303):
    model=DeviceBoundary(build(arm,seed),device)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.002,weight_decay=.0001)
    return model,optimizer


def measure(arm,device,history=512,warmup=2,steps=5,repetitions=3):
    samples=[];runtime=None
    for repetition in range(repetitions):
        model,optimizer=initialized(arm,device,30303+repetition)
        runtime=model.metadata()
        for index in range(warmup):update(model,optimizer,index,history)
        model.synchronize();begin=time.perf_counter()
        for index in range(warmup,warmup+steps):update(model,optimizer,index,history)
        model.synchronize();samples.append((time.perf_counter()-begin)/steps)
    return dict(runtime=runtime,seconds_per_update=samples,median_seconds_per_update=statistics.median(samples))


def parity(arm,history=512):
    cpu,cpu_optimizer=initialized(arm,'cpu');gpu,gpu_optimizer=initialized(arm,'cuda:0')
    inputs,query,_,_,fingerprints=fixture(2,history)
    with torch.no_grad():
        initial_cpu=cpu.decode(cpu.encode(*inputs),query).cpu()
        initial_gpu=gpu.decode(gpu.encode(*inputs),query).cpu()
    torch.testing.assert_close(initial_cpu,initial_gpu,rtol=1e-5,atol=1e-6)
    if not torch.equal(initial_cpu.argmax(-1),initial_gpu.argmax(-1)):
        raise AssertionError('initial categorical predictions differ')
    a,b=initial_cpu.softmax(-1)[...,1],initial_gpu.softmax(-1)[...,1]
    if not torch.equal(a[:,:,0]-a[:,:,1]>.25,b[:,:,0]-b[:,:,1]>.25):
        raise AssertionError('initial physical service choices differ')
    post_cpu=update(cpu,cpu_optimizer,2,history,capture=True)
    post_gpu=update(gpu,gpu_optimizer,2,history,capture=True)
    torch.testing.assert_close(post_cpu,post_gpu,rtol=1e-4,atol=1e-5)
    return dict(passed=True,initial_max_abs_error=float((initial_cpu-initial_gpu).abs().max()),
        post_update_max_abs_error=float((post_cpu-post_gpu).abs().max()),
        initial_categorical_predictions_exact=True,initial_service_choices_exact=True,
        fixture_fingerprints=fingerprints,
        limitation='One-step closeness does not imply identical CPU/CUDA training trajectories')


def recommendation(rows):
    if set(rows)!=set(ARMS) or any(not r.get('parity',{}).get('passed') for r in rows.values()):
        return dict(gpu_validated=False,prefer_gpu=False,reason='complete GPU parity evidence unavailable')
    cpu=sum(r['cpu']['median_seconds_per_update'] for r in rows.values())
    gpu=sum(r['cuda']['median_seconds_per_update'] for r in rows.values())
    return dict(gpu_validated=True,prefer_gpu=gpu<=cpu*.8,
        aggregate_cpu_over_gpu_speedup=cpu/gpu,
        reason='GPU candidate only when aggregate end-to-end update time falls by at least 20%; no scientific selection')


def run(output,cpu_only=False,smoke=False):
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():
        raise ValueError('freeze source before running preflight')
    if not cpu_only and not torch.cuda.is_available():
        raise RuntimeError('GPU preflight requested but CUDA unavailable; no silent fallback')
    protocol=ROOT/'research/dev030/protocol.json'
    result=dict(experiment='DEV-030 compute preflight',source=source,
        protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),smoke=smoke,
        gpu_validation_completed=False,test_accessed=False,scientific_training_run=False,arms={})
    output=Path(output)
    if output.exists():raise ValueError('preserve existing preflight evidence; choose a fresh output path')
    output.parent.mkdir(parents=True,exist_ok=True)
    for arm in ARMS:
        settings=dict(history=32,warmup=1,steps=2,repetitions=1) if smoke else {}
        row=dict(cpu=measure(arm,'cpu',**settings))
        if not cpu_only:
            row['parity']=parity(arm,32 if smoke else 512)
            row['cuda']=measure(arm,'cuda:0',**settings)
        result['arms'][arm]=row
        output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(arm=arm,completed=True,gpu_parity=not cpu_only)),flush=True)
    result['recommendation']=recommendation(result['arms'])
    result['gpu_validation_completed']=not cpu_only
    if smoke:
        result['recommendation']['prefer_gpu']=False
        result['recommendation']['reason']='Smoke timings cannot choose a study backend'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['recommendation']),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--cpu-only',action='store_true');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();run(a.output,a.cpu_only,a.smoke)
