#!/usr/bin/env python3
"""Private offline dual-T4 training with a hard end-to-end allocation deadline."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch

from core import ROOT,atomic_json

HERE=Path(__file__).resolve().parent

def stop_child(child):
    if child.poll() is not None:return
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)

def run(data,out,max_seconds=7140):
    from scripts.validate_research_parity import check_manifest
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'research/mathematical-engineering/codevalue003/FAIR_COMPARISON.json').read_text())
    check_manifest(manifest,'preflight')
    names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    physical=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'],text=True).splitlines()
    if len(names)!=2 or len(physical)!=2 or any('T4' not in n.upper() for n in names):
        raise RuntimeError('exactly two physically observed T4s required; no fallback')
    if len({p.split(',')[0].strip() for p in physical})!=2:raise RuntimeError('distinct physical UUIDs required')
    for i in range(2):
        x=torch.ones(1,device=f'cuda:{i}');torch.cuda.synchronize(i)
        if x.item()!=1:raise RuntimeError('CUDA device probe failed')
    report={'study':'CODEVALUE003','status':'RUNNING','physical_gpus':physical,
            'maximum_allocation_seconds':max_seconds,'gpu_count':2,'children':[],
            'protected_TEST':'NOT_OPENED','commercial_viability_proved':False}
    atomic_json(out/'RUNTIME.json',report)
    started=time.monotonic();children=[];streams=[]
    def launch(cmd,name,env):
        stream=(out/(name+'.log')).open('w');streams.append(stream)
        child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(child)
        return child
    def check_deadline():
        if time.monotonic()-started>=max_seconds:raise TimeoutError('allocation ceiling reached; incomplete artifacts retained')
    try:
        for gpu,seeds in ((0,[1701,1703]),(1,[1702])):
            cmd=[sys.executable,str(HERE/'worker.py'),'--data',str(data),'--out',str(out),
                 '--seeds',*map(str,seeds)]
            launch(cmd,f'worker-gpu{gpu}',dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),
                                              TAC_OSM_PHYSICAL_T4_SLOT=str(gpu)))
        while any(c.poll() is None for c in children):
            check_deadline()
            if any(c.poll() not in (None,0) for c in children):raise RuntimeError('training worker failed; see worker logs')
            time.sleep(2)
        if any(c.returncode for c in children):raise RuntimeError('training worker failed')
        evaluation=launch([sys.executable,str(HERE/'evaluate_run.py'),'--data',str(data),
                           '--out',str(out)],'evaluation',dict(os.environ,CUDA_VISIBLE_DEVICES='0'))
        while evaluation.poll() is None:
            check_deadline();time.sleep(2)
        if evaluation.returncode:raise RuntimeError('evaluation failed; see evaluation.log')
        report['status']='COMPLETED_INTERNAL_RESEARCH'
    except BaseException as exc:
        report['status']='INCOMPLETE_TIMEOUT' if isinstance(exc,TimeoutError) else 'FAILED'
        report['error']=str(exc)
        for c in children:stop_child(c)
        raise
    finally:
        for stream in streams:stream.close()
        report['elapsed_seconds']=time.monotonic()-started
        report['allocated_gpu_minutes_upper_bound']=2*report['elapsed_seconds']/60
        report['children']=[{'pid':c.pid,'exit_code':c.poll()} for c in children]
        atomic_json(out/'RUNTIME.json',report)
        print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.data,a.out)
