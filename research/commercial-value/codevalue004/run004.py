#!/usr/bin/env python3
"""Two verified T4s, balanced frozen tasks and one hard global quota deadline."""
import argparse
import json
import os
import platform
import numpy as np
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch
from protocol004 import ARMS,CELLS,HERE,ROOT,SEEDS,atomic_json


def assignments():
    tasks=[(arm,cell,seed) for seed in SEEDS for cell in CELLS for arm in ARMS]
    tasks.sort(key=lambda t:0 if t[0]=='tac' else 1)
    loads=[0.,0.];result=[[],[]]
    for task in tasks:
        gpu=min(range(2),key=lambda i:loads[i])
        result[gpu].append(task);loads[gpu]+=353.47 if task[0]=='tac' else 103.80
    return result

def stop(child):
    if child.poll() is not None:return
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=8)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)

def run(data,out,max_seconds=7020):
    from scripts.validate_research_parity import check_manifest
    if out.exists():raise FileExistsError('refuse to overwrite experiment output')
    out.mkdir(parents=True)
    manifest=json.loads((ROOT/'research/mathematical-engineering/codevalue004/FAIR_COMPARISON.json').read_text())
    check_manifest(manifest,'preflight')
    names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    physical=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'],text=True).splitlines()
    if len(names)!=2 or len(physical)!=2 or any('T4' not in n.upper() for n in names):raise RuntimeError('exact physical T4 pair required')
    if len({s.split(',')[0] for s in physical})!=2:raise RuntimeError('distinct physical GPUs required')
    for i in range(2):
        x=torch.ones(1,device=f'cuda:{i}');torch.cuda.synchronize(i)
        if x.item()!=1:raise RuntimeError('CUDA physical device probe failed')
    jobs=assignments()
    atomic_json(out/'TASK_ASSIGNMENTS.json',jobs)
    report={'study':'CODEVALUE004','status':'RUNNING','physical_gpus':physical,'gpu_count':2,
            'maximum_allocation_seconds':max_seconds,'protected_TEST':'NOT_OPENED',
            'commercial_viability_proved':False,'automatic_resubmission':False,
            'python_version':platform.python_version(),'torch_version':str(torch.__version__),
            'numpy_version':str(np.__version__),'cuda_version':str(torch.version.cuda),
            'cudnn_version':torch.backends.cudnn.version(),
            'matmul_allow_tf32':bool(torch.backends.cuda.matmul.allow_tf32),
            'cudnn_allow_tf32':bool(torch.backends.cudnn.allow_tf32)}
    atomic_json(out/'RUNTIME.json',report)
    children=[];streams=[];started=time.monotonic()
    def launch(cmd,name,env):
        f=(out/(name+'.log')).open('w');streams.append(f)
        c=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        children.append(c);return c
    def deadline():
        if time.monotonic()-started>=max_seconds:raise TimeoutError('allocation deadline reached; preserve incomplete evidence')
    try:
        for gpu,tasks in enumerate(jobs):
            path=out/f'gpu{gpu}-tasks.json';atomic_json(path,tasks)
            launch([sys.executable,str(HERE/'fit004.py'),'--data',str(data),'--out',str(out),'--tasks',str(path)],
                   f'worker-gpu{gpu}',dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)))
        while any(c.poll() is None for c in children):
            deadline()
            if any(c.poll() not in (0,None) for c in children):raise RuntimeError('training worker failed; no automatic retry')
            time.sleep(2)
        if any(c.returncode for c in children):raise RuntimeError('training worker failed')
        evaluation=launch([sys.executable,str(HERE/'evaluate004.py'),'--data',str(data),'--out',str(out)],
                          'evaluation',dict(os.environ,CUDA_VISIBLE_DEVICES='0'))
        while evaluation.poll() is None:deadline();time.sleep(2)
        if evaluation.returncode:raise RuntimeError('transfer evaluation failed; preserve all original outputs')
        report['status']='COMPLETE_INTERNAL_RESEARCH'
    except BaseException as exc:
        report.update(status='INCOMPLETE_TIMEOUT' if isinstance(exc,TimeoutError) else 'FAILED',error=str(exc))
        for c in children:stop(c)
        raise
    finally:
        for f in streams:f.close()
        report.update(elapsed_seconds=time.monotonic()-started,
            allocated_gpu_minutes_upper_bound=2*(time.monotonic()-started)/60,
            children=[{'pid':c.pid,'exit_code':c.poll()} for c in children])
        atomic_json(out/'RUNTIME.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.data,a.out)
