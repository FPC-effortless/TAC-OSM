#!/usr/bin/env python3
"""Bounded unpruned six-arm tranche, independent processes, no recipe selection."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]


def commands(spec,output):
    return {arm:[sys.executable,str(ROOT/'scripts/run_optimization_study_dev029.py'),
        '--arm',arm,'--learning-rate',str(spec['learning_rate']),'--curriculum',spec['curriculum'],
        '--seed',str(spec['seed']),'--until',str(spec['until']),'--device',spec['devices'][arm],
        '--output',str(Path(output)/arm)] for arm in spec['arms']}


def run(output):
    spec_path=ROOT/'research/dev031/protocol.json';spec=json.loads(spec_path.read_text())
    output=Path(output)
    if output.exists():raise ValueError('preserve evidence; use fresh output')
    output.mkdir(parents=True)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():raise ValueError('freeze source first')
    tasks=commands(spec,output);deadline=time.monotonic()+spec['overall_wall_budget_seconds']
    manifest=dict(experiment='DEV-031',source=source,protocol_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        tasks=tasks,devices=spec['devices'],full_tuning_complete=False,recipes_selected=False,
        independent_full_matrix_audit_complete=False,test_accessed=False,status={})
    (output/'execution-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def execute(arm):
        log_path=output/(arm+'.log')
        status=dict(arm=arm,training_complete=False,replays_complete=False)
        try:
            with log_path.open('w') as log:
                remaining=deadline-time.monotonic()
                if remaining<=0:raise subprocess.TimeoutExpired(tasks[arm],0)
                completed=subprocess.run(tasks[arm],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=remaining)
                if completed.returncode:
                    status['training_exit_code']=completed.returncode;return status
                status['training_complete']=True
                for budget in spec['saved_update_budgets']:
                    command=[sys.executable,str(ROOT/'scripts/replay_tuning_checkpoint_dev031.py'),
                        '--report',str(output/arm/f'{budget}.json'),
                        '--output',str(output/arm/f'{budget}-replay.json')]
                    remaining=deadline-time.monotonic()
                    if remaining<=0:raise subprocess.TimeoutExpired(command,0)
                    replay=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=remaining)
                    if replay.returncode:
                        status['replay_exit_code']=replay.returncode;return status
                status['replays_complete']=True
        except subprocess.TimeoutExpired:status['wall_budget_exhausted']=True
        return status
    workers=max(1,min(spec['cpu_workers_maximum'],(os.cpu_count() or 2)-1))
    with ThreadPoolExecutor(max_workers=workers) as cpu,ThreadPoolExecutor(max_workers=1) as gpu:
        futures={cpu.submit(execute,a):a for a in spec['cpu_dispatch_order']}
        futures.update({gpu.submit(execute,'gru768'):'gru768'})
        for future in as_completed(futures):
            status=future.result();manifest['status'][status['arm']]=status
            (output/'execution-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            print(json.dumps(status),flush=True)
    complete=all(x['training_complete'] and x['replays_complete'] for x in manifest['status'].values()) and len(manifest['status'])==6
    manifest['bounded_tranche_execution_and_replay_complete']=complete
    (output/'execution-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return 0 if complete else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();raise SystemExit(run(a.output))
