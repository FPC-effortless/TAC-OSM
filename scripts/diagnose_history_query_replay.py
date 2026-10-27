#!/usr/bin/env python3
"""Diagnose exact cross-runner mismatch without changing scientific receipts.

This command reports differences and fails on any mismatch. No tolerance gate,
fit, checkpoint mutation or result replacement. Original trainer remains pinned.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts import train_history_query_diverse as study
from scripts.original_history_trace import sha_file


def differences(a,b,path=''):
    if type(a)!=type(b):return [dict(path=path,expected=a,actual=b,kind='type')]
    if isinstance(a,dict):
        if a.keys()!=b.keys():return [dict(path=path,kind='keys')]
        return [r for k in a for r in differences(a[k],b[k],path+'/'+str(k))]
    if isinstance(a,list):
        if len(a)!=len(b):return [dict(path=path,kind='length')]
        return [r for i,(x,y) in enumerate(zip(a,b)) for r in differences(x,y,path+'/'+str(i))]
    if a==b:return []
    return [dict(path=path,expected=a,actual=b,kind='numeric' if type(a) is float else 'identity',
        **({'absolute_error':abs(a-b)} if type(a) is float else {}))]


def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--implementation-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    recipe,source,original,d,plan,_=study.frozen(args)
    manifest=json.loads((study.OUT/'FINAL_COMPARISON.json').read_text())
    study.check_manifest(manifest,'final')
    results=json.loads((study.OUT/'RESULTS.json').read_text())
    report=dict(status='DIAGNOSING_EXACT_REPLAY_NOT_A_TOLERANCE_PASS',
        original_results_sha256=sha_file(study.OUT/'RESULTS.json'),
        runtime=dict(torch=str(torch.__version__),python=platform.python_version(),machine=platform.machine(),
            cpu_capability=torch.backends.cpu.get_cpu_capability(),mkldnn=torch.backends.mkldnn.enabled,
            torch_build=torch.__config__.show()),models=[])
    args.output.mkdir(parents=True,exist_ok=True)
    for row in results['models']:
        path=ROOT/row['checkpoint']
        if sha_file(path)!=row['sha256']:raise ValueError('checkpoint changed')
        c=torch.load(path,map_location='cpu',weights_only=False)
        receipt=next(r['measured'] for r in manifest['arms'] if r['seed']==row['seed'] and r['architecture']==row['arm'])
        if (c['recipe_sha256']!=study.digest(recipe) or c['sample_order_sha256']!=study.digest(plan['orders'][str(row['seed'])])
            or c['training_script_sha256']!=sha_file(Path(study.__file__)) or c['source_sha256']!=recipe['source_sha256']
            or c['seed']!=row['seed'] or c['architecture']!=row['arm'] or c['steps']!=64 or c['examples_seen']!=1024
            or len(c['losses'])!=64 or c['losses']!=row['losses'] or c['actual_operator_step_sha256']!=receipt['actual_operator_step_sha256']
            or receipt['checkpoint_sha256']!=row['sha256']):raise ValueError('checkpoint/source/order/recipe/resource mismatch')
        if len(c['optimizer']['state'])!=43 or any(float(v['step'])!=64 for v in c['optimizer']['state'].values()):
            raise ValueError('original optimizer state incomplete')
        model=original.create_model(row['seed'],row['arm']);model.load_state_dict(c['model'],strict=True)
        actual={cohort:study.evaluate(model,d,plan[cohort],source) for cohort in ('primary','secondary')}
        out=args.output/f"actual-{row['seed']}-{row['arm']}.json";study.write(out,actual)
        delta=differences({k:row[k] for k in ('primary','secondary')},actual)
        item=dict(seed=row['seed'],arm=row['arm'],exact_match=not delta,
            mismatch_count=len(delta),maximum_numeric_absolute_error=max((r.get('absolute_error',0) for r in delta),default=0),
            identity_mismatches=[r for r in delta if r['kind']!='numeric'],differences=delta)
        report['models'].append(item)
        print('DIAGNOSIS',row['seed'],row['arm'],item['mismatch_count'],'max_numeric',item['maximum_numeric_absolute_error'],
            'identity_mismatches',len(item['identity_mismatches']),flush=True)
    report['exact_match']=all(r['exact_match'] for r in report['models'])
    study.write(args.output/'CROSS_RUNNER_DIAGNOSIS.json',report)
    if not report['exact_match']:raise SystemExit(2)
if __name__=='__main__':main()
