#!/usr/bin/env python3
"""Registered B follow-up; reuses original A AR parent bytes without refitting."""
import argparse,json
from pathlib import Path
import torch
from scripts import me043_train_adaptive as core
from tac_osm.me043_scheduled_compute import ARMS,build_model,objective as scheduled_objective

def configure():
    core.ARMS=ARMS
    core.COSTS={'scheduled_free':0.,'scheduled_001':.001}
    core.build_model=build_model
    def loss(net,x,y):
        # Step recorded on the model for periodic expert updates. Reconstructed
        # from update count, never driven by DEV scores or external time.
        net._training_update=getattr(net,'_training_update',0)+1
        return scheduled_objective(net,x,y,net._training_update)
    core.objective=loss
    def summary(rows):
        result={}
        for lane in core.LANES:
            scores={}
            for arm in ARMS:
                subset=[r for r in rows if r['lane']==lane and r['arm']==arm]
                assert len(subset)==4
                scores[arm]={'primary_MSE':sum(r['metric']['primary_MSE'] for r in subset)/4,
                    'CPU':sum(r['optimizer_CPU'] for r in subset)/4,
                    'updates':sum(r['updates'] for r in subset)/4,
                    'by_horizon':{str(h):sum(r['metric']['primary_MSE'] for r in subset if r['horizon']==h)/2
                                  for h in core.HORIZONS},
                    'native_fraction':sum(r['metric']['native_fraction'] for r in subset)/4
                        if arm.startswith('scheduled_') else None}
            result[lane]=scores
        result['gates']={}
        for arm in core.COSTS:
            exp=result['exposure'];cpu=result['cpu'];v=exp[arm]
            acc=v['primary_MSE']<=.99*min(exp['wide_context']['primary_MSE'],exp['wide_native']['primary_MSE'])
            horizon=all(v['by_horizon'][str(h)]<=1.05*min(exp['wide_context']['by_horizon'][str(h)],
                exp['wide_native']['by_horizon'][str(h)]) for h in core.HORIZONS)
            practical=cpu[arm]['primary_MSE']<=min(cpu['wide_context']['primary_MSE'],cpu['wide_native']['primary_MSE'])
            result['gates'][arm]={'one_percent_accuracy':acc,'no_large_horizon_regression':horizon,
                                 'equal_CPU_accuracy':practical,'all':acc and horizon and practical}
        return result
    core.summarize=summary


def run(source,dest):
    source=Path(source);dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
    original=json.loads((source/'results.json').read_text());assert original['completed']
    report={'study':'ME043B','source_commit':__import__('subprocess').check_output(
        ['git','rev-parse','HEAD'],text=True).strip(),'domains':{},'completed':False,
        'posthoc_after_A':True,'new_test_numeric_values_opened':0,
        'architectural_superiority_claim':False,'original_A_parent_reuse':True}
    for domain,mod in core.DOMAINS.items():
        out=dest/domain;out.mkdir()
        z,provenance=mod.get_source(out)
        data={'provenance':provenance,'fits':[],'parents':original['domains'][domain]['parents']}
        report['domains'][domain]=data
        for h in core.HORIZONS:
            for seed in core.SEEDS:
                prec=next(r for r in data['parents'] if r['horizon']==h and r['seed']==seed)
                ep=prec['endpoints'][str(prec['selected_DEV_parent_step'])]
                cp=source/domain/ep['checkpoint'];assert core.sha(cp)==ep['checkpoint_sha256']
                parent=torch.load(cp,weights_only=True)['state_dict']
                for lane in core.LANES:
                    for arm in ARMS:
                        row=core.train_case(z,mod,h,seed,arm,lane,parent,prec,out)
                        row['expert_update_steps']=row['updates']//8 if arm.startswith('scheduled_') else row['updates']
                        data['fits'].append(row);core.save(dest/'results.json',report)
        data['evaluation']=core.summarize(data['fits']);core.save(dest/'results.json',report)
    report['completed']=True;core.save(dest/'results.json',report)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',default='/workspace/me043-original')
    ap.add_argument('--out',required=True);ap.add_argument('--replay',action='store_true')
    ap.add_argument('--latency',action='store_true');a=ap.parse_args();configure()
    if a.replay:core.replay(a.out)
    elif a.latency:core.latency(a.out)
    else:run(a.source,a.out)
