#!/usr/bin/env python3
"""Seal all original checkpoints before any held-out scoring, retain every result."""
import argparse
import json
import math
from pathlib import Path
import resource
import time

import numpy as np
import torch

from core import (ARMS,MILESTONES,ROOT,SEEDS,atomic_json,checkpoint_path,make,
                  planned_digests,read_sources,sha)
from scoring import Counts,neural_file

def seal(out,data):
    from scripts.validate_research_parity import check_manifest
    manifest=json.loads((ROOT/'research/mathematical-engineering/codevalue003/FAIR_COMPARISON.json').read_text())
    source,fit=read_sources(data,{'FIT'})
    docs={r['repo']+':'+r['path']:b for r,b in fit}
    records={};actual={}
    for seed in SEEDS:
        planned=planned_digests(docs,seed)
        for arm in ARMS:
            for step in MILESTONES:
                path=checkpoint_path(out,arm,seed,step)
                rec=json.loads(path.with_suffix('.json').read_text())
                cp=torch.load(path,map_location='cpu',weights_only=True)
                if rec['checkpoint_sha256']!=sha(path):raise ValueError('checkpoint changed')
                if cp['sample_sha256']!=planned[step] or rec['sample_sha256']!=planned[step]:
                    raise ValueError('original sample schedule mismatch')
                if (cp['study'],cp['arm'],cp['seed'],cp['steps'])!=('CODEVALUE003',arm,seed,step):
                    raise ValueError('checkpoint identity mismatch')
                if cp['source_cohort_sha256']!=source['cohort_sha256']:raise ValueError('source identity mismatch')
                for key in ('optimizer','torch_rng','cuda_rng','python_rng','numpy_rng','sampler_state'):
                    if key not in cp:raise ValueError('missing optimizer/RNG/sampler state')
                if not cp['optimizer']['state']:raise ValueError('untrained optimizer')
                m=make(arm,'cpu');m.load_state_dict(cp['model'],strict=True)
                n=sum(p.numel() for p in m.parameters() if p.requires_grad)
                if n!=rec['trainable_parameters']:raise ValueError('actual final count mismatch')
                records[str(path.relative_to(out))]=rec['checkpoint_sha256']
                if step==10240:actual[arm,seed]=rec
                del cp,m
    for model in manifest['models']:model['measured']=actual[model['name'],model['seed']]
    report=check_manifest(manifest,'final')
    atomic_json(out/'FINAL_RESOURCE_GATE.json',report)
    atomic_json(out/'FINAL_MANIFEST.json',manifest)
    atomic_json(out/'SEAL.json',{'study':'CODEVALUE003','checkpoints':records,
        'all45_checkpoints_complete':True,'source_cohort_sha256':source['cohort_sha256'],
        'status':'SEALED_BEFORE_HELDOUT_SCORING','independently_qualified':False})
    return records

def run(out,data,device):
    if (out/'SEAL.json').exists():raise FileExistsError('no repeated held-out scoring')
    seal(out,data)
    _,held=read_sources(data,{'DEV','TRANSFER_DEV'})
    # Defer actual held-out reads until all nine final models have passed sealing.
    _,fit=read_sources(data,{'FIT'})
    counts={name:Counts([b for _,b in fit],order) for name,order in [('p3',3),('wb6',6)]}
    for name,baseline in counts.items():
        for r,raw in held:
            path=out/'scores'/f'{name}-{r["file"]}.json'
            started=time.monotonic();score=baseline.score(raw)
            score.update(source=r,predictor=name,seconds=time.monotonic()-started,
                         peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            atomic_json(path,score)
            print(json.dumps({'stage':'baseline_scored','predictor':name,'file':r['file'],'nll':score['nll']}),flush=True)
    del counts
    for seed in SEEDS:
        for arm in ARMS:
            for step in MILESTONES:
                cp=torch.load(checkpoint_path(out,arm,seed,step),map_location='cpu',weights_only=True)
                model=make(arm,device).eval();model.load_state_dict(cp['model'],strict=True)
                # Real saved-model prediction causality check on FIT only.
                probe=torch.tensor([list(fit[0][1][:32])],device=device)
                poisoned=probe.clone();poisoned[:,16:]=(poisoned[:,16:]+127)%256
                with torch.inference_mode():
                    delta=float((model(probe)[:,:16]-model(poisoned)[:,:16]).abs().max())
                if delta>1e-5:raise ValueError('trained checkpoint causal poison failure')
                for i,(r,raw) in enumerate(held):
                    if step!=10240 and r['partition']!='DEV':continue
                    started=time.monotonic()
                    score=neural_file(model,raw,device,completion=step==10240,
                                      measure_latency=step==10240 and i==0)
                    score.update(source=r,predictor=arm,seed=seed,step=step,
                         checkpoint_sha256=sha(checkpoint_path(out,arm,seed,step)),
                         saved_model_future_poison_max_abs=delta,
                         weights_bytes=sum(t.numel()*t.element_size() for t in model.state_dict().values()),
                         cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,
                         cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if device.startswith('cuda') else 0,
                         peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                         seconds=time.monotonic()-started)
                    atomic_json(out/'scores'/f'{arm}-{seed}-{step}-{r["file"]}.json',score)
                    print(json.dumps({'stage':'neural_scored','arm':arm,'seed':seed,'step':step,
                                      'file':r['file'],'nll':score['nll']}),flush=True)
                del cp,model
                if device.startswith('cuda'):torch.cuda.empty_cache()
    summarize(out)

def summarize(out):
    rows=[json.loads(p.read_text()) for p in sorted((out/'scores').glob('*.json'))]
    transfer=[r for r in rows if r['source']['partition']=='TRANSFER_DEV']
    baseline={p:{r['source']['file']:r for r in transfer if r['predictor']==p} for p in ('p3','wb6')}
    models={}
    for arm in ARMS:
        models[arm]=[]
        for seed in SEEDS:
            group=[r for r in transfer if r['predictor']==arm and r.get('seed')==seed and r.get('step')==10240]
            if len(group)!=10:raise ValueError('missing complete transfer source matrix')
            n=sum(r['targets'] for r in group);offer=sum(len(r['offers']) for r in group)
            exact=sum(o['exact']['4'] for r in group for o in r['offers'])
            models[arm].append({'seed':seed,'targets':n,'offers':offer,
                'nll':sum(r['nll_sum'] for r in group)/n,
                'accuracy':sum(r['top1'] for r in group)/n,
                'exact4':exact,'exact4_rate':exact/offer,
                'oracle_net_keys':exact*3,'rejection_cost4_net_keys':exact*3-(offer-exact)*4})
    def rate(r):return sum(o['exact']['4'] for o in r['offers'])/len(r['offers'])
    tac={file:np.mean([rate(r) for r in transfer if r['predictor']=='tac' and r.get('step')==10240 and r['source']['file']==file]) for file in baseline['p3']}
    groups={}
    for r in baseline['p3'].values():groups.setdefault(r['source']['group'],[]).append(r['source']['file'])
    rng=np.random.default_rng(930003);ci={};all_pass=True
    for b,byfile in baseline.items():
        gains=[]
        for _ in range(10000):
            chosen=rng.choice(sorted(groups),size=len(groups),replace=True)
            diffs=[]
            for publisher in chosen:
                files=rng.choice(groups[publisher],size=len(groups[publisher]),replace=True)
                diffs.extend(tac[f]-rate(byfile[f]) for f in files)
            gains.append(float(np.mean(diffs)))
        lo,hi=np.quantile(gains,[.025,.975])
        groupwins=sum(np.mean([tac[f]-rate(byfile[f]) for f in files])>0 for files in groups.values())
        b_rate=np.mean([rate(r) for r in byfile.values()])
        seedwins=sum(r['exact4_rate']>b_rate for r in models['tac'])
        ci[b]={'mean_gain':float(np.mean([tac[f]-rate(byfile[f]) for f in tac])),
               'paired_hierarchical95ci':[float(lo),float(hi)],'publisher_wins':groupwins,
               'seed_wins':seedwins,'uncertainty_limit':'Conditional on three observed seed realizations; publisher/file bootstrap does not integrate optimization-seed uncertainty.'}
        all_pass &= lo>0 and groupwins>=4 and seedwins>=2
    all_pass &= np.mean([r['rejection_cost4_net_keys'] for r in models['tac']])>0
    learning_curves={}
    for arm in ARMS:
        learning_curves[arm]=[]
        for seed in SEEDS:
            curve=[]
            for step in MILESTONES:
                selected=[r for r in rows if r['predictor']==arm and r.get('seed')==seed
                          and r.get('step')==step and r['source']['partition']=='DEV']
                if len(selected)!=6:raise ValueError('incomplete DEV checkpoint curve')
                n=sum(r['targets'] for r in selected)
                curve.append({'step':step,'nll':sum(r['nll_sum'] for r in selected)/n,
                              'accuracy':sum(r['top1'] for r in selected)/n})
            learning_curves[arm].append({'seed':seed,'curve':curve,
                'final_nll_improves_over320':curve[-1]['nll']<curve[0]['nll']})
    baselines={}
    for name,byfile in baseline.items():
        source_rows=list(byfile.values());n=sum(r['targets'] for r in source_rows)
        offers=sum(len(r['offers']) for r in source_rows)
        exact=sum(o['exact']['4'] for r in source_rows for o in r['offers'])
        baselines[name]={'targets':n,'offers':offers,
            'nll':sum(r['nll_sum'] for r in source_rows)/n,
            'accuracy':sum(r['top1'] for r in source_rows)/n,
            'exact4':exact,'oracle_net_keys':exact*3,
            'rejection_cost4_net_keys':exact*3-(offers-exact)*4}
    report={'study':'CODEVALUE003','status':'COMPLETE_INTERNAL_RESEARCH',
            'models':models,'baselines':baselines,'dev_learning_curves':learning_curves,
            'tac_language_signal':sum(r['final_nll_improves_over320'] for r in learning_curves['tac'])>=2,
            'tac_exact4_transfer_gains':ci,
            'utility_screen':'PASS_CONSIDER_HUMAN_PILOT' if all_pass else 'FAIL',
            'commercial_viability_proved':False,'independently_qualified':False,
            'architecture_superiority_eligible':False,'protected_TEST':'NOT_OPENED',
            'training_flops':'UNMATCHED_INCOMPLETE_OPERATOR_COVERAGE',
            'no_suggestion_baseline_net_keys':0,
            'limitations':['Only Python source prediction, not general English or executable code generation',
                'No observed developer time, acceptance, costs or revenue',
                'Transfer source identities frozen; no strict blind/access-separated TEST claim',
                'Sampled lexical duplicate screen, not semantic duplication proof',
                'Simple SSM is not Mamba; no contemporary attention comparator',
                'All original positive/negative per-file/per-seed scores retained']}
    atomic_json(out/'RESULTS.json',report)
    print(json.dumps({'stage':'complete','utility_screen':report['utility_screen']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();run(a.out,a.data,a.device)
