"""One registered synthetic-only write/read repair. Never real-world evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from scripts.diagkey001_associative_recall import model_for as old_model
from scripts.diagkey002_no_fit_mechanism import original_key_query
from tac_osm.aligned_associative_tac import AlignedAssociativeTAC
from tac_osm.parameter_budget import budgeted_optimizer, checked_optimizer_step, save_training_checkpoint, file_sha, validate_runtime_evidence

STUDY='DIAGKEY004_ALIGNED_GATED_ZERO_REPAIR_SYNTHETIC_ONLY'
SEEDS=(75001,75002,75003,75004,75005)
ARMS=('aligned_tac','tac','tac_noassoc','gru','transformer')
COUNTS={'aligned_tac':101527,'tac':101554,'tac_noassoc':101560,'gru':101500,'transformer':101616}
STEPS=4096
BATCH=16
MILESTONES=(640,4096)
CURVE=(1,64,128,256,512,640,1024,2048,3072,4096)
ROOT=Path('research/mathematical-engineering/diagkey004')
REG=ROOT/'FAIR_COMPARISON.json'
OUT=Path('research/replay-bundles/DIAGKEY004')


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')


def source_integrity():
    manifest=json.loads(REG.read_text())
    for name,expected in manifest["implementation_sha256"].items():
        if file_sha(name)!=expected:raise RuntimeError("frozen training implementation changed: "+name)


def model_for(arm):return AlignedAssociativeTAC() if arm=='aligned_tac' else old_model(arm)


def batch(seed,step,split='FIT_ZERO',size=BATCH):
    # FIT_ZERO reproduces the DIAGKEY003 schedule exactly through update640.
    return original_key_query(seed,step,0,batch_size=size,split=split)


def tensor_digest_update(digest,arrays):
    for x in arrays:digest.update(x.numpy().tobytes())


@torch.no_grad()
def evaluate(model,seed,role):
    model.eval();all_logits=[];all_y=[]
    n=128 if role=='DEV' else 512
    for start in range(0,n,32):
        t,k,v,y=batch(seed+30000,800000+start//32,'DIAGKEY004_'+role,32)
        all_logits.append(model(t,k,v));all_y.append(y)
    logits=torch.cat(all_logits);y=torch.cat(all_y)
    correct=(logits.argmax(-1)==y)
    return {'cases':n,'correct':int(correct.sum()),'accuracy':float(correct.double().mean()),
            'NLL':float(F.cross_entropy(logits.double(),y)),
            'logits':logits.numpy().tolist(),'labels':y.tolist(),'correct_bits':correct.int().tolist()}


def compact(row):return {k:v for k,v in row.items() if k not in ('logits','labels','correct_bits')}


def preflight():
    torch.set_num_threads(1)
    from scripts.validate_research_parity import check_manifest
    manifest=json.loads(REG.read_text());check_manifest(manifest)
    counts={};profile={}
    for arm in ARMS:
        m=model_for(arm);n=sum(p.numel() for p in m.parameters())
        assert n==sum(p.numel() for p in m.parameters() if p.requires_grad)==COUNTS[arm]
        counts[arm]=n
        t,k,v,y=batch(75001,0,'PROFILE')
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as p:
            F.cross_entropy(m(t,k,v),y).backward()
        profile[arm]=sum(int(e.flops or 0) for e in p.key_averages())
        if any(q.grad is None or not bool(torch.isfinite(q.grad).all()) for q in m.parameters()):
            raise RuntimeError('inactive/nonfinite full-model gradient')
    report={'status':'PASS_LIVE_COUNTS_AND_NUMERICS_ONLY','models':counts,
            'partial_supported_forward_backward_flops_batch16':profile,
            'flop_gap_fraction_vs_tac':{a:(profile[a]-profile['tac'])/profile['tac'] for a in ARMS},
            'full_training_compute_parity_verified':False,'architecture_superiority_eligible':False,
            'planned_25_fits':True,'optimizer_updates_executed':0,
            'registration_sha256':file_sha(REG)}
    dest=ROOT/'PREFLIGHT.json'
    if dest.exists():
        if json.loads(dest.read_text())!=report:raise RuntimeError('immutable original preflight changed')
    else:write_json(dest,report)
    print(json.dumps(report),flush=True)


def fit(arm,seed):
    if arm not in ARMS or seed not in SEEDS:raise RuntimeError('unregistered fit')
    source_integrity()
    torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    m=model_for(arm);folder=OUT/f'{arm}-s{seed}'
    if folder.exists():raise RuntimeError('original fit directory exists: no overwrite or refit')
    folder.mkdir(parents=True)
    receipt=folder/'runtime.json'
    opt=budgeted_optimizer(m,torch.optim.AdamW,m.parameters(),lr=.001,weight_decay=.0001,
        budget_manifest=REG,budget_receipt=receipt,budget_seed=seed,budget_architecture=arm)
    trace=[];schedule=hashlib.sha256();start_cpu=time.process_time();start_wall=time.perf_counter()
    saved=[]
    for step in range(1,STEPS+1):
        t,k,v,y=batch(seed,step-1);tensor_digest_update(schedule,(t,k,v,y))
        m.train();opt.zero_grad(set_to_none=True);logits=m(t,k,v);loss=F.cross_entropy(logits,y)
        if not bool(torch.isfinite(loss)):raise RuntimeError('nonfinite loss: retain failed run, do not restart')
        loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.0));checked_optimizer_step(opt)
        if step in CURVE:
            dev=evaluate(m,seed,'DEV')
            trace.append({'step':step,'FIT_loss':float(loss.detach()),'gradient_norm':norm,'DEV':compact(dev)})
            print(json.dumps({'arm':arm,'seed':seed,**trace[-1]}),flush=True)
        if step in MILESTONES:
            path=folder/f'step-{step}.pt'
            save_training_checkpoint({'weights':m.state_dict(),'optimizer':opt.state_dict(),
                'torch_rng_state':torch.get_rng_state(),'numpy_rng_state':np.random.get_state(),
                'python_rng_state':random.getstate(),'step':step,'seed':seed,'arm':arm,
                'schedule_sha256':schedule.hexdigest(),'registration_sha256':file_sha(REG)},path)
            saved.append({'step':step,'path':str(path),'sha256':file_sha(path)})
    value=json.loads(receipt.read_text())
    if value['successful_optimizer_steps']!=STEPS or value['status']!='CHECKPOINT_VERIFIED':raise RuntimeError('missing live update proof')
    write_json(folder/'FIT_RECEIPT.json',{'arm':arm,'seed':seed,'steps':STEPS,'examples':STEPS*BATCH,
        'trainable_parameters':COUNTS[arm],'deployed_parameters':COUNTS[arm],
        'training_schedule_sha256':schedule.hexdigest(),'curve':trace,'checkpoints':saved,
        'runtime_sha256':file_sha(receipt),'training_process_cpu_seconds':time.process_time()-start_cpu,
        'training_wall_seconds':time.perf_counter()-start_wall,'test_opened':False,
        'registration_sha256':file_sha(REG)})


def verify_fit(arm,seed,git_owned=True):
    source_integrity()
    folder=OUT/f'{arm}-s{seed}';r=json.loads((folder/'FIT_RECEIPT.json').read_text())
    if r['registration_sha256']!=file_sha(REG):raise RuntimeError('registration modified since fitting')
    if r['steps']!=STEPS or r['examples']!=STEPS*BATCH:raise RuntimeError('budget drift')
    for cp in r['checkpoints']:
        p=Path(cp['path'])
        if file_sha(p)!=cp['sha256']:raise RuntimeError('original checkpoint changed')
        if git_owned and subprocess.check_output(['git','show','HEAD:'+str(p)])!=p.read_bytes():raise RuntimeError('original model is not Git owned')
    final=r['checkpoints'][-1]
    evidence={'receipt_path':str(folder/'runtime.json'),'receipt_sha256':r['runtime_sha256'],
              'checkpoint_path':final['path'],'checkpoint_sha256':final['sha256']}
    validate_runtime_evidence({'seed':seed,'architecture':arm,'trainable_parameters':COUNTS[arm],
        'deployed_parameters':COUNTS[arm],'checkpoint_sha256':final['sha256'],
        'runtime_evidence':evidence,'_prefit_registration_sha256':file_sha(REG),'_expected_optimizer_steps':STEPS})
    return r


def score_or_replay(replay=False):
    torch.set_num_threads(1)
    fits={(a,s):verify_fit(a,s) for a in ARMS for s in SEEDS}
    for seed in SEEDS:
        if len({fits[a,seed]['training_schedule_sha256'] for a in ARMS})!=1:raise RuntimeError('different training exposure/order')
    # All 25 final original fits and both milestone checkpoint sets are frozen
    # before the FIRST synthetic TEST logits are computed.
    rows=[]
    for arm in ARMS:
        for seed in SEEDS:
            for cp in fits[arm,seed]['checkpoints']:
                saved=torch.load(cp['path'],map_location='cpu',weights_only=False)
                m=model_for(arm);m.load_state_dict(saved['weights'],strict=True)
                result=evaluate(m,seed,'TEST')
                path=OUT/f'{arm}-s{seed}'/f"TEST-step-{cp['step']}.json"
                out={**result,'arm':arm,'seed':seed,'step':cp['step'],'checkpoint_sha256':cp['sha256']}
                if replay:
                    original=json.loads(path.read_text())
                    if original['labels']!=out['labels'] or original['correct_bits']!=out['correct_bits']:raise RuntimeError('synthetic recall replay differs')
                    if not np.allclose(original['logits'],out['logits'],atol=1e-5,rtol=1e-5):raise RuntimeError('original logit tolerance failed')
                    if abs(original['NLL']-out['NLL'])>1e-5:raise RuntimeError('original NLL replay failed')
                else:
                    if path.exists():raise RuntimeError('original scores exist: use replay')
                    write_json(path,out)
                rows.append({**compact(out),'checkpoint_sha256':cp['sha256']})
    summary={'study_id':STUDY,'synthetic_only':True,'models':25,'milestone_score_sets':50,
        'real_world_gain':False,'architecture_superiority_eligible':False,
        'full_FLOP_parity_verified':False,'results':{},'rows':rows}
    for arm in ARMS:
        summary['results'][arm]={}
        for step in MILESTONES:
            values=[r['accuracy'] for r in rows if r['arm']==arm and r['step']==step]
            summary['results'][arm][str(step)]={'mean_accuracy':sum(values)/len(values),'seed_accuracies':values,
                'all_five_ge_90_percent':all(v>=.9 for v in values)}
    summary['next_decision']=('Zero-delay repair gate passes; separately register interference curriculum; do not infer real-world memory.'
        if summary['results']['aligned_tac'][str(STEPS)]['all_five_ge_90_percent'] else
        'Zero-delay repair gate failed: stop this synthetic tuning series; redesign write/read path and return to real-data capability work.')
    path=OUT/('INDEPENDENT_REPLAY.json' if replay else 'ORIGINAL_SUMMARY.json')
    if replay:
        original=json.loads((OUT/'ORIGINAL_SUMMARY.json').read_text())
        if original!=summary:raise RuntimeError('summary replay differs')
        report={'status':'PASS_ORIGINAL_25_SYNTHETIC_MODEL_REPLAY','new_optimizer_updates':0,
                'models':25,'milestone_score_sets':50,'registration_sha256':file_sha(REG),
                'summary_sha256':file_sha(OUT/'ORIGINAL_SUMMARY.json')}
        if path.exists() and json.loads(path.read_text())!=report:raise RuntimeError('immutable replay audit changed')
        if not path.exists():write_json(path,report)
    else:write_json(path,summary)
    print(json.dumps(summary if not replay else report),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('preflight','fit','score','replay','fit-all'))
    p.add_argument('--arm',choices=ARMS);p.add_argument('--seed',type=int);a=p.parse_args()
    if a.mode=='preflight':preflight()
    elif a.mode=='fit':fit(a.arm,a.seed)
    elif a.mode in ('score','replay'):score_or_replay(a.mode=='replay')
    else:
        from concurrent.futures import ThreadPoolExecutor
        def launch(job):
            arm,seed=job
            log=Path('/workspace')/f'diagkey004-{arm}-{seed}.log'
            with log.open('w') as out:
                subprocess.run([sys.executable,'-m','scripts.diagkey004_aligned_recall','fit','--arm',arm,'--seed',str(seed)],stdout=out,stderr=subprocess.STDOUT,check=True)
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(launch,[(arm,s) for s in SEEDS for arm in ARMS]))


if __name__=='__main__':main()
