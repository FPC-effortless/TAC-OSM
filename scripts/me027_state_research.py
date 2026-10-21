"""Fixed real-development learned-state/resource study. No new cohort access."""
from __future__ import annotations
import argparse
import json
import platform
import resource
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from tac_osm.state_value import StateValueModel
from scripts.me014b_exact_action_memory import batch
from scripts.me022_native_readout import predictions,scores,save_json
from scripts.me023_predictive_research import sha,load_source,atomic_checkpoint
from scripts.me025_real_transformer_parity import fit_only_popularity

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'research/mathematical-engineering/me027/contract.json'
C=json.loads(CONTRACT.read_text())


def make_model(arm,mean,scale,pop):return StateValueModel(arm,mean,scale,pop)


def fit(arm,seed,train,dev,mean,scale,pop,preprocess_cpu,out):
    start=time.process_time();torch.manual_seed(seed)
    net=torch.jit.script(make_model(arm,mean,scale,pop))
    assert sum(p.numel() for p in net.parameters())==C['parameters'][arm]
    body,head=[],[]
    for name,p in net.named_parameters():(head if name.startswith('head.') else body).append(p)
    optimizer=torch.optim.AdamW([{'params':body,'lr':C['base_lr']},{'params':head,'lr':C['head_lr']}],weight_decay=C['weight_decay'])
    setup_cpu=time.process_time()-start;cpu=preprocess_cpu+setup_cpu;train_wall=0.;optimizer_cpu=0.;max_step_cpu=0.
    gen=torch.Generator().manual_seed(seed+300);weights=torch.tensor(C['prefix_weights']).reshape(1,4)
    name=f'{arm}-seed{seed}';trace=[];checkpoints=[];seen=set();step=0
    def record(labels,gradient=None):
        p=predictions(net,dev);metrics=scores(p,dev['y'].numpy())
        trace.append({'step':step,'charged_cpu_seconds':cpu,'optimizer_cpu_seconds':optimizer_cpu,
            'optimizer_wall_seconds':train_wall,'labels':labels,'dev':metrics,'gradient_norm_before_clip':gradient})
        checkpoint={'arm':arm,'seed':seed,'step':step,'state_dict':net.state_dict(),'optimizer':optimizer.state_dict(),
            'rng':torch.get_rng_state(),'sampler_rng':gen.get_state(),'contract':C,'duration_mean':mean,'duration_scale':scale,
            'charged_cpu_seconds':cpu,'optimizer_cpu_seconds':optimizer_cpu,'setup_cpu_seconds':setup_cpu,
            'shared_preprocessing_cpu_seconds_charged':preprocess_cpu,'optimizer_wall_seconds':train_wall}
        atomic_checkpoint(out/f'{name}-latest.pt',checkpoint)
        if labels:
            path=out/f'{name}-step{step:06d}.pt';atomic_checkpoint(path,checkpoint)
            pp=out/f'{name}-step{step:06d}-predictions.npy';np.save(pp,p)
            jit=None
            if f'cpu{C["primary_cpu_budget_seconds"]}' in labels:
                jit=out/f'{name}-primary.jit.pt';torch.jit.save(net.eval(),str(jit))
            checkpoints.append({'labels':labels,'step':step,'charged_cpu_seconds':cpu,'optimizer_cpu_seconds':optimizer_cpu,
                'setup_cpu_seconds':setup_cpu,'shared_preprocessing_cpu_seconds_charged':preprocess_cpu,
                'sampled_windows':step*C['batch'],'historical_event_presentations':step*C['batch']*128,
                'supervised_responses':step*C['batch']*4,'dev':metrics,'checkpoint':path.name,'checkpoint_sha256':sha(path),
                'prediction':pp.name,'prediction_sha256':sha(pp),'standalone':jit.name if jit else None,
                'standalone_sha256':sha(jit) if jit else None})
        save_json(out/f'{name}-trace.json',trace)
        save_json(out/f'{name}-partial.json',{'arm':arm,'seed':seed,'trace':trace,'checkpoints':checkpoints})
    record([])
    while step<max(C['exposure_steps']) or cpu<max(C['cpu_budgets_seconds']):
        if step>=C['maximum_steps_per_fit'] or cpu>=C['maximum_charged_cpu_seconds_per_fit']:
            record(['partial_ceiling']);raise RuntimeError('fixed training ceiling reached; partial state preserved')
        t,w=time.process_time(),time.perf_counter();net.train()
        idx=torch.randint(len(train['y']),(C['batch'],),generator=gen);optimizer.zero_grad(set_to_none=True)
        logits=net.prefix_logits(*batch(train,idx))
        target=torch.cat((train['metadata'][idx][:,C['prefix_depths'],1],train['y'][idx]),-1)
        loss=(F.binary_cross_entropy_with_logits(logits,target,reduction='none')*weights).sum(-1).mean()
        assert torch.isfinite(loss);loss.backward();gradient=torch.nn.utils.clip_grad_norm_(net.parameters(),C['gradient_clip'])
        assert torch.isfinite(gradient);optimizer.step()
        elapsed=time.process_time()-t;optimizer_cpu+=elapsed;cpu+=elapsed;max_step_cpu=max(max_step_cpu,elapsed)
        train_wall+=time.perf_counter()-w;step+=1
        labels=[]
        for n in C['exposure_steps']:
            if step==n:labels.append(f'step{n}')
        for b in C['cpu_budgets_seconds']:
            label=f'cpu{b}'
            if cpu>=b and label not in seen:labels.append(label);seen.add(label)
        if step in C['trace_steps'] or labels:record(labels,float(gradient))
    row={'arm':arm,'seed':seed,'parameters':C['parameters'][arm],'embedding_parameters':128000,
        'nonembedding_parameters':C['parameters'][arm]-128000,'actual_final_steps':step,
        'final_charged_cpu_seconds':cpu,'max_single_step_cpu_seconds':max_step_cpu,
        'process_lifetime_peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'isolated_training_peak_memory_measured':False,'checkpoints':checkpoints,'trace':trace}
    print(json.dumps({'arm':arm,'seed':seed,'steps':step,'cpu':cpu,'budgets':{label:{'step':q['step'],'Brier':q['dev']['Brier']} for q in checkpoints for label in q['labels']}}),flush=True)
    return row


def run(parent,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    wall,study_cpu=time.perf_counter(),time.process_time();auth_start=time.process_time()
    data=load_source(parent);train,dev=data['train'],data['dev'];auth_cpu=time.process_time()-auth_start
    assert sha(Path(parent)/'data-snapshot.pt')==C['parent_snapshot_sha256']
    assert data['source']['zip_sha256']==C['source_sha256']
    t=time.process_time();mean=float(train['duration'].mean());scale=max(float(train['duration'].std(unbiased=False)),1e-5)
    pop=fit_only_popularity(train);preprocessing=time.process_time()-t
    cpu_model=next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),'unknown')
    refs={'prior32':scores(((dev['metadata'][:,-32:,1].sum(-1)+1)/34).numpy(),dev['y'].numpy()),
          'FIT_prevalence':scores(np.full(len(dev['y']),float(train['y'].mean())),dev['y'].numpy())}
    r={'study':C['study'],'contract':C,'contract_sha256':sha(CONTRACT),'source':data['source'],
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_authentication_and_restore_cpu_seconds':auth_cpu,'shared_preprocessing_cpu_seconds':preprocessing,
        'duration_mean':mean,'duration_scale':scale,'models':[],'unmatched_nontrained_references':refs,
        'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,
                       'cpu':cpu_model,'platform':platform.platform(),'threads':1},
        'independent_test_opened':False,'architecture_promoted':False,'strict_architecture_superiority_allowed':False}
    save_json(out/'results.json',r)
    for i,seed in enumerate(C['seeds']):
        for arm in C['arms'][i:]+C['arms'][:i]:
            r['models'].append(fit(arm,seed,train,dev,mean,scale,pop,preprocessing,out));save_json(out/'results.json',r)
    r['whole_study_wall_seconds']=time.perf_counter()-wall;r['whole_study_process_cpu_seconds']=time.process_time()-study_cpu
    save_json(out/'results.json',r)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',default='results/me022');p.add_argument('--out',default='results/me027')
    a=p.parse_args();run(a.parent,a.out)
