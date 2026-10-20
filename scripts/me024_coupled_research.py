"""Fixed native coupling/RMS experiment; only authenticated exposed data."""
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
from tac_osm.coupled_native import CoupledNative
from scripts.me014b_exact_action_memory import batch
from scripts.me022_native_readout import predictions, scores, save_json
from scripts.me023_predictive_research import atomic_checkpoint, sha, load_source as original_source

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT/"research/mathematical-engineering/me024/contract.json"
C = json.loads(CONTRACT.read_text())


def load_source(parent="results/me022", previous="results/me023"):
    r = json.loads((Path(previous)/"results.json").read_text())
    assert len(r["models"]) == 12
    for row in r["models"]:
        assert sha(Path(previous)/row["checkpoint"]) == row["checkpoint_sha256"]
    assert sha(Path(parent)/"data-snapshot.pt") == C["parent_data_snapshot_sha256"]
    return original_source(parent)


def make_model(arm, mean, scale):
    return CoupledNative(C["coupling"] if arm.startswith("coupled") else 0.,
                         "rms" in arm, mean, scale)


def feature_diagnostic(model, train, dev):
    model.eval()
    with torch.no_grad():
        f = [model.state_features(model.encode(train["items"][i:i+256],train["metadata"][i:i+256]))
             for i in range(0,1024,256)]
        mean = torch.cat(f).mean(0,keepdim=True)
        out=[]
        for i in range(0,len(dev["y"]),384):
            sl=slice(i,i+384); meta=dev["metadata"][sl]
            prior=(meta[:,-32:,1].sum(-1,keepdim=True)+1)/34
            out.append(torch.sigmoid(model.readout_features(mean.expand(len(meta),-1),
                dev["action"][sl],dev["duration"][sl],prior)).flatten())
        p=torch.cat(out).numpy()
    return scores(p,dev["y"].numpy()),mean,p


def fit(arm,seed,train,dev,mean,scale,root):
    torch.manual_seed(seed)
    model=torch.jit.script(make_model(arm,mean,scale))
    assert sum(p.numel() for p in model.parameters()) == C["parameters"]
    groups=[[],[]]
    for name,p in model.named_parameters(): groups[int(name.startswith('head.'))].append(p)
    opt=torch.optim.AdamW([{'params':groups[0],'lr':C['base_lr']},
                           {'params':groups[1],'lr':C['readout_lr']}],weight_decay=C['weight_decay'])
    gen=torch.Generator().manual_seed(seed+300)
    name=f'{arm}-seed{seed}'; checkpoint=root/f'{name}.pt'
    trace=[]; cpu=wall=0.
    def record(step,grad=None):
        p=predictions(model,dev)
        trace.append({'step':step,'dev':scores(p,dev['y'].numpy()),'train_cpu_seconds':cpu,
                      'train_wall_seconds':wall,'grad_norm_before_clip':grad})
        save_json(root/f'{name}-trace.json',trace)
        atomic_checkpoint(checkpoint,{'state_dict':model.state_dict(),'optimizer':opt.state_dict(),
            'rng':torch.get_rng_state(),'sampler_rng':gen.get_state(),'arm':arm,'seed':seed,
            'step':step,'duration_mean':mean,'duration_scale':scale,'contract':C,
            'train_cpu_seconds':cpu,'train_wall_seconds':wall})
        return p
    record(0)
    weights=torch.tensor(C['prefix_loss_weights']).reshape(1,4)
    for step in range(1,C['updates']+1):
        t,w=time.process_time(),time.perf_counter(); model.train()
        idx=torch.randint(len(train['y']),(C['batch'],),generator=gen)
        opt.zero_grad(set_to_none=True)
        logits=model.prefix_logits(*batch(train,idx))
        target=torch.cat((train['metadata'][idx][:,C['prefix_depths'],1],train['y'][idx]),-1)
        loss=(F.binary_cross_entropy_with_logits(logits,target,reduction='none')*weights).sum(-1).mean()
        assert torch.isfinite(loss)
        loss.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),C['clip_grad'])
        assert torch.isfinite(grad)
        opt.step(); cpu+=time.process_time()-t; wall+=time.perf_counter()-w
        if step in C['trace_steps'] or cpu>C['cpu_seconds_ceiling_per_fit']: p=record(step,float(grad))
        if cpu>C['cpu_seconds_ceiling_per_fit']: raise RuntimeError('Compute ceiling exceeded; partial state retained')
    replaced,feature_mean,fp=feature_diagnostic(model,train,dev)
    # Frozen trajectory intervention. It is not a separately trained control.
    off=torch.jit.script(CoupledNative(0.,'rms' in arm,mean,scale))
    off.load_state_dict(model.state_dict(),strict=True)
    off_p=predictions(off,dev)
    np.savez_compressed(root/f'{name}-predictions.npz',probability=p,
        mean_feature_probability=fp,feedback_disabled_probability=off_p)
    atomic_checkpoint(root/f'{name}-feature-mean.pt',{'features':feature_mean})
    standalone=root/f'{name}.jit.pt'; torch.jit.save(model.eval(),str(standalone))
    times=[]
    with torch.no_grad():
        for j in range(10):
            t=time.perf_counter(); model(*batch(dev,slice(0,128)))
            if j>=3: times.append(time.perf_counter()-t)
    row={'arm':arm,'seed':seed,'parameters':C['parameters'],'steps':step,
        'sampled_windows':step*C['batch'],'historical_event_presentations':step*C['batch']*128,
        'supervised_response_presentations':step*C['batch']*4,'dev':trace[-1]['dev'],
        'feature_mean_dev':replaced,'feature_mean_brier_delta':replaced['Brier']-trace[-1]['dev']['Brier'],
        'feedback_disabled_dev':scores(off_p,dev['y'].numpy()),'train_cpu_seconds':cpu,'train_wall_seconds':wall,
        'checkpoint':checkpoint.name,'checkpoint_sha256':sha(checkpoint),
        'standalone':standalone.name,'standalone_sha256':sha(standalone),
        'inference_median_seconds_batch128':float(np.median(times)),
        'process_lifetime_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'trace':trace}
    print(json.dumps({k:row[k] for k in ('arm','seed','dev','feature_mean_brier_delta','train_cpu_seconds')}),flush=True)
    return row


def run(parent,previous,out):
    torch.set_num_threads(1); root=Path(out); root.mkdir(parents=True,exist_ok=False)
    start,pcpu=time.perf_counter(),time.process_time()
    data=load_source(parent,previous); train,dev=data['train'],data['dev']
    mean=float(train['duration'].mean()); scale=max(float(train['duration'].std(unbiased=False)),1e-5)
    r={'study':C['study'],'contract':C,'source':data['source'],'contract_sha256':sha(CONTRACT),
       'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
       'duration_mean':mean,'duration_scale':scale,'models':[],
       'environment':{'torch':str(torch.__version__),'numpy':np.__version__,'platform':platform.platform(),'threads':1},
       'historical_development_only':True,'new_test_outcomes':0,'full_compute_parity':False,'architecture_promoted':False}
    save_json(root/'results.json',r)
    for i,seed in enumerate(C['seeds']):
        for arm in C['arms'][i:]+C['arms'][:i]:
            r['models'].append(fit(arm,seed,train,dev,mean,scale,root)); save_json(root/'results.json',r)
    r['execution_wall_seconds']=time.perf_counter()-start
    r['execution_process_cpu_seconds']=time.process_time()-pcpu
    save_json(root/'results.json',r)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--parent',default='results/me022')
    p.add_argument('--previous',default='results/me023'); p.add_argument('--out',default='results/me024')
    a=p.parse_args(); run(a.parent,a.previous,a.out)
