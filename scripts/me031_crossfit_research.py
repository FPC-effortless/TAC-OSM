"""ME031 three-way FIT-user cross-fitted conditional residual prediction.

No historical DEV scores or reserved-cohort access. Freeze encoder, shared base,
and all residual probes before scoring any outer evaluation predictions.
"""
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
from tac_osm.me031_crossfit_state import CrossfitNativeEncoder, ResidualProbe, role_indices
from scripts.me023_predictive_research import load_source, atomic_checkpoint
from scripts.me025_real_transformer_parity import fit_only_popularity
from scripts.me014b_exact_action_memory import batch
from scripts.me028_native_readout_fit import file_sha, write_json
from scripts.me022_native_readout import scores

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'research/mathematical-engineering/me031/PROTOCOL.md'
SEEDS=(14001,14002,14003)
ARMS=('common_residual','state_residual')
WEIGHTS=torch.tensor([1/6,1/6,1/6,1/2]).reshape(1,4)


def subset(data,indices):
    return {k:(v[indices] if isinstance(v,torch.Tensor) else [v[i] for i in indices.tolist()] if isinstance(v,list) else v)
            for k,v in data.items()}


def targets(data):
    return torch.cat((data['metadata'][:,[32,64,96],1],data['y']),-1)


def save_model(root,name,net,opt,sampler,step,trace,metadata):
    atomic_checkpoint(root/(name+'.pt'),{'state_dict':net.state_dict(),
        'optimizer':opt.state_dict(),'sampler_rng':sampler.get_state(),
        'model_rng':torch.get_rng_state(),'step':step,**metadata})
    write_json(root/(name+'-trace.json'),trace)


def train_encoder(data,seed,rotation,root):
    start=time.process_time()
    mean=float(data['duration'].mean());scale=max(float(data['duration'].std(unbiased=False)),1e-5)
    pop=fit_only_popularity(data)
    torch.manual_seed(seed)
    net=torch.jit.script(CrossfitNativeEncoder(mean,scale,pop))
    body,head=[],[]
    for n,p in net.named_parameters():
        (head if n.startswith(('head.','observable_hidden.','observable_output.')) else body).append(p)
    opt=torch.optim.AdamW([{'params':body,'lr':.001},{'params':head,'lr':.006}],weight_decay=1e-4)
    sampler=torch.Generator().manual_seed(seed+300)
    charged=time.process_time()-start;trace=[];y=targets(data)
    name=f'encoder-r{rotation}-s{seed}'
    for step in range(1,301):
        t=time.process_time()
        idx=torch.randint(len(y),(128,),generator=sampler)
        opt.zero_grad(set_to_none=True)
        logits=net.prefix_logits(*batch(data,idx))
        loss=(F.binary_cross_entropy_with_logits(logits,y[idx],reduction='none')*WEIGHTS).sum(-1).mean()
        if not torch.isfinite(loss):raise ArithmeticError('nonfinite encoder loss')
        loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        if not torch.isfinite(gn):raise ArithmeticError('nonfinite encoder gradient')
        opt.step();dt=time.process_time()-t;charged+=dt
        trace.append({'step':step,'loss':float(loss.detach()),'cpu_seconds':dt})
        if step in (100,200,300):
            save_model(root,name,net,opt,sampler,step,trace,{'seed':seed,'rotation':rotation,
                'mean':mean,'scale':scale,'popularity':pop})
        if charged>180:raise RuntimeError('encoder CPU cap exceeded; preserve partial')
    net.eval();net.save(str(root/(name+'.jit.pt')))
    for p in net.parameters():p.requires_grad_(False)
    return net,{'name':name,'seed':seed,'rotation':rotation,'charged_cpu_seconds':charged,
                'steps':300,'histories':38400,'responses':153600,'parameters':135377}


def extract(net,data):
    values=[]
    with torch.no_grad():
        for i in range(0,len(data['y']),256):
            values.append(net.prefix_features(*batch(data,slice(i,i+256))))
    return torch.cat(values)


def head_logits(net,features,offset):
    net.eval()
    with torch.no_grad():
        return torch.cat([net(features[i:i+512],offset[i:i+512])
                          for i in range(0,len(features),512)])


def train_probe(features,y,offset,with_state,seed,root,name,steps=500):
    start=time.process_time();torch.manual_seed(seed)
    net=torch.jit.script(ResidualProbe(with_state))
    opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=1e-4)
    sampler=torch.Generator().manual_seed(seed+300)
    charged=time.process_time()-start;trace=[]
    for step in range(1,steps+1):
        t=time.process_time()
        idx=torch.randint(len(y),(128,),generator=sampler)
        net.train();opt.zero_grad(set_to_none=True)
        loss=(F.binary_cross_entropy_with_logits(net(features[idx],offset[idx]),y[idx],
              reduction='none')*WEIGHTS).sum(-1).mean()
        if not torch.isfinite(loss):raise ArithmeticError('nonfinite probe loss')
        loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        if not torch.isfinite(gn):raise ArithmeticError('nonfinite probe gradient')
        opt.step();dt=time.process_time()-t;charged+=dt
        trace.append({'step':step,'loss':float(loss.detach()),'cpu_seconds':dt})
        if step in (100,250,500):
            save_model(root,name,net,opt,sampler,step,trace,{'seed':seed,'with_state':with_state})
        if charged>30:raise RuntimeError('probe CPU cap exceeded; preserve partial')
    net.eval();net.save(str(root/(name+'.jit.pt')))
    for p in net.parameters():p.requires_grad_(False)
    return net,{'name':name,'steps':steps,'histories':steps*128,'responses':steps*128*4,
                'charged_cpu_seconds':charged,'parameters':sum(p.numel() for p in net.parameters()),
                'seed':seed,'with_state':with_state}


def roles_manifest(data):
    users=torch.tensor(data['users'],dtype=torch.long)
    result=[]
    for rotation in range(3):
        ix=role_indices(users,rotation)
        sets=[set(users[i].tolist()) for i in ix]
        assert all(not sets[a]&sets[b] for a,b in ((0,1),(0,2),(1,2)))
        result.append({'rotation':rotation,**{name:{'indices':idx.tolist(),
            'users':sorted(us),'windows':len(idx),'user_count':len(us)}
            for name,idx,us in zip(('representation','readout','evaluation'),ix,sets)}})
    return result


def run(out):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    start=time.process_time();wall=time.perf_counter()
    data=load_source(ROOT/'results/me022')['train']
    roles=roles_manifest(data)
    prep=time.process_time()-start
    write_json(out/'roles.json',roles)
    record={'study':'ME031_FIT_USER_CROSSFIT_RESIDUAL_STATE_VALUE','complete':False,
        'protocol_sha256':file_sha(PROTOCOL),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_snapshot_sha256':file_sha(ROOT/'results/me022/data-snapshot.pt'),
        'python':platform.python_version(),'torch':torch.__version__,'threads':1,
        'cpu':Path('/proc/cpuinfo').read_text().split('model name')[1].splitlines()[0],
        'historical_DEV_scored':False,'reserved_cohorts_opened':False,
        'source_FIT_rows':len(data['y']),'source_preparation_cpu_seconds':prep,
        'cells':[],'all_models_frozen_before_outer_scoring':False,
        'conditional_mutual_information_estimated':False,'candidate_promoted':False,
        'parity':{'trained_probe_parameters':'PASS:1197 versus1205; frozen135377 encoder plus1205 baseline shared',
                  'data_exposure':'PASS common sampling; state access is intentional diagnostic ablation',
                  'selection':'PASS fixed endpoints; no target scores during fitting',
                  'evaluation':'PASS identical predeclared FIT-user targets; not independent qualification',
                  'actual_compute':'FAIL: matched exposure, unequal actual spend; no architecture superiority'}}
    write_json(out/'results.json',record)
    for seed in SEEDS:
        for role in roles:
            rotation=role['rotation'];tag=f'r{rotation}-s{seed}'
            cell_start=time.process_time()
            r,s,t=[subset(data,torch.tensor(role[k]['indices'])) for k in ('representation','readout','evaluation')]
            split_cpu=time.process_time()-cell_start
            enc,enc_record=train_encoder(r,seed,rotation,out)
            e=time.process_time();fs=extract(enc,s);ft=extract(enc,t)
            # FIT readout-user moments only, no labels and no evaluation-user moments.
            mu=fs[...,:90].reshape(-1,90).mean(0)
            scale=fs[...,:90].reshape(-1,90).std(0,unbiased=False).clamp_min(.05)
            fs[...,:90]=(fs[...,:90]-mu)/scale;ft[...,:90]=(ft[...,:90]-mu)/scale
            feature_cpu=time.process_time()-e
            features={'readout':fs,'evaluation':ft,'mean':mu,'scale':scale}
            atomic_checkpoint(out/(tag+'-features.pt'),features)
            ys=targets(s)
            headseed=seed+rotation*1000+10000
            base,base_record=train_probe(fs,ys,fs[...,90],False,headseed,out,'base-'+tag)
            # Compute frozen offsets without reading outer evaluation labels.
            offset_start=time.process_time()
            offset_s=head_logits(base,fs,fs[...,90]);offset_t=head_logits(base,ft,ft[...,90])
            offset_cpu=time.process_time()-offset_start
            probes=[]
            order=ARMS if (rotation+SEEDS.index(seed))%2==0 else ARMS[::-1]
            for arm in order:
                _,pr=train_probe(fs,ys,offset_s,arm=='state_residual',headseed+10000,out,arm+'-'+tag)
                probes.append({'arm':arm,**pr})
            cell={'tag':tag,'seed':seed,'rotation':rotation,'encoder':enc_record,'base':base_record,
                  'probes':probes,'feature_construction_cpu_seconds':feature_cpu,
                  'role_subset_cpu_seconds':split_cpu,'base_offset_cpu_seconds':offset_cpu,
                  'shared_charged_cpu_seconds':prep+split_cpu+enc_record['charged_cpu_seconds']+feature_cpu+base_record['charged_cpu_seconds']+offset_cpu,
                  'whole_cell_cpu_seconds':time.process_time()-cell_start}
            if max(cell['shared_charged_cpu_seconds']+p['charged_cpu_seconds'] for p in probes)>240:
                raise RuntimeError('per-arm whole pipeline charged CPU cap exceeded')
            record['cells'].append(cell);write_json(out/'results.json',record)
            print(json.dumps({'completed_frozen_cell':tag,'evaluation_not_scored':True}),flush=True)
    record['all_models_frozen_before_outer_scoring']=True
    # Freeze complete raw payload hashes before first outer score.
    index={p.name:file_sha(p) for p in out.iterdir() if p.suffix in ('.pt','.json') and p.name!='results.json'}
    write_json(out/'frozen-payloads.json',index)
    write_json(out/'results.json',record)
    record['evaluation']=evaluate_frozen(out,data,roles,record)
    record['complete']=True
    record['whole_study_cpu_seconds']=time.process_time()-start
    record['whole_study_wall_seconds']=time.perf_counter()-wall
    record['process_lifetime_peak_RSS_KiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(out/'results.json',record)
    print(json.dumps(record['evaluation'],sort_keys=True),flush=True)


def evaluate_frozen(out,data,roles,record):
    assert record['all_models_frozen_before_outer_scoring']
    by_seed={seed:{arm:np.zeros(len(data['y']),dtype=np.float64) for arm in ('prior','base',*ARMS)} for seed in SEEDS}
    metrics=[]
    for cell in record['cells']:
        role=roles[cell['rotation']];ix=torch.tensor(role['evaluation']['indices'])
        y=data['y'][ix].numpy();cache=torch.load(out/(cell['tag']+'-features.pt'),weights_only=True)
        f=cache['evaluation'];prior=f[...,90]
        base=torch.jit.load(str(out/(cell['base']['name']+'.jit.pt')))
        offset=head_logits(base,f,prior)
        logits={'prior':prior,'base':offset}
        for pr in cell['probes']:
            net=torch.jit.load(str(out/(pr['name']+'.jit.pt')))
            logits[pr['arm']]=head_logits(net,f,offset)
        for arm,logit in logits.items():
            p=torch.sigmoid(logit[:,-1]).numpy().astype(np.float64)
            np.save(out/(arm+'-'+cell['tag']+'-predictions.npy'),p)
            by_seed[cell['seed']][arm][ix.numpy()]=p
            metrics.append({'seed':cell['seed'],'rotation':cell['rotation'],'arm':arm,'metrics':scores(p,y)})
    truth=data['y'].numpy().ravel();users,inv=np.unique(data['users'],return_inverse=True)
    pooled={str(seed):{arm:scores(p,truth) for arm,p in arms.items()} for seed,arms in by_seed.items()}
    ds=[];vsbase=[];ece=[];folds=[]
    for seed in SEEDS:
        arms=by_seed[seed]
        ds.append((arms['state_residual']-truth)**2-(arms['common_residual']-truth)**2)
        vsbase.append(float(np.mean((arms['state_residual']-truth)**2-(arms['base']-truth)**2)))
        ece.append(pooled[str(seed)]['state_residual']['ECE10']-pooled[str(seed)]['common_residual']['ECE10'])
        wins=0
        for rot in range(3):
            m={row['arm']:row['metrics']['Brier'] for row in metrics if row['seed']==seed and row['rotation']==rot}
            wins+=m['state_residual']<m['common_residual']
        folds.append(wins)
    event=np.mean(ds,axis=0);sums=np.bincount(inv,weights=event);count=np.bincount(inv)
    rng=np.random.default_rng(31031);boot=[]
    for _ in range(1000):
        draw=rng.integers(len(users),size=len(users));boot.append(sums[draw].sum()/count[draw].sum())
    ci=np.quantile(boot,[.025,.975]).tolist()
    paired=[float(d.mean()) for d in ds]
    gates={'state_beats_common':all(d<0 for d in paired) and np.mean(paired)<=-.0005,
           'state_beats_base':all(d<0 for d in vsbase) and np.mean(vsbase)<=-.0005,
           'at_least_two_folds_per_seed':all(w>=2 for w in folds),
           'calibration':float(np.mean(ece))<=.003,'bootstrap_upper_below_zero':ci[1]<0}
    gates['finite_family_state_candidate']=all(gates.values())
    return {'metrics_per_cell':metrics,'pooled_per_seed':pooled,
        'state_minus_common_per_seed_Brier':paired,'state_minus_common_mean_Brier':float(np.mean(paired)),
        'state_minus_base_per_seed_Brier':vsbase,'fold_wins_per_seed':folds,
        'mean_state_minus_common_ECE10':float(np.mean(ece)),
        'user_cluster_CI95':ci,'bootstrap_draws':1000,'bootstrap_seed':31031,
        'evaluation_users':len(users),'evaluation_rows_per_seed':len(truth),
        'gates':gates,'conditional_mutual_information_estimated':False,
        'previously_used_FIT_source_not_independent_test':True}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/me031');a=p.parse_args()
    run(Path(a.out))
