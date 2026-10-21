"""ME033 predeclared history/query x embedding-adaptation mechanism study."""
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
from tac_osm.me033_semantic_moments import COUNTS,build_model
from scripts.me032_audit import verify as verify_parent
from scripts.me031_crossfit_research import (
    ROOT,SEEDS,WEIGHTS,subset,targets,roles_manifest,load_source,atomic_checkpoint,file_sha,write_json
)
from scripts.me022_native_readout import scores

ARMS=tuple(COUNTS)
PROTOCOL=ROOT/'research/mathematical-engineering/me033/PROTOCOL.md'
CONTRASTS=('history_vs_query_adapt','history_adaptation','history_vs_common',
           'query_adaptation','history_vs_query_frozen','interaction',
           'native_vs_history','native_vs_common')


def offered_items(data):
    return torch.cat((data['items'][:,[32,64,96]],data['action'].unsqueeze(-1)),-1)


def save_progress(out,name,net,opt,sampler,step,trace,meta):
    atomic_checkpoint(out/(name+'.pt'),{'state_dict':net.state_dict(),'optimizer':opt.state_dict(),
        'sampler_rng':sampler.get_state(),'model_rng':torch.get_rng_state(),'step':step,**meta})
    write_json(out/(name+'-trace.json'),trace)


def fit_one(arm,data,context,common,embedding,seed,name,out):
    start=time.process_time();net=torch.jit.script(build_model(arm,common,embedding,seed))
    body,head=[],[]
    for n,p in net.named_parameters():
        if p.requires_grad:(body if n.startswith('memory.') else head).append(p)
    opt=torch.optim.AdamW([{'params':body,'lr':.001},{'params':head,'lr':.003}],weight_decay=1e-4)
    sampler=torch.Generator().manual_seed(seed+300)
    y=targets(data);offered=offered_items(data);trace=[]
    params=dict(net.named_parameters())
    emb_name='memory.core.emb.weight' if arm=='native_adapt' else 'memory.emb.weight'
    watched=None if arm=='common_adapt' else params[emb_name]
    common_before={k:v.clone() for k,v in net.common.state_dict().items()}
    charged=time.process_time()-start
    for step in range(1,301):
        t=time.process_time();idx=torch.randint(len(y),(128,),generator=sampler)
        net.train();opt.zero_grad(set_to_none=True)
        logits=net(data['items'][idx],data['metadata'][idx],offered[idx],context[idx])
        loss=(F.binary_cross_entropy_with_logits(logits,y[idx],reduction='none')*WEIGHTS).sum(-1).mean()
        if not torch.isfinite(loss):raise ArithmeticError('nonfinite real loss')
        loss.backward();g=torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],1.)
        if not torch.isfinite(g):raise ArithmeticError('nonfinite gradient')
        egrad=0. if watched is None or watched.grad is None else float(watched.grad.norm())
        opt.step();dt=time.process_time()-t;charged+=dt
        trace.append({'step':step,'real_loss':float(loss.detach()),'cpu_seconds':dt,'embedding_grad_after_clip':egrad})
        if step in (100,200,300):save_progress(out,name,net,opt,sampler,step,trace,{'arm':arm,'seed':seed})
        if charged>180:raise RuntimeError('registered S-stage CPU cap exceeded; preserve partial')
    changed=any(not torch.equal(v,net.common.state_dict()[k]) for k,v in common_before.items())
    assert changed==(arm=='common_adapt')
    emb_changed=False if watched is None else not torch.equal(watched.detach(),embedding)
    if arm!='common_adapt':assert emb_changed==('frozen' not in arm)
    net.eval();net.save(str(out/(name+'.jit.pt')))
    return {'arm':arm,'name':name,'seed':seed,'steps':300,'histories':38400,'responses':153600,
        'trainable_parameters':sum(p.numel() for p in net.parameters() if p.requires_grad),
        'stored_model_parameters':sum(p.numel() for p in net.parameters()),
        'charged_cpu_seconds':charged,'common_head_changed':changed,'embedding_changed':emb_changed,
        'nonzero_embedding_gradient_steps':sum(t['embedding_grad_after_clip']>0 for t in trace)}


def run(out):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    start=time.process_time();wall=time.perf_counter();parent_root=ROOT/'results/me032'
    parent=verify_parent(parent_root);data=load_source(ROOT/'results/me022')['train']
    roles=roles_manifest(data);assert roles==json.loads((parent_root/'roles.json').read_text())
    prep=time.process_time()-start;write_json(out/'roles.json',roles)
    record={'study':'ME033_SEMANTIC_HISTORY_QUERY_ADAPTATION','complete':False,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'protocol_sha256':file_sha(PROTOCOL),'parent_results_sha256':file_sha(parent_root/'results.json'),
        'source_snapshot_sha256':file_sha(ROOT/'results/me022/data-snapshot.pt'),
        'python':platform.python_version(),'torch':torch.__version__,'threads':1,
        'cpu':Path('/proc/cpuinfo').read_text().split('model name')[1].splitlines()[0],
        'source_verification_cpu_seconds':prep,'cells':[],'all_models_frozen_before_scoring':False,
        'historical_DEV_scored':False,'reserved_cohorts_opened':False,'candidate_promoted':False,
        'parity':{'parameters':'PASS within history/query pairs; FAIL across freeze/adapt and unequal common reference',
            'data_exposure':'PASS R pretraining/teacher inherited equally; same S examples/targets; identity path intentional ablation',
            'selection':'PASS fixed300 for all arms, no search or outer curves',
            'evaluation':'PASS identical reused FIT-user targets, not chronological/independent qualification',
            'actual_compute':'FAIL exposure diagnostic; equal caps are not equal spend'}}
    write_json(out/'results.json',record)
    for seed in SEEDS:
        for rotation,role in enumerate(roles):
            tag=f'r{rotation}-s{seed}';t=time.process_time()
            pcell=next(c for c in parent['cells'] if c['seed']==seed and c['rotation']==rotation)
            ck=torch.load(parent_root/(tag+'-context.pt'),weights_only=True)
            teacher=torch.load(parent_root/(pcell['teacher']['name']+'.pt'),weights_only=True)
            embedding=ck['state_dict']['emb.weight']
            s=subset(data,torch.tensor(role['readout']['indices']))
            atomic_checkpoint(out/(tag+'-inherited.pt'),{
                'common_state':teacher['state_dict'],'embedding':embedding,
                'context_state':ck['state_dict'],'context_S':ck['S'],'context_T':ck['T'],
                'parent_context_sha256':file_sha(parent_root/(tag+'-context.pt')),
                'parent_teacher_sha256':file_sha(parent_root/(pcell['teacher']['name']+'.pt'))})
            inherited_prep=time.process_time()-t
            offset=(SEEDS.index(seed)+rotation)%len(ARMS);students=[]
            for arm in ARMS[offset:]+ARMS[:offset]:
                students.append(fit_one(arm,s,ck['S'],teacher['state_dict'],embedding,
                    seed+rotation*1000+66000,arm+'-'+tag,out))
            shared=prep+pcell['shared_charged_cpu_seconds']+inherited_prep
            if any(shared+st['charged_cpu_seconds']>300 for st in students):raise RuntimeError('charged pipeline cap exceeded')
            record['cells'].append({'tag':tag,'seed':seed,'rotation':rotation,'students':students,
                'shared_charged_cpu_seconds':shared,'parent_shared_charged_cpu_seconds':pcell['shared_charged_cpu_seconds'],
                'inherited_preparation_cpu_seconds':inherited_prep,
                'parent_context_name':tag+'-context.pt','parent_teacher_name':pcell['teacher']['name']+'.pt'})
            write_json(out/'results.json',record);print(json.dumps({'frozen_cell':tag,'outer_scores_opened':False}),flush=True)
    record['all_models_frozen_before_scoring']=True
    write_json(out/'frozen-payloads.json',{p.name:file_sha(p) for p in out.iterdir() if p.is_file() and p.name!='results.json'})
    write_json(out/'results.json',record);record['evaluation']=evaluate(out,data,roles,record)
    record['complete']=True;record['new_physical_CPU_seconds']=time.process_time()-start
    record['new_physical_wall_seconds']=time.perf_counter()-wall
    record['process_lifetime_peak_RSS_KiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(out/'results.json',record);print(json.dumps(record['evaluation']),flush=True)


def predict(net,data,context):
    offered=offered_items(data);net.eval()
    with torch.no_grad():return torch.cat([net(data['items'][i:i+256],data['metadata'][i:i+256],offered[i:i+256],context[i:i+256])[:,-1].sigmoid()
                                         for i in range(0,len(data['y']),256)]).numpy().astype(np.float64)


def evaluate(out,data,roles,record):
    assert record['all_models_frozen_before_scoring']
    from tac_osm.me032_conditional_memory import CommonHead
    preds={seed:{a:np.zeros(len(data['y'])) for a in ('teacher',*ARMS)} for seed in SEEDS};metrics=[]
    for cell in record['cells']:
        ix=torch.tensor(roles[cell['rotation']]['evaluation']['indices']);t=subset(data,ix)
        inherited=torch.load(out/(cell['tag']+'-inherited.pt'),weights_only=True);ctx=inherited['context_T']
        base=CommonHead();base.load_state_dict(inherited['common_state'],strict=True)
        with torch.no_grad():pp={'teacher':(ctx[:,-1,26]+base(ctx[:,-1,:26])).sigmoid().numpy().astype(np.float64)}
        for st in cell['students']:
            net=torch.jit.load(str(out/(st['name']+'.jit.pt')));pp[st['arm']]=predict(net,t,ctx)
        for arm,p in pp.items():
            preds[cell['seed']][arm][ix.numpy()]=p
            np.save(out/(arm+'-'+cell['tag']+'-predictions.npy'),p)
            metrics.append({'seed':cell['seed'],'rotation':cell['rotation'],'arm':arm,'metrics':scores(p,t['y'].numpy())})
    y=data['y'].numpy().ravel();pooled={str(seed):{a:scores(p,y) for a,p in row.items()} for seed,row in preds.items()}
    users,inv=np.unique(data['users'],return_inverse=True);counts=np.bincount(inv);rng=np.random.default_rng(33033)
    loss={seed:{a:(p-y)**2 for a,p in row.items()} for seed,row in preds.items()}
    delta={k:[] for k in CONTRASTS}
    pairs={'history_vs_query_adapt':('history_adapt','query_adapt'),
        'history_adaptation':('history_adapt','history_frozen'),'history_vs_common':('history_adapt','common_adapt'),
        'query_adaptation':('query_adapt','query_frozen'),'history_vs_query_frozen':('history_frozen','query_frozen'),
        'native_vs_history':('native_adapt','history_adapt'),'native_vs_common':('native_adapt','common_adapt')}
    for seed in SEEDS:
        for name,(a,b) in pairs.items():delta[name].append(loss[seed][a]-loss[seed][b])
        delta['interaction'].append((loss[seed]['history_adapt']-loss[seed]['query_adapt'])-
                                     (loss[seed]['history_frozen']-loss[seed]['query_frozen']))
    contrasts={}
    for name in CONTRASTS:
        event=np.mean(delta[name],axis=0);sums=np.bincount(inv,weights=event);boot=[]
        for _ in range(1000):
            draw=rng.integers(len(users),size=len(users));boot.append(sums[draw].sum()/counts[draw].sum())
        contrasts[name]={'per_seed_Brier':[float(d.mean()) for d in delta[name]],'mean_Brier':float(event.mean()),
            'user_cluster_CI95':np.quantile(boot,[.025,.975]).tolist()}
    def passes(name,threshold):
        c=contrasts[name];return all(x<0 for x in c['per_seed_Brier']) and c['mean_Brier']<=threshold and c['user_cluster_CI95'][1]<0
    def calibration(arm,controls):
        return all(np.mean([pooled[str(seed)][arm]['ECE10']-pooled[str(seed)][c]['ECE10'] for seed in SEEDS])<=.003 for c in controls)
    history_gates={n:passes(n,-.0005) for n in ('history_vs_query_adapt','history_adaptation','history_vs_common')}
    history_gates['calibration']=calibration('history_adapt',('query_adapt','history_frozen','common_adapt'))
    history_gates['candidate']=all(history_gates.values())
    native_gates={n:passes(n,-.0005) for n in ('native_vs_history','native_vs_common')}
    native_gates['calibration']=calibration('native_adapt',('history_adapt','common_adapt'))
    native_gates['candidate']=all(native_gates.values())
    return {'pooled_per_seed':pooled,'metrics_per_cell':metrics,'contrasts':contrasts,
        'history_gates':history_gates,'native_gates':native_gates,'adaptation_specific_interaction':passes('interaction',-.0002),
        'bootstrap_seed':33033,'bootstrap_draws':1000,'users':len(users),'rows_per_seed':len(y),
        'conditional_mutual_information_estimated':False,'independent_qualification':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/me033');a=p.parse_args();run(Path(a.out))
