"""ME032 centered correction with memory learning, source-only user cross-fit."""
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
from tac_osm.me032_conditional_memory import FrozenContext,CommonHead,CenteredReadout,make_student,COUNTS
from scripts.me031_crossfit_research import (
    ROOT,SEEDS,WEIGHTS,subset,targets,roles_manifest,load_source,atomic_checkpoint,file_sha,write_json
)
from scripts.me031_audit import verify_payloads as verify_parent
from scripts.me014b_exact_action_memory import batch
from scripts.me022_native_readout import scores

ARMS=tuple(COUNTS)
PROTOCOL=ROOT/'research/mathematical-engineering/me032/PROTOCOL.md'


def context_features(net,data):
    net.eval()
    with torch.no_grad():return torch.cat([net(*batch(data,slice(i,i+256))) for i in range(0,len(data['y']),256)])


def build_student(arm,common_state,seed):
    torch.manual_seed(seed);net=make_student(arm,common_state)
    # Identical correction matrices across native and pool, irrespective of body RNG consumption.
    torch.manual_seed(seed+1);net.correction=CenteredReadout()
    return net


def save_progress(out,name,net,opt,sampler,step,trace,meta):
    atomic_checkpoint(out/(name+'.pt'),{'state_dict':net.state_dict(),'optimizer':opt.state_dict(),
        'sampler_rng':sampler.get_state(),'model_rng':torch.get_rng_state(),'step':step,**meta})
    write_json(out/(name+'-trace.json'),trace)


def fit_teacher(features,y,seed,name,out):
    start=time.process_time();torch.manual_seed(seed);net=torch.jit.script(CommonHead())
    opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=1e-4)
    sampler=torch.Generator().manual_seed(seed+300);trace=[];charged=time.process_time()-start
    for step in range(1,501):
        t=time.process_time();idx=torch.randint(len(y),(128,),generator=sampler)
        opt.zero_grad(set_to_none=True)
        logits=features[idx,:,26]+net(features[idx,:,:26])
        loss=(F.binary_cross_entropy_with_logits(logits,y[idx],reduction='none')*WEIGHTS).sum(-1).mean()
        if not torch.isfinite(loss):raise ArithmeticError('nonfinite teacher loss')
        loss.backward();g=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        if not torch.isfinite(g):raise ArithmeticError('nonfinite teacher gradient')
        opt.step();dt=time.process_time()-t;charged+=dt
        trace.append({'step':step,'loss':float(loss.detach()),'cpu_seconds':dt})
        if step in (100,250,500):save_progress(out,name,net,opt,sampler,step,trace,{'seed':seed})
        if charged>30:raise RuntimeError('teacher cap exceeded; preserve partial')
    net.eval();net.save(str(out/(name+'.jit.pt')))
    return net,{'name':name,'seed':seed,'steps':500,'charged_cpu_seconds':charged,
                'trainable_parameters':1205,'histories':64000,'responses':256000}


def fit_student(arm,data,context,common_state,seed,name,out):
    start=time.process_time();net=torch.jit.script(build_student(arm,common_state,seed))
    groups=[[],[]]
    for n,p in net.named_parameters():
        if p.requires_grad:groups[int(not n.startswith('memory.'))].append(p)
    opt=torch.optim.AdamW([{'params':groups[0],'lr':.001},{'params':groups[1],'lr':.003}],weight_decay=1e-4)
    sampler=torch.Generator().manual_seed(seed+300);trace=[];y=targets(data)
    original_common={k:v.clone() for k,v in net.common.state_dict().items()}
    memory_parameter=dict(net.named_parameters())['memory.pool.0.weight' if arm=='conditional_pool' else 'memory.transition.weight']
    charged=time.process_time()-start
    for step in range(1,301):
        t=time.process_time();idx=torch.randint(len(y),(128,),generator=sampler)
        net.train();opt.zero_grad(set_to_none=True)
        logits=net(data['items'][idx],data['metadata'][idx],context[idx])
        loss=(F.binary_cross_entropy_with_logits(logits,y[idx],reduction='none')*WEIGHTS).sum(-1).mean()
        if not torch.isfinite(loss):raise ArithmeticError('nonfinite student loss')
        loss.backward();g=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        if not torch.isfinite(g):raise ArithmeticError('nonfinite student gradient')
        # Measured after clipping, before update; observes actual memory credit path.
        memory_gradient=float(memory_parameter.grad.norm())
        opt.step();dt=time.process_time()-t;charged+=dt
        trace.append({'step':step,'loss':float(loss.detach()),'cpu_seconds':dt,
                      'memory_gradient_norm_after_clip':memory_gradient})
        if step in (100,200,300):save_progress(out,name,net,opt,sampler,step,trace,{'seed':seed,'arm':arm})
        if charged>180:raise RuntimeError('student cap exceeded; preserve partial')
    unchanged=all(torch.equal(v,net.common.state_dict()[k]) for k,v in original_common.items())
    assert unchanged==(arm!='joint_native')
    assert any(t['memory_gradient_norm_after_clip']>0 for t in trace[1:])
    net.eval();net.save(str(out/(name+'.jit.pt')))
    return {'arm':arm,'name':name,'seed':seed,'steps':300,'charged_cpu_seconds':charged,
        'trainable_parameters':sum(p.numel() for p in net.parameters() if p.requires_grad),
        'stored_parameters':sum(p.numel() for p in net.parameters()),
        'histories':38400,'responses':153600,'common_head_bit_exact_unchanged':unchanged,
        'positive_memory_gradient_steps':sum(t['memory_gradient_norm_after_clip']>0 for t in trace),
        'sampler_seed':seed+300}


def run(out):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    start=time.process_time();wall=time.perf_counter()
    parent_root=ROOT/'results/me031';parent=verify_parent(parent_root)
    data=load_source(ROOT/'results/me022')['train'];roles=roles_manifest(data)
    assert roles==json.loads((parent_root/'roles.json').read_text())
    prep=time.process_time()-start;write_json(out/'roles.json',roles)
    record={'study':'ME032_CENTERED_CONDITIONAL_MEMORY_TRAINING','complete':False,
        'protocol_sha256':file_sha(PROTOCOL),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_snapshot_sha256':file_sha(ROOT/'results/me022/data-snapshot.pt'),
        'parent_results_sha256':file_sha(parent_root/'results.json'),'source_preparation_cpu_seconds':prep,
        'python':platform.python_version(),'torch':torch.__version__,'threads':1,
        'cpu':Path('/proc/cpuinfo').read_text().split('model name')[1].splitlines()[0],
        'cells':[],'candidate_promoted':False,'historical_DEV_scored':False,'reserved_cohorts_opened':False,
        'all_models_frozen_before_evaluation':False,'stage_trainable_counts':COUNTS,
        'parity':{'parameters':'PASS <=1% active S-stage gap; shared historical teacher reported separately',
            'data_exposure':'PASS shared R teacher, identical S samples/targets; no student memory pretraining',
            'selection':'PASS fixed300 S /500 R endpoints, no tuning',
            'evaluation':'PASS identical original FIT-user targets, not independent qualification',
            'actual_compute':'FAIL exposure-matched diagnostic, no fully matched compute claim'}}
    write_json(out/'results.json',record)
    for seed in SEEDS:
        for rotation,role in enumerate(roles):
            tag=f'r{rotation}-s{seed}';cell_start=time.process_time()
            r,s,t=[subset(data,torch.tensor(role[k]['indices'])) for k in ('representation','readout','evaluation')]
            pc=next(c for c in parent['cells'] if c['seed']==seed and c['rotation']==rotation)
            name=pc['encoder']['name'];ck=torch.load(parent_root/(name+'.pt'),weights_only=True)
            # Only R encoder's embedding and preprocessing are inherited, never S base/probes.
            context=FrozenContext(ck['state_dict']['emb.weight'],ck['mean'],ck['scale'],ck['popularity'],torch.zeros(26),torch.ones(26))
            raw=context_features(torch.jit.script(context),r)
            context.feature_mean.copy_(raw[...,:26].reshape(-1,26).mean(0))
            context.feature_scale.copy_(raw[...,:26].reshape(-1,26).std(0,unbiased=False).clamp_min(.05))
            context=torch.jit.script(context).eval()
            cr,cs,ct=[context_features(context,part) for part in (r,s,t)]
            y_r=targets(r)
            feature_cpu=time.process_time()-cell_start
            context.save(str(out/('context-'+tag+'.jit.pt')))
            atomic_checkpoint(out/(tag+'-context.pt'),{'R':cr,'S':cs,'T':ct,
                'state_dict':context.state_dict(),'parent_encoder_sha256':file_sha(parent_root/(name+'.pt')),
                'parent_name':name,'mean':ck['mean'],'scale':ck['scale']})
            teacher,teacher_record=fit_teacher(cr,y_r,seed+rotation*1000+32000,'teacher-'+tag,out)
            state={k:v.detach().clone() for k,v in teacher.state_dict().items()}
            students=[];offset=(SEEDS.index(seed)+rotation)%3
            for arm in ARMS[offset:]+ARMS[:offset]:
                students.append(fit_student(arm,s,cs,state,seed+rotation*1000+64000,arm+'-'+tag,out))
            shared=prep+pc['encoder']['charged_cpu_seconds']+feature_cpu+teacher_record['charged_cpu_seconds']
            if any(shared+p['charged_cpu_seconds']>300 for p in students):raise RuntimeError('charged pipeline cap exceeded')
            record['cells'].append({'tag':tag,'seed':seed,'rotation':rotation,
                'parent_encoder_name':name,'parent_encoder_sha256':file_sha(parent_root/(name+'.pt')),
                'parent_encoder_train_cpu_seconds':pc['encoder']['charged_cpu_seconds'],
                'parent_encoder_R_steps':300,'feature_building_cpu_seconds':feature_cpu,
                'teacher':teacher_record,'students':students,'shared_charged_cpu_seconds':shared,
                'whole_cell_cpu_seconds':time.process_time()-cell_start})
            write_json(out/'results.json',record)
            print(json.dumps({'frozen_cell':tag,'outer_evaluation_not_scored':True}),flush=True)
    record['all_models_frozen_before_evaluation']=True
    write_json(out/'frozen-payloads.json',{p.name:file_sha(p) for p in out.iterdir() if p.is_file() and p.name!='results.json'})
    write_json(out/'results.json',record)
    record['evaluation']=evaluate(out,data,roles,record)
    record['complete']=True;record['new_study_physical_CPU_seconds']=time.process_time()-start
    record['new_study_wall_seconds']=time.perf_counter()-wall
    record['process_lifetime_peak_RSS_KiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(out/'results.json',record);print(json.dumps(record['evaluation']),flush=True)


def predict_student(net,data,ctx):
    net.eval()
    with torch.no_grad():return torch.cat([net(data['items'][i:i+256],data['metadata'][i:i+256],ctx[i:i+256])[:,-1].sigmoid()
                                         for i in range(0,len(data['y']),256)]).numpy().astype(np.float64)


def evaluate(out,data,roles,record):
    assert record['all_models_frozen_before_evaluation']
    preds={seed:{arm:np.zeros(len(data['y'])) for arm in ('prior','teacher',*ARMS)} for seed in SEEDS};metrics=[]
    for cell in record['cells']:
        ix=torch.tensor(roles[cell['rotation']]['evaluation']['indices']);part=subset(data,ix)
        ctx=torch.load(out/(cell['tag']+'-context.pt'),weights_only=True)['T']
        teacher=torch.jit.load(str(out/(cell['teacher']['name']+'.jit.pt'))).eval()
        with torch.no_grad():
            probabilities={'prior':ctx[:,-1,26].sigmoid().numpy().astype(np.float64),
                'teacher':(ctx[:,-1,26]+teacher(ctx[:,-1,:26])).sigmoid().numpy().astype(np.float64)}
        for student in cell['students']:
            net=torch.jit.load(str(out/(student['name']+'.jit.pt')))
            probabilities[student['arm']]=predict_student(net,part,ctx)
        for arm,p in probabilities.items():
            preds[cell['seed']][arm][ix.numpy()]=p
            np.save(out/(arm+'-'+cell['tag']+'-predictions.npy'),p)
            metrics.append({'seed':cell['seed'],'rotation':cell['rotation'],'arm':arm,'metrics':scores(p,part['y'].numpy())})
    y=data['y'].numpy().ravel();pooled={str(seed):{arm:scores(p,y) for arm,p in arms.items()} for seed,arms in preds.items()}
    users,inv=np.unique(data['users'],return_inverse=True);count=np.bincount(inv);rng=np.random.default_rng(32032)
    contrasts={}
    for control in ('joint_native','conditional_pool','teacher'):
        diffs=[(preds[seed]['conditional_native']-y)**2-(preds[seed][control]-y)**2 for seed in SEEDS]
        event=np.mean(diffs,axis=0);sums=np.bincount(inv,weights=event);boot=[]
        for _ in range(1000):
            draw=rng.integers(len(users),size=len(users));boot.append(sums[draw].sum()/count[draw].sum())
        contrasts[control]={'per_seed_Brier':[float(d.mean()) for d in diffs],'mean_Brier':float(event.mean()),
            'user_cluster_CI95':np.quantile(boot,[.025,.975]).tolist()}
    wins=[]
    for seed in SEEDS:
        counts=[]
        for control in ('joint_native','conditional_pool'):
            n=0
            for rot in range(3):
                m={r['arm']:r['metrics']['Brier'] for r in metrics if r['seed']==seed and r['rotation']==rot}
                n+=m['conditional_native']<m[control]
            counts.append(n)
        wins.append(counts)
    gates={control:all(d<0 for d in c['per_seed_Brier']) and c['mean_Brier']<=-.0005 for control,c in contrasts.items()}
    gates['calibration']=all(np.mean([pooled[str(seed)]['conditional_native']['ECE10']-pooled[str(seed)][control]['ECE10'] for seed in SEEDS])<=.003
                                  for control in ('joint_native','conditional_pool','teacher'))
    gates['fold_consistency']=all(min(w)>=2 for w in wins)
    gates['bootstrap']=all(contrasts[c]['user_cluster_CI95'][1]<0 for c in ('joint_native','conditional_pool','teacher'))
    gates['engineering_candidate']=all(gates.values())
    return {'pooled_per_seed':pooled,'metrics_per_cell':metrics,'contrasts':contrasts,'fold_wins_joint_and_pool':wins,
        'gates':gates,'users':len(users),'rows_per_seed':len(y),'bootstrap_seed':32032,'bootstrap_draws':1000,
        'independent_qualification':False,'conditional_mutual_information_estimated':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/me032');a=p.parse_args();run(Path(a.out))
