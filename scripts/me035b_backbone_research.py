"""Bounded second architecture iteration: strong AR backbone + native memory."""
import argparse,json,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from tac_osm.me035_ar_backbone import initialize
from scripts.me035_architecture_research import ROOT,source,SEEDS,metrics,predict,file_sha,write_json,atomic_checkpoint
from scripts.me035_audit import verify

PROTOCOL=ROOT/'research/mathematical-engineering/me035b/PROTOCOL.md'
STEPS=(0,25,50,100,250)


def fit_one(seed,parent,coef,train,out):
    start=time.process_time();net=torch.jit.script(initialize(parent,coef,seed))
    groups=[[],[],[]]
    for n,p in net.named_parameters():
        j=2 if n.startswith('direct.') else 1 if n.startswith(('mix.','output.')) else 0
        groups[j].append(p)
    opt=torch.optim.AdamW([{'params':groups[0],'lr':.0002},{'params':groups[1],'lr':.0005},{'params':groups[2],'lr':.00001}],weight_decay=1e-4)
    anchor=net.direct.weight.detach().clone();gen=torch.Generator().manual_seed(seed+350)
    trace=[];checks=[];step=0;charged=time.process_time()-start
    def save():
        name=f'ar_native-s{seed}-u{step}'
        atomic_checkpoint(out/(name+'.pt'),{'state_dict':net.state_dict(),'optimizer':opt.state_dict(),
            'sampler_rng':gen.get_state(),'model_rng':torch.get_rng_state(),'step':step,'seed':seed})
        net.save(str(out/(name+'.jit.pt')));write_json(out/(name+'-trace.json'),trace)
        checks.append({'step':step,'name':name,'charged_CPU_seconds':charged})
    save()
    for step in range(1,251):
        t=time.process_time();idx=torch.randint(len(train['x']),(64,),generator=gen)
        opt.zero_grad(set_to_none=True);p=net(train['x'][idx]);mse=F.mse_loss(p,train['y'][idx])
        loss=mse+25.*(net.direct.weight-anchor).square().mean()
        if not torch.isfinite(loss):raise ArithmeticError('nonfinite real objective')
        loss.backward();g=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        if not torch.isfinite(g):raise ArithmeticError('nonfinite gradient')
        mg=sum(float(p.grad.square().sum()) for n,p in net.named_parameters() if not n.startswith(('direct.','mix.','output.')) and p.grad is not None)**.5
        opt.step();dt=time.process_time()-t;charged+=dt
        trace.append({'step':step,'MSE':float(mse.detach()),'objective':float(loss.detach()),'CPU_seconds':dt,'memory_gradient_after_clip':mg})
        if step in STEPS:save()
        if charged>150:
            save();raise RuntimeError('registered CPU cap exceeded')
    return {'seed':seed,'steps':250,'new_histories':16000,'active_parameters':32154,
        'charged_CPU_seconds':charged,'checkpoints':checks,'positive_memory_gradient_steps':sum(t['memory_gradient_after_clip']>0 for t in trace)}


def evaluate(out,dev,record):
    rows=[]
    for fit in record['fits']:
        for ck in fit['checkpoints']:
            p=predict(torch.jit.load(str(out/(ck['name']+'.jit.pt'))),dev['x'])
            np.save(out/(ck['name']+'-predictions.npy'),p)
            rows.append({'seed':fit['seed'],'step':ck['step'],'metrics':metrics(p,dev['y'].numpy())})
    means={str(s):{k:float(np.mean([v['metrics'][k] for v in rows if v['step']==s])) for k in ('MSE','MAE')} for s in STEPS}
    step=min(STEPS[1:],key=lambda s:(means[str(s)]['MSE'],s))
    old=record['parent_MSE_by_seed']
    wins=[next(v['metrics']['MSE'] for v in rows if v['seed']==s and v['step']==step)<old[str(s)] for s in SEEDS]
    gap=means[str(step)]['MSE']/record['ridge_MSE']-1
    return {'per_model':rows,'means':means,'selected_step':step,'selected_MSE':means[str(step)]['MSE'],
        'relative_MSE_vs_ridge':gap,'parent_improved_per_seed':wins,'development_competitiveness_gate':gap<=.01 and all(wins),
        'independent_qualification':False,'selection_uses_development':True}


def run(out):
    out.mkdir(exist_ok=False,parents=True);torch.set_num_threads(1);start=time.process_time();wall=time.perf_counter()
    parent_root=ROOT/'results/me035';parent=verify(parent_root);train,dev,src=source(parent_root)
    chosen=parent['evaluation']['selected_development_candidate'];assert chosen['arm']=='native_multiscale' and chosen['step']==250
    coef=np.load(parent_root/'ridge-coefficients.npy')
    r={'study':'ME035B_TRAINABLE_AR_BACKBONE_NATIVE_REFINEMENT','complete':False,'fits':[],
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'protocol_sha256':file_sha(PROTOCOL),'parent_results_sha256':file_sha(parent_root/'results.json'),
        'ridge_coefficients_sha256':file_sha(parent_root/'ridge-coefficients.npy'),
        'ridge_MSE':parent['evaluation']['references']['ridge']['MSE'],'parent_MSE_by_seed':{},
        'source':src,'all_models_frozen_before_scoring':False,'architecture_superiority_claim':False,'candidate_promoted':False}
    write_json(out/'results.json',r)
    for seed in SEEDS:
        old=next(f for f in parent['fits'] if f['seed']==seed and f['arm']==chosen['arm'])
        ck=next(c for c in old['checkpoints'] if c['step']==chosen['step'])
        weights=torch.load(parent_root/(ck['name']+'.pt'),weights_only=True)
        r['parent_MSE_by_seed'][str(seed)]=next(m['metrics']['MSE'] for m in parent['evaluation']['per_model'] if m['seed']==seed and m['arm']==chosen['arm'] and m['step']==250)
        fit=fit_one(seed,weights['state_dict'],coef,train,out)
        fit.update(parent_checkpoint=ck['name']+'.pt',parent_sha256=file_sha(parent_root/(ck['name']+'.pt')),
            charged_parent_CPU_seconds=ck['charged_CPU_seconds'],charged_ridge_CPU_seconds=parent['ridge']['charged_cpu_seconds'])
        r['fits'].append(fit);write_json(out/'results.json',r)
        print(json.dumps({'completed_fit':seed,'CPU_seconds':fit['charged_CPU_seconds'],'development_scores_opened':False}),flush=True)
    r['all_models_frozen_before_scoring']=True
    write_json(out/'frozen-models.json',{p.name:file_sha(p) for p in out.iterdir() if p.is_file() and p.name!='results.json'})
    r['evaluation']=evaluate(out,dev,r);r['complete']=True;r['new_physical_CPU_seconds']=time.process_time()-start;r['wall_seconds']=time.perf_counter()-wall
    write_json(out/'results.json',r);print(json.dumps(r['evaluation']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('results/me035b'));a=p.parse_args()
    try:run(a.out)
    except Exception:
        if a.out.exists():write_json(a.out/'failure.json',{'error':traceback.format_exc()})
        raise
