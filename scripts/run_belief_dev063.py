"""DEV063 registered public-posterior learning; teacher NEVER exported."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import copy
import json
import multiprocessing
import resource
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from torch.nn import functional as F
from tac_osm.development.belief_bottleneck import BeliefBottleneck as BeliefAgent,configurations
from data_belief_dev063 import BeliefWorld,snapshot,corpus,batch,relation
from run_learned_stack_dev055 import write,digest
PROTOCOL=ROOT/'research/dev063-belief-integrity/protocol.json'


def gate_fit():
    torch.manual_seed(6340900);m=BeliefAgent();opt=torch.optim.Adam(m.gate.parameters(),lr=.01)
    for _ in range(300):
        x=torch.eye(6);loss=F.binary_cross_entropy_with_logits(m.gate(x).squeeze(-1),torch.tensor([1.,1.,1.,1.,1.,0.]));opt.zero_grad();loss.backward();opt.step()
    assert (m.gate(torch.eye(6)).squeeze(-1)>=0).tolist()==[True,True,True,True,True,False]
    return m.gate.state_dict()


def make(arm,spec,seed,gate):
    m=BeliefAgent(spec['backend'],spec['width'],spec['aux'],seed,clock_mode=spec['clock_mode'],bypass=spec['bypass']);m.gate.load_state_dict(gate)
    for p in m.gate.parameters():p.requires_grad_(False)
    return m


def objective(m,x,warm=False):
    o=m(x['events'],x['present'],x['keys'],x['candidates'],x['prices']);kl=F.kl_div(o['posterior'].clamp_min(1e-12).log(),x['posterior'],reduction='batchmean');policy=-(x['policy']*o['logits'].log_softmax(-1)).sum(-1).mean();future=F.kl_div(o['future'].log_softmax(-1),x['future'],reduction='batchmean')/5;cost=F.smooth_l1_loss(o['cost'],x['cost']/40)
    loss=4*kl if warm else 4*kl+policy+.25*future+.1*cost
    return loss,{k:float(v.detach()) for k,v in dict(loss=loss,kl=kl,policy=policy,future=future,cost=cost).items()}


@torch.no_grad()
def rollout(m,seed,history,count,horizon=10,collect=False,error=.1,inspection=.1):
    worlds=[BeliefWorld(seed,i,history,error,inspection) for i in range(count)];beliefs=[w.reference() for w in worlds];dtype=next(m.parameters()).dtype;keys=torch.tensor(np.stack([w.identities for w in worlds]),dtype=dtype);state=m.initial(keys);observed=[]
    for i,w in enumerate(worlds):
        e=torch.tensor(np.stack(w.events),dtype=dtype);take=m.gate(e[:,:6]).squeeze(-1)>=0;retained=e[take][-64:];state['events'][i,:len(retained)]=retained;state['present'][i,:len(retained)]=True
    telemetry=[{'posterior':[],'public_posterior':[],'truth':[],'stop_probability':[],'top2_margin':[]} for _ in worlds]
    for _ in range(horizon):
        if all(w.stopped for w in worlds):break
        payload=[w.candidates() for w in worlds];d=torch.tensor(np.stack([v[0] for v in payload]),dtype=dtype);prices=keys.new_tensor([[w.error,w.inspection] for w in worlds]);out=m.decide(state,d,prices);choices=m.choice(out);pending=[]
        for i,w in enumerate(worlds):
            if w.stopped:pending.append([]);continue
            if collect:observed.append(snapshot(w,beliefs[i]))
            from tac_osm.development.uncertain_world import STATES
            public=np.zeros((4,3));public[:,relation(w)]=beliefs[i].p
            t=telemetry[i];t['posterior'].append(out['posterior'][i].tolist());t['public_posterior'].append(public.flatten().tolist());t['truth'].append(int(np.flatnonzero((STATES==w.bits).all(-1))[0])*3+relation(w));stop=int(np.flatnonzero(payload[i][1]==0)[0]);t['stop_probability'].append(float((out['logits'][i]/m.temperature).softmax(-1)[stop]));top=out['logits'][i].topk(2).values;t['top2_margin'].append(float(top[0]-top[1]));action=int(payload[i][1][int(choices[i])]);start=len(w.events);receipt=w.act(action);beliefs[i].update(action,receipt);pending.append(w.events[start:])
        for tick in range(max(map(len,pending),default=0)):
            present=torch.tensor([tick<len(e) for e in pending]);e=keys.new_tensor(np.stack([v[tick] if tick<len(v) else np.zeros(46) for v in pending]));state=m.observe(state,e,present)
    return {'seed':seed,'history':history,'worlds':count,'horizon':horizon,'error':error,'inspection':inspection,'rows':[dict(w.finish(),**t) for w,t in zip(worlds,telemetry,strict=True)]},observed


@torch.no_grad()
def fixed_metrics(m,records,gate):
    kl=[];brier=[];empirical=[];public_empirical=[];completion_p=[];completion_y=[];tv=[];nll={t:[] for t in (.5,1.,2.,4.)};dtype=next(m.parameters()).dtype
    for start in range(0,len(records),64):
        x=batch(records,range(start,min(start+64,len(records))),gate);x={k:v.to(dtype) if v.is_floating_point() else v for k,v in x.items()};o=m(x['events'],x['present'],x['keys'],x['candidates'],x['prices']);p=o['posterior'];q=x['posterior'];kl.extend(F.kl_div(p.clamp_min(1e-12).log(),q,reduction='none').sum(-1).tolist());tv.extend(((p-q).abs().sum(-1)/2).tolist());brier.extend(((p-q)**2).sum(-1).tolist());truth=torch.tensor([records[i]['truth'] for i in range(start,min(start+64,len(records)))],device=p.device);onehot=F.one_hot(truth,12).to(p.dtype);empirical.extend(((p-onehot)**2).sum(-1).tolist());public_empirical.extend(((q-onehot)**2).sum(-1).tolist());completion_p.extend(p[:,:3].sum(-1).tolist());completion_y.extend((truth//3==0).int().tolist())
        for t in nll:nll[t].extend((-(x['policy']*(o['logits']/t).log_softmax(-1)).sum(-1)).tolist())
    from evaluate_uncertainty_dev060 import calibration
    return {'empirical_multiclass_Brier':float(np.mean(empirical)),'public_empirical_multiclass_Brier':float(np.mean(public_empirical)),'completion_calibration':calibration(completion_p,completion_y),'posterior_KL':float(np.mean(kl)),'per_query_KL':kl,'p95_KL':float(np.quantile(kl,.95)),'p99_KL':float(np.quantile(kl,.99)),'posterior_TV':float(np.mean(tv)),'expected_Brier_excess':float(np.mean(brier)),'policy_temperature_NLL':{t:float(np.mean(v)) for t,v in nll.items()}}


def group(job):
    seed,lrs,specs,folder,dataset,smoke=job;torch.set_num_threads(1);torch.use_deterministic_algorithms(True);folder=Path(folder);folder.mkdir(parents=True,exist_ok=False);start=time.process_time();p=json.loads(PROTOCOL.read_text());train=torch.load(Path(dataset)/'train.pt',weights_only=True);val=torch.load(Path(dataset)/'validation.pt',weights_only=True);gate=gate_fit();models={a:make(a,s,seed,gate) for a,s in specs.items()};ema={a:copy.deepcopy(m).eval() for a,m in models.items()};optim={a:torch.optim.AdamW([v for v in m.parameters() if v.requires_grad],lr=lrs[a],weight_decay=1e-4) for a,m in models.items()};curves={a:[] for a in models};pool=[];collection=[];rng=torch.Generator().manual_seed(seed);updates=8 if smoke else p['updates'];rounds=(3,5,7) if smoke else (1280,1536,1792);warm=2 if smoke else 1024;status='RUNNING'
    try:
        for step in range(1,updates+1):
            records=train+pool
            if pool:ids=torch.randint(len(train),(32,),generator=rng).tolist()+[len(train)+int(i) for i in torch.randint(len(pool),(32,),generator=rng)]
            else:ids=torch.randint(len(train),(min(64,len(train)),),generator=rng).tolist()
            x=batch(records,ids,models['tac_rel_belief'].gate)
            for a,m in models.items():
                loss,metrics=objective(m,x,step<=warm);assert torch.isfinite(loss);optim[a].zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);optim[a].step()
                with torch.no_grad():
                    for e,v in zip(ema[a].parameters(),m.parameters(),strict=True):e.mul_(.995).add_(v,alpha=.005)
                if step in (1,warm,updates) or step%128==0:
                    with torch.no_grad():_,vm=objective(ema[a],batch(val,range(min(64,len(val))),m.gate))
                    curves[a].append({'update':step,'train':metrics,'validation':vm});write(folder/f'{a}-progress.json',curves[a])
            if step in rounds:
                stage=rounds.index(step)
                for a,m in ema.items():
                    c,visited=rollout(copy.deepcopy(m).double(),6342000+seed*4+stage,32,2 if smoke else 64,(2,5,10)[stage],True);pool+=visited;collection.append({'arm':a,'round':stage,'cell':c})
            if time.process_time()-start>p['group_cpu_cap_seconds']:status='CAPPED';break
        else:status='COMPLETE'
    except Exception:
        import traceback
        status='FAILED';write(folder/'failure.json',{'error':traceback.format_exc(),'step':step})
    torch.save(pool,folder/'dagger.pt');write(folder/'collection.json',collection);reports=[]
    for a,m in models.items():
        dest=folder/a;dest.mkdir();torch.save(m.state_dict(),dest/'raw.pt');deploy=copy.deepcopy(ema[a]).double().eval();fm=fixed_metrics(deploy,val,models['tac_rel_belief'].gate);temperature=min(fm['policy_temperature_NLL'],key=lambda t:(fm['policy_temperature_NLL'][t],t));deploy.temperature.fill_(temperature);torch.save(deploy.state_dict(),dest/'agent.pt');cells=[rollout(deploy,6341002+10000000*int(smoke),32,2 if smoke else 32)[0]];report={'arm':a,'seed':seed,'spec':specs[a],'lr':lrs[a],'status':status,'updates':step,'curves':curves[a],'validation_fixed':fm,'validation':cells,'checkpoint_sha256':digest(dest/'agent.pt'),'source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()};write(dest/'report.json',report);reports.append(report)
    if time.process_time()-start>p['group_cpu_cap_seconds'] and status=='COMPLETE':status='CAPPED'
    summary={'seed':seed,'status':'SMOKE_ONLY' if smoke else status,'cpu_seconds':time.process_time()-start,'reports':reports,'dagger_snapshots':len(pool),'collection_cost':sum(r['cost'] for c in collection for r in c['cell']['rows'])};write(folder/'summary.json',summary);return summary


def run(directory,smoke=False):
    torch.set_num_threads(1);start=time.process_time();child_start=resource.getrusage(resource.RUSAGE_CHILDREN);root=Path(directory);root.mkdir(parents=True,exist_ok=False);p=json.loads(PROTOCOL.read_text());specs=configurations();write(root/'specs.json',specs);ds=root/'_datasets';ds.mkdir();offset=10000000*int(smoke)
    if not smoke and subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():raise RuntimeError('Commit registered source before fitting')
    for name,seed,count in [('train',6341001,p['snapshots']),('validation',6341002,p['validation_snapshots'])]:torch.save(corpus(seed+offset,8 if smoke else count),ds/f'{name}.pt')
    preparation=time.process_time()-start;groups=[];fits=[];pilots=[];seeds=p['seeds'][:1] if smoke else p['seeds']
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as executor:
        jobs=[(93400+offset,{a:lr for a in specs},specs,str(root/f'pilot-{i}'),str(ds),smoke) for i,lr in enumerate(p['lrs'][:1] if smoke else p['lrs'])]
        for f in as_completed([executor.submit(group,j) for j in jobs]):
            g=f.result();pilots.append(g);groups.append(g);write(root/'pilots.json',pilots);print('pilot',g['status'],round(g['cpu_seconds'],1),flush=True)
        complete_pilots=smoke or all(g['status']=='COMPLETE' for g in pilots)
        if complete_pilots:
            def rank(r):return r['validation_fixed']['posterior_KL'],r['validation_fixed']['policy_temperature_NLL'][1.],r['lr']
            lrs={a:min([r for g in pilots for r in g['reports'] if r['arm']==a],key=rank)['lr'] for a in specs};write(root/'selection.json',lrs)
            jobs=[(seed+offset,lrs,specs,str(root/f'group-{seed+offset}'),str(ds),smoke) for seed in seeds]
            for i in range(0,len(jobs),2):
                if preparation+sum(g['cpu_seconds'] for g in groups)+len(jobs[i:i+2])*p['group_cpu_cap_seconds']>p['fit_cpu_cap_seconds']:break
                for f in as_completed([executor.submit(group,j) for j in jobs[i:i+2]]):
                    g=f.result();fits.append(g);groups.append(g);write(root/'groups.json',fits);print('group',g['seed'],g['status'],round(g['cpu_seconds'],1),flush=True)
    child_end=resource.getrusage(resource.RUSAGE_CHILDREN);fit_cpu=time.process_time()-start+child_end.ru_utime+child_end.ru_stime-child_start.ru_utime-child_start.ru_stime
    complete=fit_cpu<=p['fit_cpu_cap_seconds'] and len(fits)==len(seeds) and all(g['status']==('SMOKE_ONLY' if smoke else 'COMPLETE') for g in fits) and preparation+sum(g['cpu_seconds'] for g in groups)<=p['fit_cpu_cap_seconds'];summary={'status':'SMOKE_ONLY' if smoke else 'FITS_COMPLETE' if complete else 'INCOMPLETE','test_opened':False,'fits':len(fits)*len(specs),'pilot_fits':len(pilots)*len(specs),'fitting_cpu_seconds':fit_cpu,'protocol_sha256':digest(PROTOCOL)};write(root/'summary.json',summary)
    if not smoke and complete:
        fixed=corpus(6341005,256);torch.save(fixed,root/'test-fixed.pt')
        for g in sorted(fits,key=lambda g:g['seed']):
            for r in g['reports']:
                folder=root/f"group-{r['seed']}"/r['arm'];gate=gate_fit();m=make(r['arm'],r['spec'],r['seed'],gate).double().eval();m.load_state_dict(torch.load(folder/'agent.pt',weights_only=True));floatgate=copy.deepcopy(m.gate).float();fm=fixed_metrics(m,fixed,floatgate);cells=[rollout(m,6341003,h,64)[0] for h in (32,128,512,1024)];cells+=[rollout(m,6341004,h,64,error=e,inspection=price)[0] for h,e,price in ((32,.2,.1),(32,.1,.4))];write(folder/'test.json',{'fixed':fm,'cells':cells});print('tested',r['arm'],r['seed'],flush=True)
        summary.update(status='COMPLETE',test_opened=True);write(root/'summary.json',summary)
    write(root/'manifest.json',{str(f.relative_to(root)):digest(f) for f in sorted(root.rglob('*')) if f.is_file() and f.name!='manifest.json'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();run(a.directory,a.smoke)
