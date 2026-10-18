"""Registered twenty-seed DEV061.5 fitting/evaluation; no teacher in export."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import copy
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from torch.nn import functional as F
from tac_osm.development.robust_identity import IdentityAgent,configurations,choose
from data_identity_dev0615 import IdentityWorld,snapshot,corpus,canonical,batch,BOOK
from run_learned_stack_dev055 import write,digest
PROTOCOL=ROOT/'research/dev0615/protocol.json'


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()


def serialize(rows):
    return [{k:torch.from_numpy(v) if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else v for k,v in r.items()} for r in rows]


def setup_gate():
    torch.manual_seed(6150901);m=IdentityAgent();optimizer=torch.optim.Adam(list(m.admission.parameters())+list(m.relation_gate.parameters()),lr=.01)
    for _ in range(300):
        kind=torch.randint(4,(256,));x=F.one_hot(kind,4).float()
        loss=F.binary_cross_entropy_with_logits(m.admission(x).squeeze(-1),(kind!=3).float())+F.binary_cross_entropy_with_logits(m.relation_gate(x).squeeze(-1),(kind==1).float())
        optimizer.zero_grad();loss.backward();optimizer.step()
    x=torch.eye(4);assert torch.equal(m.admission(x).squeeze(-1)>=0,torch.tensor([True,True,True,False]));assert torch.equal(m.relation_gate(x).squeeze(-1)>=0,torch.tensor([False,True,False,False]))
    return {'admission':m.admission.state_dict(),'relation_gate':m.relation_gate.state_dict(),'updates':300,'loss':float(loss.detach())}


def make(arm,spec,seed,gate=None):
    m=IdentityAgent(arm,spec['width'],spec['aux_width'],seed)
    if gate:
        m.admission.load_state_dict(gate['admission']);m.relation_gate.load_state_dict(gate['relation_gate'])
    for module in (m.admission,m.relation_gate):
        for p in module.parameters():p.requires_grad_(False)
    return m


def objective(model,data,warm=False):
    o=model(data['events'],data['present'],data['keys'],data['query_valid'],data['candidates'],data['valid']);q=data['query_valid'];valid=data['valid'];pair=valid[:,:,None]&q[:,None]
    current=F.binary_cross_entropy_with_logits(o['current'][q],data['current'][q]);complete=F.binary_cross_entropy_with_logits(o['completion'],data['completion'])
    reads=F.nll_loss(o['reads'].clamp_min(1e-12).log()[q],data['parent_labels'][q])
    policy=-(data['policy']*o['logits'].log_softmax(-1)).sum(-1).mean()
    one=F.binary_cross_entropy_with_logits(o['outcomes'][pair],data['one'][pair]);done=F.binary_cross_entropy_with_logits(o['done'][valid],data['done'][valid]);cost=F.smooth_l1_loss(o['cost'][valid],data['cost'][valid]/20)
    safe=data['policy']>0;positive=o['logits'].masked_fill(~safe,1e9).min(-1).values;negative=o['logits'].masked_fill(safe|~valid,-1e9).max(-1).values;margin=F.relu(.5+negative-positive).mean()
    loss=.5*current+.5*complete+.25*reads
    if not warm:loss=loss+policy+.5*one+.25*done+.1*cost+.05*margin
    return loss,{k:float(v.detach()) for k,v in dict(loss=loss,policy=policy,current=current,completion=complete,read=reads,one=one,cost=cost,margin=margin).items()}


@torch.no_grad()
def rollout(model,seed,n,h,count,family='seen',horizon=None,collect=False,start_index=0):
    dtype=next(model.parameters()).dtype;worlds=[IdentityWorld(seed,i,n,h,family) for i in range(start_index,start_index+count)];payload=[w.public_candidates() for w in worlds]
    keys=torch.tensor(np.stack([p[2] for p in payload]),dtype=dtype);qv=torch.tensor(np.stack([p[3] for p in payload]));state=model.initial(keys,qv)
    observed=[];writes=np.zeros(count,int)
    for i,w in enumerate(worlds):
        e=torch.tensor(w.public(),dtype=dtype);admit=model.admission(e[:,:4]).squeeze(-1)>=0;retained=e[admit][-64:];state['events'][i,:len(retained)]=retained;state['present'][i,:len(retained)]=True;writes[i]=int(admit.sum())
    stats=[{'top2_gaps':[],'epsilon_boundary_distance':[],'completion_probability':[],'stop_probability':[],'healthy':[]} for _ in worlds]
    for _ in range(horizon or 3*n+6):
        if all(w.stopped for w in worlds):break
        payload=[w.public_candidates() for w in worlds];d=torch.tensor(np.stack([p[0] for p in payload]),dtype=dtype);v=torch.tensor(np.stack([p[1] for p in payload]));out=model.decide(state,d,v)
        choice=choose(out['logits'],v).tolist();probs=(out['logits']/model.temperature).softmax(-1);receipts=[]
        top=out['logits'].topk(2,-1).values
        for i,w in enumerate(worlds):
            if w.stopped:receipts.append([]);continue
            if collect:observed.append(snapshot(w))
            s=stats[i];s['top2_gaps'].append(float(top[i,0]-top[i,1]));distance=(out['logits'][i].max()-out['logits'][i]-1e-6).abs().masked_fill(~v[i],1e9).min();s['epsilon_boundary_distance'].append(float(distance))
            s['completion_probability'].append(float(out['completion'][i].sigmoid()));s['healthy'].append(not bool(w.bits.any()));stop=next(j for j,c in enumerate(payload[i][4]) if c[0]=='stop');s['stop_probability'].append(float(probs[i,stop]))
            receipts.append(w.public(w.act(payload[i][4][choice[i]])))
        for tick in range(max(map(len,receipts),default=0)):
            present=torch.tensor([tick<len(r) for r in receipts]);e=torch.tensor(np.stack([r[tick] if tick<len(r) else np.zeros(38,np.float32) for r in receipts]),dtype=dtype);state,trace=model.observe(state,e,present);writes+=trace['gate'].cpu().numpy().astype(int)
    rows=[]
    for i,w in enumerate(worlds):
        r=w.finish(int(writes[i]));r.pop('initial_events');r.pop('keys');r.pop('parents');r.update(stats[i]);r['family']=family;r['economic_cost']=r['cost']-r['write_cost'];rows.append(r)
    return {'seed':seed,'entities':n,'history':h,'family':family,'horizon':horizon or 3*n+6,'rows':rows},observed


def score(cells):
    rows=[r for c in cells for r in c['rows']]
    return {'completion':float(np.mean([r['success'] for r in rows])),'faults':sum(r['final_faults'] for r in rows),'cost':float(np.mean([r['economic_cost'] for r in rows])),'worlds':len(rows)}


@torch.no_grad()
def validation(model,seed,smoke=False):
    model.eval();cells=[]
    for family in ('seen','unseen','fresh'):
        for n in ((2,) if smoke else (2,4,6)):
            c,_=rollout(model,seed,n,32 if smoke else 64,2 if smoke else 16,family);cells.append(c)
    return cells


@torch.no_grad()
def calibrate(model,records):
    values=[]
    for offset in range(0,len(records),128):
        x=batch(records,list(range(offset,min(offset+128,len(records)))),model,torch.Generator().manual_seed(6150998+offset),not model.fixed)
        x={k:v.double() if v.is_floating_point() else v for k,v in x.items()};o=model(x['events'],x['present'],x['keys'],x['query_valid'],x['candidates'],x['valid']);values.append((o['logits'],x['policy']))
    losses={t:float(torch.cat([-(y*(z/t).log_softmax(-1)).sum(-1) for z,y in values]).mean()) for t in (.5,1.,2.,4.)};temperature=min(losses,key=lambda t:(losses[t],t));model.temperature.fill_(temperature);return {'policy_temperature':temperature,'grid_NLL':losses}


def train_group(job):
    seed,lrs,specs,root,dataset,pilot,smoke=job;torch.set_num_threads(1);torch.use_deterministic_algorithms(True);p=json.loads(PROTOCOL.read_text());folder=Path(root);folder.mkdir(parents=True,exist_ok=False);start=time.process_time()
    train=torch.load(Path(dataset)/'train.pt',weights_only=True);val=torch.load(Path(dataset)/'validation.pt',weights_only=True);gate=torch.load(Path(dataset)/'gate.pt',weights_only=True)
    models={a:make(a,s,seed,gate) for a,s in specs.items()};ema={a:copy.deepcopy(m).eval() for a,m in models.items() if a.endswith('robust')};optim={a:torch.optim.AdamW([v for v in m.parameters() if v.requires_grad],lr=lrs[a],weight_decay=1e-4) for a,m in models.items()};curves={a:[] for a in models};pool=[];collection=[]
    base_rng=torch.Generator().manual_seed(seed+615);updates=8 if smoke else p['updates'];warm=2 if smoke else p['warmup'];rounds=(3,5,7) if smoke else tuple(p['dagger_updates']);status='RUNNING'
    try:
        for step in range(1,updates+1):
            base_ids=torch.randint(len(train),(min(64,len(train)),),generator=base_rng).tolist();robust_records=train if not pool else train+pool
            if pool:
                initial=torch.randint(len(train),(32,),generator=base_rng).tolist();new=[len(train)+int(i) for i in torch.randint(len(pool),(32,),generator=base_rng)];robust_ids=initial+new
            elif step<=rounds[0]:
                shallow=[i for i,r in enumerate(train) if r['depth']<=2];robust_ids=[shallow[int(i)] for i in torch.randint(len(shallow),(min(64,len(shallow)),),generator=base_rng)]
            else:robust_ids=base_ids
            cached={}
            for arm,m in models.items():
                m.train();robust=arm.endswith('robust');records=robust_records if robust else train;ids=robust_ids if robust else base_ids;key=(robust,m.fixed)
                if key not in cached:cached[key]=batch(records,ids,m,torch.Generator().manual_seed(seed*10000+step),not m.fixed)
                loss,metrics=objective(m,cached[key],step<=warm);assert torch.isfinite(loss),'nonfinite fitting loss';optim[arm].zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);optim[arm].step()
                if robust:
                    with torch.no_grad():
                        for e,v in zip(ema[arm].parameters(),m.parameters(),strict=True):e.mul_(p['ema']).add_(v,alpha=1-p['ema'])
                if step in (1,warm,updates) or step%128==0:
                    saved=copy.deepcopy(ema.get(arm,m)).eval();vd=batch(val,list(range(min(128,len(val)))),saved,torch.Generator().manual_seed(6150997),not saved.fixed)
                    with torch.no_grad():_,vm=objective(saved,vd,False)
                    curves[arm].append({'update':step,'training':metrics,'validation':vm,'cpu_seconds_group':time.process_time()-start});write(folder/f'{arm}-progress.json',curves[arm])
            if step in rounds:
                stage=rounds.index(step);new=[];models_for_collection={a:copy.deepcopy(v).double().eval() for a,v in ema.items()}
                for arm,m in models_for_collection.items():
                    for block in range(1 if smoke else 4):
                        n=(2,4,6,4)[block];h=(16,32,64,32)[block];limit=(2,6,3*n+6)[stage]
                        cell,visited=rollout(m,6152000+seed*4+stage,n,h,2 if smoke else 32,'fresh',limit,True,start_index=block*32);new+=visited
                        collection.append({'arm':arm,'round':stage,'cell':cell,'visited':len(visited),'cost':sum(r['economic_cost']-20*r['final_faults']*(not r['stopped']) for r in cell['rows'])})
                pool+=canonical(new,next(iter(models.values())).admission)
                write(folder/'collection-progress.json',{'round':stage,'snapshots':len(pool),'cells':len(collection)})
            if time.process_time()-start>p['group_cpu_cap_seconds']:status='CAPPED';break
        else:status='COMPLETE'
    except Exception:
        import traceback
        status='FAILED';write(folder/'failure.json',{'error':traceback.format_exc(),'step':step})
    torch.save(serialize(pool),folder/'dagger.pt');write(folder/'collection.json',collection)
    reports=[]
    for arm,m in models.items():
        dest=folder/arm;dest.mkdir();torch.save(m.state_dict(),dest/'raw.pt');selected=copy.deepcopy(ema.get(arm,m)).double().eval();calibration=calibrate(selected,val[:8] if smoke else val);torch.save(selected.state_dict(),dest/'agent.pt')
        cells=validation(selected,6151002+10000000*int(smoke),smoke);report={'arm':arm,'seed':seed,'lr':lrs[arm],'spec':specs[arm],'source':git('rev-parse','HEAD'),'status':status,'updates':step,'curves':curves[arm],'validation':cells,'calibration':calibration,'checkpoint_sha256':digest(dest/'agent.pt'),'selected':'EMA' if arm in ema else 'RAW','dtype':'float64'};write(dest/'report.json',report);reports.append(report)
    if time.process_time()-start>p['group_cpu_cap_seconds'] and status=='COMPLETE':status='CAPPED'
    summary={'seed':seed,'status':'SMOKE_ONLY' if smoke else status,'updates':step,'pilot':pilot,'cpu_seconds':time.process_time()-start,'snapshots_aggregated':len(pool),'collection_cost':sum(r['cost'] for r in collection),'reports':reports};write(folder/'summary.json',summary);return summary


def rank(r):
    cells=r['validation'];m=score(cells);families=[score([c for c in cells if c['family']==f])['completion'] for f in ('seen','unseen','fresh')]
    return m['faults'],-min(families),m['cost'],r['curves'][-1]['validation']['loss'],r['lr']


def run(directory,smoke=False):
    torch.set_num_threads(1);preparation_start=time.process_time();p=json.loads(PROTOCOL.read_text());root=Path(directory);root.mkdir(parents=True,exist_ok=False)
    if not smoke and git('status','--porcelain','--untracked-files=no'):raise RuntimeError('Commit all registered source before fitting')
    specs=configurations();write(root/'configurations.json',specs);ds=root/'_datasets';ds.mkdir();gate=setup_gate();torch.save(gate,ds/'gate.pt');offset=10000000 if smoke else 0
    for name,seed,count in [('train',p['train_seed'],24 if smoke else p['snapshots']),('validation',p['validation_seed'],8 if smoke else p['validation_snapshots'])]:
        rows=canonical(corpus(seed+offset,count),make('split_robust',specs['split_robust'],0,gate).admission);torch.save(serialize(rows),ds/f'{name}.pt')
    write(root/'data-sha256.json',{name:digest(ds/f'{name}.pt') for name in ('train','validation','gate')});write(root/'codebook.json',BOOK.tolist())
    preparation_cpu=time.process_time()-preparation_start
    write(root/'preparation.json',{'cpu_seconds':preparation_cpu})
    all_groups=[];fits=[];pilots=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as executor:
        jobs=[(p['pilot_seed']+offset,{a:lr for a in specs},specs,str(root/f'pilot-{i}'),str(ds),True,smoke) for i,lr in enumerate(p['lrs'][:1] if smoke else p['lrs'])]
        for future in as_completed([executor.submit(train_group,j) for j in jobs]):
            g=future.result();pilots.append(g);all_groups.append(g);write(root/'pilots.json',pilots);print('pilot',g['seed'],g['status'],round(g['cpu_seconds'],1),flush=True)
        if not smoke and any(g['status']!='COMPLETE' for g in pilots):
            write(root/'summary.json',{'status':'INCOMPLETE','stage':'PILOT','test_opened':False,'source':git('rev-parse','HEAD'),'fitting_cpu_seconds':sum(g['cpu_seconds'] for g in pilots)})
            return
        lrs={a:min([r for g in pilots for r in g['reports'] if r['arm']==a],key=rank)['lr'] for a in specs};write(root/'selection.json',lrs)
        seeds=p['seeds'][:1] if smoke else p['seeds'];jobs=[(s+offset,lrs,specs,str(root/f'group-{s+offset}'),str(ds),False,smoke) for s in seeds]
        for index in range(0,len(jobs),2):
            if preparation_cpu+sum(g['cpu_seconds'] for g in all_groups)+len(jobs[index:index+2])*p['group_cpu_cap_seconds']>p['fit_cpu_cap_seconds']:break
            for future in as_completed([executor.submit(train_group,j) for j in jobs[index:index+2]]):
                g=future.result();fits.append(g);all_groups.append(g);write(root/'groups.json',fits);print('group',g['seed'],g['status'],round(g['cpu_seconds'],1),'totalCPUmin',round(sum(v['cpu_seconds'] for v in all_groups)/60,1),flush=True)
    complete=len(fits)==len(seeds) and all(g['status'] in (('SMOKE_ONLY',) if smoke else ('COMPLETE',)) for g in fits) and preparation_cpu+sum(g['cpu_seconds'] for g in all_groups)<=p['fit_cpu_cap_seconds']
    summary={'status':'SMOKE_ONLY' if smoke else 'FITS_COMPLETE' if complete else 'INCOMPLETE','source':git('rev-parse','HEAD'),'protocol_sha256':digest(PROTOCOL),'fits':len(fits)*7,'pilot_fits':len(pilots)*7,'fitting_cpu_seconds':preparation_cpu+sum(g['cpu_seconds'] for g in all_groups),'test_opened':False,'elapsed_seconds':time.monotonic()-start};write(root/'summary.json',summary)
    if not smoke and complete:
        for group in sorted(fits,key=lambda g:g['seed']):
            for r in group['reports']:
                m=make(r['arm'],r['spec'],r['seed']).double().eval();folder=root/f"group-{r['seed']}"/r['arm'];m.load_state_dict(torch.load(folder/'agent.pt',weights_only=True));cells=[]
                for family in ('seen','unseen','fresh'):
                    for n in (2,4,6):
                        for h in (32,128):cells.append(rollout(m,p['test_seed'],n,h,32,family)[0])
                for family in ('seen','unseen','fresh'):
                    for h in (512,1024):cells.append(rollout(m,p['secondary_seed'],6,h,16,family)[0])
                for family in ('gaussian','collision'):cells.append(rollout(m,p['stress_seed'],6,128,16,family)[0])
                cells.append(rollout(m,p['test_seed'],6,128,32,'fresh',2*6+4)[0])
                write(folder/'test.json',{'cells':cells});print('tested',r['arm'],r['seed'],score(cells[:18]),flush=True)
        summary['status']='COMPLETE';summary['test_opened']=True;write(root/'summary.json',summary)
    write(root/'manifest.json',{str(f.relative_to(root)):digest(f) for f in sorted(root.rglob('*')) if f.is_file() and f.name!='manifest.json'})


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--directory',required=True);a.add_argument('--smoke',action='store_true');v=a.parse_args();run(v.directory,v.smoke)
