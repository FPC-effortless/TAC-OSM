"""Registered DEV061 factorial. Teachers and public structural targets stay here."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from torch.nn import functional as F
from tac_osm.development.learned_agent_world import World,targets,teacher
from tac_osm.development.structured_memory import create,configurations,PublicFeatureMLP
from run_learned_stack_dev055 import write,digest,git
PROTOCOL=ROOT/'research/dev061/protocol.json'


class ResearchWorld(World):
    def __init__(self,seed,index,entities,history,family='clean'):
        super().__init__(seed,index,entities,history);self.family=family
        rng=np.random.default_rng(np.random.SeedSequence([seed,index,entities,history,611]))
        original=self.keys.copy()
        options=np.array([6,7,0,1,2,3])[:entities] if family=='unseen_ids' else rng.choice(6,entities,replace=False)
        rng.shuffle(options);mapping={int(k):int(v) for k,v in zip(original,options)}
        for k in range(8):
            if k not in mapping:mapping[k]=int(rng.integers(8 if family=='unseen_ids' else 6))
        for e in self.events:
            for start in (4,12):
                if e[start:start+8].any():
                    old=int(e[start:start+8].argmax());e[start:start+8]=0;e[start+mapping[old]]=1
        self.keys=np.array(options)
        if family=='noisy_initial':
            for e in self.events:
                if e[0]>0 and rng.random()<.1:e[20]=1-e[20]

    def act(self,command):
        receipt=super().act(command)
        if self.family=='noisy_receipts' and receipt:
            rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.index,self.tick,612]))
            for e in receipt:
                if rng.random()<.1:e[20]=1-e[20]
            self.actions[-1]['receipt']=[e.tolist() for e in receipt]
        return receipt


def raw_data(seed,count):
    records=[]
    for i in range(count):
        n=(2,4,6)[i%3];h=(16,32,64,96)[(i//3)%4];w=ResearchWorld(seed,i,n,h)
        rng=np.random.default_rng(np.random.SeedSequence([seed,i,613]))
        if rng.random()<1/3:
            for _ in range(n+4):
                if not w.bits.any():break
                commands=w.candidates()[4];w.act(commands[int(teacher(w.bits,w.parents,commands).argmax())])
            for _ in range(int(rng.integers(n+3))):w.act(('inspect',int(rng.integers(n))))
        else:
            for _ in range(int(rng.integers(n+1))):
                commands=[c for c in w.candidates()[4] if c[0]!='stop'];w.act(commands[int(rng.integers(len(commands)))])
        desc,valid,q,qv,commands=w.candidates();y=targets(w,commands)
        current=np.zeros(6,np.float32);current[:n]=w.bits
        records.append({'events':np.stack(w.events),'descriptors':desc,'valid':valid,'queries':q,'query_valid':qv,'targets':dict(y,current=current)})
    length=max(len(r['events']) for r in records);events=torch.zeros(count,length,22);present=torch.zeros(count,length,dtype=torch.bool)
    for i,r in enumerate(records):events[i,:len(r['events'])]=torch.from_numpy(r['events']);present[i,:len(r['events'])]=True
    ds={'events':events,'present':present}
    for key in ('descriptors','valid','queries','query_valid'):ds[key]=torch.from_numpy(np.stack([r[key] for r in records]))
    for key in ('policy','future','one','two','done','later_done','cost','completion','current'):ds[key]=torch.from_numpy(np.stack([r['targets'][key] for r in records]))
    return ds


def pretrain_admission(model,seed,steps):
    if isinstance(model,PublicFeatureMLP):return {'steps':0,'reason':'Programmed public-feature control, no admission network'}
    torch.manual_seed(seed+614);params=list(model.admission.parameters())
    if not model.entity:params+=list(model.address.parameters())
    optimizer=torch.optim.Adam(params,lr=.01)
    for _ in range(steps):
        size=256;kind=torch.randint(4,(size,));keys=torch.randint(6,(size,2))
        e=torch.cat((F.one_hot(kind,4),F.one_hot(keys[:,0],8),F.one_hot(keys[:,1],8),torch.randint(2,(size,1)),torch.rand(size,1)),1).float()
        gate=model.admission(e).squeeze(-1);loss=F.binary_cross_entropy_with_logits(gate,(kind!=3).float())+.02*gate.sigmoid().mean()
        if not model.entity:
            count=torch.randint(32,(size,));occ=(torch.arange(32)[None]<count[:,None]).float()
            address=model.address(torch.cat((e,occ),-1));loss+=F.cross_entropy(address,count)
        optimizer.zero_grad();loss.backward();optimizer.step()
    for p in params:p.requires_grad_(False)
    return {'steps':steps,'loss':float(loss.detach()),'gate_accuracy':float(((gate>=0)==(kind!=3)).float().mean()),'frozen':True}


@torch.no_grad()
def prepare(model,ds):
    count=len(ds['events']);out=dict(ds)
    latest=torch.zeros(count,16,22);latest_present=torch.zeros(count,16)
    signal_rows=[];route_rows=[]
    if not isinstance(model,PublicFeatureMLP) and not model.entity:
        bank=torch.zeros(count,32,22);occupied=torch.zeros(count,32)
        for start in range(0,count,256):
            raw=bank[start:start+256];occ=occupied[start:start+256]
            for t in range(ds['events'].shape[1]):
                e=ds['events'][start:start+256,t];gate,address=model.write_logits(e,occ)
                mask=(gate>=0)&ds['present'][start:start+256,t];selected=F.one_hot(address.argmax(-1),32).float()*mask[:,None]
                raw=raw*(1-selected[:,:,None])+selected[:,:,None]*e[:,None];occ=torch.maximum(occ,selected)
            bank[start:start+256]=raw;occupied[start:start+256]=occ
        out.update(bank=bank,occupied=occupied)
    for start in range(0,count,256):
        e=ds['events'][start:start+256]
        mask=(e[:,:,:4].argmax(-1)!=3) if isinstance(model,PublicFeatureMLP) else model.admission(e).squeeze(-1)>=0
        mask=mask&ds['present'][start:start+256]
        for local in range(len(e)):
            signals=e[local,mask[local]];routes=signals[:,4:12].argmax(-1)+8*(signals[:,:4].argmax(-1)==1).long()
            signal_rows.append(signals);route_rows.append(routes)
            for event,slot in zip(signals,routes):latest[start+local,slot]=event;latest_present[start+local,slot]=1
    out.update(latest=latest,latest_present=latest_present)
    if not isinstance(model,PublicFeatureMLP) and model.entity:
        maximum=max(len(r) for r in signal_rows);route_max=max(max(torch.bincount(r,minlength=16)).item() for r in route_rows)
        signals=torch.zeros(count,maximum,22);valid=torch.zeros(count,maximum,dtype=torch.bool);routes=torch.zeros(count,maximum,dtype=torch.long);rank=torch.zeros_like(routes)
        routed=torch.zeros(count,16,route_max,22);counts=torch.zeros(count,16,dtype=torch.long)
        for i,(events,where) in enumerate(zip(signal_rows,route_rows)):
            signals[i,:len(events)]=events;valid[i,:len(events)]=True;routes[i,:len(events)]=where
            for t,(e,slot) in enumerate(zip(events,where)):
                pos=counts[i,slot];rank[i,t]=pos;routed[i,slot,pos]=e;counts[i,slot]+=1
        out.update(signals=signals,signal_present=valid,signal_route=routes,signal_rank=rank,routed=routed,route_counts=counts)
    # Raw scan caches are no longer needed inside minibatches but remain public provenance.
    return out


def read_targets(state,queries):
    memory,occ=state;raw=memory[:,:,-22:];keys=queries.argmax(-1);b,n=keys.shape
    mask=(raw[:,:,0]>0)|(raw[:,:,2]>0)
    same=raw[:,:,4:12].argmax(-1)[:,None,:]==keys[:,:,None]
    eligible=same&mask[:,None]&(occ[:,:32]>0)[:,None]
    scores=raw[:,:,21][:,None].expand(-1,n,-1).masked_fill(~eligible,-1e9)
    indices=scores.argmax(-1);indices=torch.where(eligible.any(-1),indices,torch.full_like(indices,32))
    return indices


def per_example(out,d):
    valid=d['valid'];qv=d['query_valid'];mask=valid[:,:,None]&qv[:,None];n=qv.shape[1]
    policy=-(d['policy']*out['logits'].log_softmax(-1)).sum(-1)
    current=(F.binary_cross_entropy_with_logits(out['current_logits'],d['current'],reduction='none')*qv).sum(-1)/qv.sum(-1)
    done=F.binary_cross_entropy_with_logits(out['completion'],d['completion'],reduction='none')
    if 'future_logits' not in out:return policy+.5*current+.5*done,{'policy':policy,'current':current,'completion':done}
    future=-(d['future']*out['future_logits'].log_softmax(-1)).sum(-1);future=(future*valid).sum(-1)/valid.sum(-1)
    one=(F.binary_cross_entropy_with_logits(out['outcomes'],d['one'],reduction='none')*mask).sum((1,2))/mask.sum((1,2))
    choice=out['future_choice'].detach();later=d['two'].gather(2,choice[:,:,None,None].expand(-1,-1,1,n)).squeeze(2)
    two=(F.binary_cross_entropy_with_logits(out['later_outcomes'],later,reduction='none')*mask).sum((1,2))/mask.sum((1,2))
    nextdone=(F.binary_cross_entropy_with_logits(out['next_completion'],d['done'],reduction='none')*valid).sum(-1)/valid.sum(-1)
    cost=(F.smooth_l1_loss(out['cost'],d['cost']/20,reduction='none')*valid).sum(-1)/valid.sum(-1)
    later_target=d['later_done'].gather(2,choice[:,:,None]).squeeze(2)
    laterdone=(F.binary_cross_entropy_with_logits(out['later_completion'],later_target,reduction='none')*valid).sum(-1)/valid.sum(-1)
    total=policy+.3*future+.7*one+.3*two+.5*(done+nextdone+laterdone)/3+.1*cost+.5*current
    return total,{'policy':policy,'current':current,'completion':done,'one':one,'two':two}


def loss(model,ds,ids,warmup=False,counterfactual=False):
    d={k:v[ids] for k,v in ds.items() if k not in ('events','present')};state=model.training_state(d)
    out=model(state,d['descriptors'],d['valid'],d['queries'],d['query_valid']);example,terms=per_example(out,d)
    read_loss=torch.zeros(len(ids))
    if not isinstance(model,PublicFeatureMLP):
        labels=read_targets(state,d['queries']);r=F.cross_entropy(out['state_read_logits'][:,:,0,:].transpose(1,2),labels,reduction='none')
        read_loss=(r*d['query_valid']).sum(-1)/d['query_valid'].sum(-1)
    total=terms['current']+.5*read_loss if warmup else example+.5*read_loss
    retention=torch.tensor(0.);importance_mean=0.
    if counterfactual and getattr(model,'retention',False):
        memory,occupied=state;b=len(ids);available=occupied[:,16:32]>0
        random=torch.rand(b,16).masked_fill(~available,-1);slot=16+random.argmax(-1);has=available.any(-1)
        remove=F.one_hot(slot,32).bool();removed=(memory.masked_fill(remove[:,:,None],0),torch.cat((occupied[:,:32].masked_fill(remove,0),occupied[:,32:]),-1))
        with torch.no_grad():
            perturbed=model(removed,d['descriptors'],d['valid'],d['queries'],d['query_valid']);cf,_=per_example(perturbed,d)
        risk=(d['cost']+20*d['one'].sum(-1))/20
        risk_full=(out['logits'].softmax(-1)*risk).sum(-1);risk_cf=(perturbed['logits'].softmax(-1)*risk).sum(-1)
        importance=(cf-example+risk_cf-risk_full).detach();target=(importance>.01).float()
        # Stored record timestamps identify its WRITE-TIME prefix feature, not final state.
        date=memory[torch.arange(b),slot,-1].detach()
        trace=model.training_diagnostics;which=(trace['events'][:,:,-1]==date[:,None]).long().argmax(-1)
        gate=trace['keep_logits'].gather(1,which[:,None]).squeeze(-1)
        if has.any():retention=F.binary_cross_entropy_with_logits(gate[has],target[has])
        retention=retention+.01*(trace['keep_logits'].sigmoid()*trace['valid']).sum()/trace['valid'].sum().clamp_min(1)
        total=total+retention;importance_mean=float(importance[has].mean()) if has.any() else 0.
    result=total.mean()
    stats={k:float(v.mean().detach()) for k,v in terms.items()}
    stats.update(loss=float(result.detach()),read=float(read_loss.mean().detach()),retention=float(retention.detach()),importance=importance_mean)
    return result,stats


@torch.no_grad()
def evaluate(model,seed,n,h,count=32,family='clean'):
    worlds=[ResearchWorld(seed,i,n,h,family) for i in range(count)];model.eval();state=model.initial(count);writes=np.zeros(count,dtype=int)
    telemetry=[{'completion_probability':[],'stop_probability':[],'healthy':[],'current_brier':[],'stale_reads':0,'latest_hits':0,'wrong_reads':0,'query_reads':0} for _ in worlds]
    work={k:0 for k in ('read_candidates','retained_read_slots','operator_calls','latent_transitions')}
    for t in range(h):state,trace=model.observe(state,torch.from_numpy(np.stack([w.events[t] for w in worlds])));writes+=trace['gate'].numpy().astype(int)
    active=list(range(count))
    for _ in range(2*n+4):
        if not active:break
        public=[worlds[i].candidates() for i in active];values=[torch.from_numpy(np.stack([p[k] for p in public])) for k in range(4)]
        selected=(state[0][active],state[1][active]);out=model(selected,*values);choices=out['logits'].argmax(-1).tolist();prob=out['logits'].softmax(-1)
        for k in work:work[k]+=model.last_work[k]
        remaining=[];receipts=[]
        for j,i in enumerate(active):
            w=worlds[i];t=telemetry[i];stop=next(k for k,c in enumerate(public[j][4]) if c[0]=='stop')
            t['completion_probability'].append(float(out['completion'][j].sigmoid()));t['stop_probability'].append(float(prob[j,stop]));t['healthy'].append(bool(not w.bits.any()))
            if 'current_logits' in out:
                predictions=out['current_logits'][j,:n].sigmoid().numpy();t['current_brier'].append(float(np.mean((predictions-w.bits)**2)))
            if 'state_read_logits' in out:
                chosen=out['state_read_logits'][j,:n,0].argmax(-1).tolist();raw=selected[0][j,:,-22:].numpy()
                for entity,s in enumerate(chosen):
                    key=int(w.keys[entity]);records=[e for e in w.events if (e[0]>0 or e[2]>0) and e[4+key]>.5];latest=max(float(e[21]) for e in records)
                    t['query_reads']+=1
                    matching=s<32 and (raw[s,0]>0 or raw[s,2]>0) and raw[s,4+key]>.5
                    if matching:
                        if float(raw[s,21])<latest-1e-7:t['stale_reads']+=1
                        else:t['latest_hits']+=1
                    else:t['wrong_reads']+=1
            receipt=w.act(public[j][4][choices[j]])
            if not w.stopped:remaining.append(i);receipts.append(receipt)
        memory,occupied=state
        for t in range(max(map(len,receipts),default=0)):
            present=torch.tensor([t<len(r) for r in receipts]);events=torch.from_numpy(np.stack([r[t] if t<len(r) else np.zeros(22,np.float32) for r in receipts]))
            update,trace=model.observe((memory[remaining],occupied[remaining]),events,present)
            memory=memory.index_copy(0,torch.tensor(remaining),update[0]);occupied=occupied.index_copy(0,torch.tensor(remaining),update[1]);writes[remaining]+=trace['gate'].numpy().astype(int)
        state=memory,occupied;active=remaining
    rows=[]
    for i,w in enumerate(worlds):
        r=w.finish(int(writes[i]));initial=r.pop('initial_events');r.pop('keys');r.pop('parents')
        r['public_initial_sha256']=hashlib.sha256(np.asarray(initial,np.float32).tobytes()).hexdigest();r['family']=family;r.update(telemetry[i])
        r['physical_repair']=r['final_faults']==0;r['economic_cost']=r['cost']-r['write_cost']
        r['unnecessary_repairs']=sum(a['kind']=='repair' and not a['before'][a['target']] for a in r['actions'])
        r['healthy_nonstop']=sum(a['kind']!='stop' and not any(a['before']) for a in r['actions'])
        rows.append(r)
    return {'seed':seed,'entities':n,'history':h,'family':family,'worlds':count,'rows':rows,'work':work,
            'state_tensor_bytes':sum(v.numel()*v.element_size() for v in state)//count}


def validate(model,p,smoke=False):return [evaluate(model,p['data']['validation_seed'],n,h,2 if smoke else 16) for n in (2,4,6) for h in (32,96)]


def rank(report):
    if report['status']!='COMPLETE':return (float('inf'),)*4
    cells=report['validation'];rows=[r for c in cells for r in c['rows']]
    return (-sum(np.mean([r['success'] for r in c['rows']])>.95 for c in cells),sum(r['final_faults'] for r in rows),-np.mean([r['success'] for r in rows]),np.mean([r['economic_cost'] for r in rows]))


def fit(job):
    arm,seed,config,lr,folder,dataset,smoke=job;p=json.loads(PROTOCOL.read_text());settings=p['training']
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(seed)
    path=Path(folder);path.mkdir(parents=True,exist_ok=False);start=time.process_time();model=create(arm,config,seed)
    report={'arm':arm,'seed':seed,'config':config,'lr':lr,'source':git('rev-parse','HEAD'),'status':'RUNNING','parameters':sum(v.numel() for v in model.parameters()),'curves':[]}
    try:
        report['admission']=pretrain_admission(model,seed,3 if smoke else settings['admission_updates'])
        train=prepare(model,torch.load(Path(dataset)/'train.pt',weights_only=True));validation=prepare(model,torch.load(Path(dataset)/'validation.pt',weights_only=True))
        # Avoid copying entire raw histories into minibatches.
        optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=lr,weight_decay=.0001)
        rng=torch.Generator().manual_seed(seed+615)
        warm=2 if smoke else settings['belief_warmup_updates'];updates=3 if smoke else settings['controller_updates']
        for step in range(1,warm+updates+1):
            model.train();controller=max(0,step-warm)
            if getattr(model,'retention',False):model.retention_ready.fill_(controller>settings['retention_warmup_controller_updates'])
            ids=torch.randint(len(train['current']),(min(64,len(train['current'])),),generator=rng)
            value,metrics=loss(model,train,ids,step<=warm,counterfactual=bool(getattr(model,'retention_ready',False)))
            if not torch.isfinite(value):raise FloatingPointError('nonfinite loss')
            optimizer.zero_grad();value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            if step in (1,warm,warm+updates) or controller in (400,800):
                model.eval()
                with torch.no_grad():_,v=loss(model,validation,torch.arange(min(128,len(validation['current']))),False,False)
                report['curves'].append({'update':step,'controller_update':controller,'train':metrics,'validation':v,'cpu_seconds':time.process_time()-start});write(path/'progress.json',report['curves'])
            if time.process_time()-start>360:report['status']='CAPPED';break
        else:report['status']='COMPLETE'
        report['updates']=step;report['validation']=validate(model,p,smoke)
        torch.save(model.state_dict(),path/'agent.pt');report['checkpoint_sha256']=digest(path/'agent.pt')
    except Exception:
        import traceback
        report['status']='FAILED';report['error']=traceback.format_exc();torch.save(model.state_dict(),path/'FAILED.pt')
    report['cpu_seconds']=time.process_time()-start;write(path/'report.json',report);return report


def run(directory,smoke=False):
    torch.set_num_threads(1);p=json.loads(PROTOCOL.read_text());root=Path(directory);root.mkdir(parents=True,exist_ok=False)
    if not smoke and git('status','--porcelain','--untracked-files=no'):raise RuntimeError('Commit registered source before main fitting')
    specs=configurations();write(root/'configurations.json',specs);dataset=root/'_datasets';dataset.mkdir()
    offset=10000000 if smoke else 0
    for name,key,count in [('train','train_seed',p['data']['train_snapshots']),('validation','validation_seed',p['data']['validation_snapshots'])]:
        torch.save(raw_data(p['data'][key]+offset,24 if smoke else count),dataset/f'{name}.pt')
    write(root/'data-fingerprints.json',{name:digest(dataset/f'{name}.pt') for name in ('train','validation')})
    tuning=[];fits=[];started=time.monotonic()
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs=[(arm,p['training']['tuning_seed']+offset,specs[arm],lr,str(root/f'tune-{arm}-{i}'),str(dataset),smoke) for arm in p['learned_arms'] for i,lr in enumerate(p['training']['lrs'][:1] if smoke else p['training']['lrs'])]
        for future in as_completed([pool.submit(fit,j) for j in jobs]):
            r=future.result();tuning.append(r);write(root/'tuning.json',tuning);print('tune',r['arm'],r['lr'],r['status'],rank(r),'cpu',round(r['cpu_seconds'],1),flush=True)
        if any(not any(r['arm']==a and r['status']=='COMPLETE' for r in tuning) for a in p['learned_arms']):raise RuntimeError('Incomplete pilots; main test remains closed')
        selected={a:min([r for r in tuning if r['arm']==a],key=rank)['lr'] for a in p['learned_arms']};write(root/'selection.json',selected)
        jobs=[(arm,seed+offset,specs[arm],selected[arm],str(root/f'{arm}-{seed+offset}'),str(dataset),smoke) for arm in p['learned_arms'] for seed in (p['training']['seeds'][:1] if smoke else p['training']['seeds'])]
        # Bounded waves prevent starting new fits after the total CPU cap.
        for start in range(0,len(jobs),2):
            if sum(r['cpu_seconds'] for r in tuning+fits)>10800:break
            for future in as_completed([pool.submit(fit,j) for j in jobs[start:start+2]]):
                r=future.result();fits.append(r);write(root/'fits.json',fits);print('fit',r['arm'],r['seed'],r['status'],rank(r),'cpu',round(r['cpu_seconds'],1),flush=True)
    complete=len(fits)==(7 if smoke else 70) and all(r['status']=='COMPLETE' for r in fits)
    summary={'status':'SMOKE_ONLY' if smoke else 'COMPLETE' if complete else 'INCOMPLETE','source':git('rev-parse','HEAD'),'protocol_sha256':digest(PROTOCOL),'fits':len(fits),'elapsed_seconds':time.monotonic()-started,'fitting_cpu_seconds':sum(r['cpu_seconds'] for r in tuning+fits),'test_opened':False}
    write(root/'summary.json',summary)
    if not smoke and complete:
        # Every LR selection and all70fits close before this first holdout call.
        for r in fits:
            model=create(r['arm'],r['config'],r['seed']).eval();model.load_state_dict(torch.load(root/f"{r['arm']}-{r['seed']}/agent.pt",weights_only=True))
            cells=[evaluate(model,p['data']['test_seed'],n,h,32) for n in (2,4,6) for h in (32,128,512,1024)]
            cells += [evaluate(model,p['data']['test_seed'],n,96,32,family) for family in ('unseen_ids','noisy_initial','noisy_receipts') for n in (2,4,6)]
            fixed=prepare(model,raw_data(p['data']['fixed_query_seed'],p['data']['fixed_test_snapshots']))
            with torch.no_grad():
                state=model.training_state(fixed);out=model(state,fixed['descriptors'],fixed['valid'],fixed['queries'],fixed['query_valid']);qv=fixed['query_valid']
                brier=float(((out['current_logits'].sigmoid()-fixed['current'])**2)[qv].mean())
                error=float(((out['current_logits']>=0)!=fixed['current'].bool())[qv].float().mean())
                fixed_rows=[{'brier':float((((out['current_logits'][i].sigmoid()-fixed['current'][i])**2)*qv[i]).sum()/qv[i].sum())} for i in range(len(qv))]
                if 'state_read_logits' in out:
                    labels=read_targets(state,fixed['queries']);chosen=out['state_read_logits'][:,:,0].argmax(-1);raw=state[0][:,:,-22:]
                    for i in range(len(qv)):
                        latest=labels[i];select=chosen[i];matching=(raw[i,select.clamp_max(31),4:12].argmax(-1)==fixed['queries'][i].argmax(-1))&(select<32)
                        matching=matching&((raw[i,select.clamp_max(31),0]>0)|(raw[i,select.clamp_max(31),2]>0))
                        stale=matching&(raw[i,select.clamp_max(31),21]<raw[i,latest.clamp_max(31),21]-1e-7)&(latest<32)
                        fixed_rows[i].update(queries=int(qv[i].sum()),latest_hits=int(((select==latest)&qv[i]).sum()),stale=int((stale&qv[i]).sum()),wrong=int((~matching&qv[i]).sum()),latest_available=int(((latest<32)&qv[i]).sum()))
            write(root/f"{r['arm']}-{r['seed']}/test.json",{'cells':cells,'fixed_current_brier':brier,'fixed_current_error':error,'fixed_rows':fixed_rows})
            print('test',r['arm'],r['seed'],round(np.mean([v['success'] for c in cells[:12] for v in c['rows']]),4),flush=True)
        summary['test_opened']=True;write(root/'summary.json',summary)
    write(root/'manifest.json',{str(f.relative_to(root)):digest(f) for f in sorted(root.rglob('*')) if f.is_file() and '_datasets' not in f.parts and f.name!='manifest.json'})


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--directory',required=True);a.add_argument('--smoke',action='store_true');args=a.parse_args();run(args.directory,args.smoke)
