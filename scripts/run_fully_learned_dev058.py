"""Train/evaluate DEV058; all teachers remain outside learned_agent.py."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from torch.nn import functional as F
from tac_osm.development.learned_agent import LearnedAgent,AgentConfig
from tac_osm.development.learned_agent_world import World,snapshots
from run_learned_stack_dev055 import write,digest,git
PROTOCOL=ROOT/'research/dev058/protocol.json'


def writer_fit(model,seed,updates):
    torch.manual_seed(seed+580)
    parameters=list(model.admission.parameters())+list(model.address.parameters())
    before=torch.cat([p.detach().flatten().clone() for p in parameters])
    optimizer=torch.optim.Adam(parameters,lr=.01)
    curves=[]
    for step in range(updates):
        size=256
        kind=torch.randint(4,(size,));keys=torch.randint(8,(size,2))
        event=torch.cat((F.one_hot(kind,4),F.one_hot(keys[:,0],8),F.one_hot(keys[:,1],8),
                         torch.randint(2,(size,1)),torch.rand(size,1)*.7),-1).float()
        count=torch.randint(model.config.slots,(size,))
        occupied=(torch.arange(model.config.slots)[None]<count[:,None]).float()
        gate,address=model.write_logits(event,occupied)
        loss=F.binary_cross_entropy_with_logits(gate,(kind!=3).float())+F.cross_entropy(address,count)+.02*gate.sigmoid().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
        if step%300==0 or step+1==updates:
            curves.append({'update':step+1,'loss':float(loss.detach()),
                           'gate_accuracy':float(((gate.detach()>=0)==(kind!=3)).float().mean()),
                           'address_accuracy':float((address.detach().argmax(-1)==count).float().mean())})
    changed=float((torch.cat([p.detach().flatten() for p in parameters])-before).abs().sum())
    for p in parameters:p.requires_grad_(False)
    return {'updates':updates,'curves':curves,'parameter_change_l1':changed,'frozen_after_learned_pretraining':True}


@torch.no_grad()
def cache(model,records):
    """Run NEURAL admission/addressing; no oracle filtering of stored events."""
    m=model.config.slots
    banks=[];occupancies=[];retained=0;observed=0
    for start in range(0,len(records),256):
        rows=records[start:start+256];b=len(rows);length=max(len(r['events']) for r in rows)
        events=torch.zeros(b,length,22);present=torch.zeros(b,length,dtype=torch.bool)
        for i,r in enumerate(rows):events[i,:len(r['events'])]=torch.from_numpy(r['events']);present[i,:len(r['events'])]=True
        raw=torch.zeros(b,m,22);occupied=torch.zeros(b,m)
        for t in range(length):
            gate,address=model.write_logits(events[:,t],occupied)
            admit=(gate>=0)&present[:,t];choice=address.argmax(-1)
            selected=F.one_hot(choice,m).float()*admit[:,None]
            raw=raw*(1-selected[:,:,None])+selected[:,:,None]*events[:,t,None]
            occupied=torch.maximum(occupied,selected)
            retained+=int(admit.sum());observed+=int(present[:,t].sum())
        banks.append(raw);occupancies.append(occupied)
    bank=torch.cat(banks);occupied=torch.cat(occupancies)
    data={'bank':bank,'occupied':occupied}
    for key in ('descriptors','valid','queries','query_valid'):
        data[key]=torch.from_numpy(np.stack([r[key] for r in records]))
    for key in records[0]['targets']:
        data[key]=torch.from_numpy(np.stack([r['targets'][key] for r in records]))
    attention=torch.full((len(records),13,2),m,dtype=torch.long)
    for i,r in enumerate(records):
        raw=bank[i].numpy();occ=occupied[i].numpy()>0
        for a,desc in enumerate(r['descriptors']):
            if not r['valid'][a] or desc[0]>0:continue
            key=int(desc[3:].argmax())
            for head,kinds in enumerate(((0,2),(1,))):
                matches=[s for s in range(m) if occ[s] and raw[s,4+key]>.5 and any(raw[s,k]>.5 for k in kinds)]
                if matches:attention[i,a,head]=max(matches,key=lambda s:raw[s,21])
    data['attention']=attention
    return data,{'observed_events':observed,'admitted_writes':retained,'skipped_writes':observed-retained}


def loss_for(model,ds,ids):
    data={k:v[ids] for k,v in ds.items()}
    state=model.encode_bank(data['bank'],data['occupied'])
    result=model(state,data['descriptors'],data['valid'],data['queries'],data['query_valid'])
    valid=data['valid'];qvalid=data['query_valid'];n=qvalid.shape[1]
    policy=-(data['policy']*result['logits'].log_softmax(-1)).sum(-1).mean()
    future=-(data['future']*result['future_logits'].log_softmax(-1)).sum(-1)
    future=future[valid].mean()
    mask=valid[:,:,None]&qvalid[:,None]
    dynamics=F.binary_cross_entropy_with_logits(result['outcomes'][mask],data['one'][mask])
    choice=result['future_choice'].detach()
    later=data['two'].gather(2,choice[:,:,None,None].expand(-1,-1,1,n)).squeeze(2)
    dynamics2=F.binary_cross_entropy_with_logits(result['later_outcomes'][mask],later[mask])
    later_done=data['later_done'].gather(2,choice[:,:,None]).squeeze(-1)
    verifier=(F.binary_cross_entropy_with_logits(result['completion'],data['completion'])+
              F.binary_cross_entropy_with_logits(result['next_completion'][valid],data['done'][valid])+
              F.binary_cross_entropy_with_logits(result['later_completion'][valid],later_done[valid]))/3
    cost=F.smooth_l1_loss(result['cost'][valid],data['cost'][valid]/20)
    read=F.cross_entropy(result['read_logits'][valid].reshape(-1,model.config.slots+1),data['attention'][valid].flatten())
    graph=sum(-(data['graph']*edges.clamp_min(1e-9).log()).sum(-1)[valid].mean() for edges in result['graph_edges'])/max(1,len(result['graph_edges']))
    loss=policy+.3*future+.7*dynamics+.3*dynamics2+.5*verifier+.1*cost+.5*read+.3*graph
    return loss,{'loss':float(loss.detach()),'policy':float(policy.detach()),'future_policy':float(future.detach()),
                 'dynamics':float(dynamics.detach()),'dynamics2':float(dynamics2.detach()),'verifier':float(verifier.detach()),
                 'read':float(read.detach()),'graph':float(graph.detach()),
                 'progress_accuracy':float((data['policy'].gather(1,result['logits'].argmax(-1)[:,None]).squeeze(-1)>0).float().mean())}


@torch.no_grad()
def evaluate(model,seed,entities,history,count=64,lesion='none'):
    model.eval();worlds=[World(seed,i,entities,history) for i in range(count)]
    state=model.initial(count);writes=np.zeros(count,dtype=int);work={'read_candidates':0,'retained_read_slots':0,'operator_calls':0,'latent_transitions':0}
    for t in range(history):
        e=torch.tensor(np.stack([w.events[t] for w in worlds]))
        state,trace=model.observe(state,e,force_write=lesion=='all_writes');writes+=trace['gate'].numpy().astype(int)
    active=list(range(count))
    for _ in range(2*entities+4):
        if not active:break
        packed=[worlds[i].candidates() for i in active]
        tensors=[torch.from_numpy(np.stack([row[k] for row in packed])) for k in range(4)]
        selected=(state[0][active],state[1][active])
        result=model(selected,*tensors,lesion=lesion)
        for k in work:work[k]+=model.last_work[k]
        indices=result['logits'].argmax(-1).tolist();remaining=[];receipts=[]
        for j,i in enumerate(active):
            received=worlds[i].act(packed[j][4][indices[j]])
            if not worlds[i].stopped:remaining.append(i);receipts.append(received)
        if remaining:
            mem,occ=state
            for t in range(max(map(len,receipts))):
                present=torch.tensor([t<len(r) for r in receipts])
                e=torch.tensor(np.stack([r[t] if t<len(r) else np.zeros(22,np.float32) for r in receipts]))
                updated,trace=model.observe((mem[remaining],occ[remaining]),e,present,force_write=lesion=='all_writes')
                mem=mem.index_copy(0,torch.tensor(remaining),updated[0]);occ=occ.index_copy(0,torch.tensor(remaining),updated[1])
                writes[remaining]+=trace['gate'].numpy().astype(int)
            state=mem,occ
        active=remaining
    rows=[w.finish(int(writes[i])) for i,w in enumerate(worlds)]
    return {'entities':entities,'history':history,'seed':seed,'lesion':lesion,'worlds':count,
            'success':float(np.mean([r['success'] for r in rows])),'cost':float(np.mean([r['cost'] for r in rows])),
            'final_faults':float(np.mean([r['final_faults'] for r in rows])),
            'writes':int(writes.sum()),'observations':sum(r['observations'] for r in rows),'work':work,'rows':rows}


def controller_fit(model,train,validation,seed,updates,lr,log_every):
    model.train();generator=torch.Generator().manual_seed(seed+581)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=.0001)
    curves=[];gradients={};started=time.monotonic()
    groups=('encoder','action_encoder','query_encoder','read_query','read_key','read_value','node','graph_q','graph_k','graph_v','operator','operator_selector','transition','verifier','current_verifier','future_policy','policy','cost','outcome')
    before={group:torch.cat([p.detach().flatten().clone() for p in getattr(model,group).parameters()]) for group in groups}
    for step in range(1,updates+1):
        ids=torch.randint(len(train['bank']),(64,),generator=generator)
        loss,metrics=loss_for(model,train,ids)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
        optimizer.zero_grad();loss.backward()
        if step==1:
            gradients={g:sum(float(p.grad.abs().sum()) for p in getattr(model,g).parameters() if p.grad is not None) for g in groups}
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if step==1 or step%log_every==0 or step==updates:
            model.eval()
            with torch.no_grad():_,val=loss_for(model,validation,torch.arange(min(256,len(validation['bank']))))
            point={'update':step,'train':metrics,'validation':val};curves.append(point)
            print('curve',step,round(metrics['loss'],4),round(val['progress_accuracy'],4),round(val['read'],4),flush=True);model.train()
    changes={g:float((torch.cat([p.detach().flatten() for p in getattr(model,g).parameters()])-before[g]).abs().sum()) for g in groups}
    model.eval()
    return {'updates':updates,'seconds':time.monotonic()-started,'curves':curves,'first_gradient_l1':gradients,'parameter_change_l1':changes}


def functional(cells):return all(c['success']>=(.8 if c['entities']==6 else .9) for c in cells)


def run(output,smoke=False):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    p=json.loads(PROTOCOL.read_text());out=Path(output);out.mkdir(parents=True,exist_ok=False)
    source=git('rev-parse','HEAD');started=time.monotonic()
    if not smoke and git('status','--porcelain','--untracked-files=no'):raise RuntimeError('commit source before main outcomes')
    print('generate training snapshots',flush=True)
    offset=10_000_000 if smoke else 0
    train_rows=snapshots(p['training']['train_seed']+offset,32 if smoke else p['training']['training_snapshots'])
    val_rows=snapshots(p['training']['validation_seed']+offset,16 if smoke else p['training']['validation_snapshots'])
    records=[];selected=None
    for recipe in p['development_recipes'][:1] if smoke else p['development_recipes']:
        fits=[]
        for seed in p['training']['seeds'][:1] if smoke else p['training']['seeds']:
            print('fit',recipe['name'],seed,flush=True)
            config=AgentConfig(dimension=recipe['dimension'],read_top_k=recipe['top_k'])
            model=LearnedAgent(config,seed);directory=out/f'{recipe["name"]}-{seed}';directory.mkdir()
            report={'recipe':recipe,'seed':seed,'config':model.configuration(),'source':source}
            try:
                report['writer']=writer_fit(model,seed,3 if smoke else p['training']['writer_updates'])
                train,tc=cache(model,train_rows);validation,vc=cache(model,val_rows)
                report['cache']={'train':tc,'validation':vc}
                report['training']=controller_fit(model,train,validation,seed,3 if smoke else p['training']['controller_updates'],recipe['lr'],p['training']['log_every'])
                torch.save(model.state_dict(),directory/'agent.pt');report['checkpoint_sha256']=digest(directory/'agent.pt')
                validation_cells=[evaluate(model,p['training']['validation_seed']+offset,n,h,3 if smoke else 64) for n in (2,4,6) for h in (32,96)]
                report['validation']=validation_cells;report['functional_validation']=functional(validation_cells)
                report['status']='COMPLETE';fits.append((model,report,directory))
                print('validation',recipe['name'],seed,[(r['entities'],r['history'],r['success']) for r in validation_cells],flush=True)
            except Exception:
                report['status']='FAILED';report['error']=traceback.format_exc();torch.save(model.state_dict(),directory/'FAILED.pt');print(report['error'],flush=True)
            write(directory/'report.json',report);records.append(report)
            write(out/'development.json',records)
            if not smoke and not report.get('functional_validation',False):break
        if smoke or (len(fits)==2 and all(r['functional_validation'] for _,r,_ in fits)):
            selected=fits;break
    test={}
    if selected and not smoke:
        for model,report,directory in selected:
            cells=[evaluate(model,p['training']['test_seed'],n,h) for n in (2,4,6) for h in p['evaluation']['histories']]
            lesions={name:[evaluate(model,p['training']['test_seed'],n,h,64,name) for n in (2,4,6) for h in (32,96)]
                     for name in ('zero_memory','all_writes','uniform_reads','no_casm','zero_dynamics','zero_policy')}
            untrained=LearnedAgent(model.config,report['seed']+100_000).eval()
            untrained_cells=[evaluate(untrained,p['training']['test_seed'],n,h) for n in (2,4,6) for h in (32,96)]
            test[str(report['seed'])]={'cells':cells,'lesions':lesions,'untrained':untrained_cells,'functional_test':functional(cells)}
            write(directory/'test.json',test[str(report['seed'])])
    summary={'study':'DEV-058','source':source,'protocol_sha256':digest(PROTOCOL),'smoke':smoke,
             'status':'SMOKE_ONLY' if smoke else 'FUNCTIONAL_TARGET_MET_UNAUDITED' if selected and all(r['functional_test'] for r in test.values()) else 'FUNCTIONAL_TARGET_NOT_MET',
             'selected_recipe':selected[0][1]['recipe'] if selected else None,
             'selected_seeds':[r['seed'] for _,r,_ in selected] if selected else [],
             'elapsed_seconds':time.monotonic()-started,'torch_version':torch.__version__,'numpy_version':np.__version__,
             'deployment':'LearnedAgent uses only public tensors and learned weights. Simulator/teachers excluded.'}
    write(out/'summary.json',summary);write(out/'manifest.json',{str(f.relative_to(out)):digest(f) for f in sorted(out.rglob('*')) if f.is_file()})
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();run(args.output,args.smoke)
