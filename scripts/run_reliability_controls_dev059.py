"""Registered CPU-budget DEV059 common-target reliability/control comparison."""
from __future__ import annotations
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
from tac_osm.development.reliability_controllers import create,configurations,DenseController
from run_fully_learned_dev058 import writer_fit
from run_learned_stack_dev055 import write,digest,git
PROTOCOL=ROOT/'research/dev059/protocol.json'


def data(seed,count):
    result=[]
    for i in range(count):
        n=(2,4,6)[i%3];h=(16,32,64,96)[(i//3)%4];w=World(seed,i,n,h)
        rng=np.random.default_rng(np.random.SeedSequence([seed,i,590]))
        if rng.random()<1/3:
            for _ in range(n+4):
                if not w.bits.any():break
                commands=w.candidates()[4]
                w.act(commands[int(teacher(w.bits,w.parents,commands).argmax())])
            for _ in range(int(rng.integers(n+3))):w.act(('inspect',int(rng.integers(n))))
        else:
            for _ in range(int(rng.integers(n+1))):
                commands=[c for c in w.candidates()[4] if c[0]!='stop']
                w.act(commands[int(rng.integers(len(commands)))])
        descriptors,valid,queries,qvalid,commands=w.candidates()
        result.append({'events':np.stack(w.events),'descriptors':descriptors,'valid':valid,'queries':queries,
                       'query_valid':qvalid,'targets':targets(w,commands)})
    length=max(len(r['events']) for r in result)
    if length>192:raise ValueError('registered raw-history capacity')
    events=torch.zeros(count,length,22);present=torch.zeros(count,length,dtype=torch.bool)
    for i,r in enumerate(result):events[i,:len(r['events'])]=torch.from_numpy(r['events']);present[i,:len(r['events'])]=True
    output={'events':events,'present':present}
    for key in ('descriptors','valid','queries','query_valid'):
        output[key]=torch.from_numpy(np.stack([r[key] for r in result]))
    for key in ('policy','future','one','two','done','later_done','cost','completion'):
        output[key]=torch.from_numpy(np.stack([r['targets'][key] for r in result]))
    return output


@torch.no_grad()
def retained_bank(model,ds):
    events,present=ds['events'],ds['present'];m=model.config.slots
    bank=torch.zeros(len(events),m,22);occupied=torch.zeros(len(events),m)
    for start in range(0,len(events),256):
        raw=bank[start:start+256];occ=occupied[start:start+256]
        for t in range(events.shape[1]):
            e=events[start:start+256,t]
            gate,address=model.write_logits(e,occ)
            selected=F.one_hot(address.argmax(-1),m).float()*((gate>=0)&present[start:start+256,t])[:,None]
            raw=raw*(1-selected[:,:,None])+selected[:,:,None]*e[:,None]
            occ=torch.maximum(occ,selected)
        bank[start:start+256]=raw;occupied[start:start+256]=occ
    ds=dict(ds);ds.update(bank=bank,occupied=occupied)
    return ds


def task_loss(model,ds,ids,stop_weight):
    d={k:v[ids] for k,v in ds.items()}
    if isinstance(model,DenseController):state=model.training_state(d)
    else:state=model.encode_bank(d['bank'],d['occupied'])
    out=model(state,d['descriptors'],d['valid'],d['queries'],d['query_valid'])
    policy=-(d['policy']*out['logits'].log_softmax(-1)).sum(-1)
    weighted=(policy*(1+(stop_weight-1)*d['completion'])).mean()
    valid=d['valid'];mask=valid[:,:,None]&d['query_valid'][:,None]
    future=-(d['future']*out['future_logits'].log_softmax(-1)).sum(-1)[valid].mean()
    one=F.binary_cross_entropy_with_logits(out['outcomes'][mask],d['one'][mask])
    choice=out['future_choice'].detach();n=d['queries'].shape[1]
    later=d['two'].gather(2,choice[:,:,None,None].expand(-1,-1,1,n)).squeeze(2)
    two=F.binary_cross_entropy_with_logits(out['later_outcomes'][mask],later[mask])
    later_done=d['later_done'].gather(2,choice[:,:,None]).squeeze(-1)
    verifier=(F.binary_cross_entropy_with_logits(out['completion'],d['completion'])+
              F.binary_cross_entropy_with_logits(out['next_completion'][valid],d['done'][valid])+
              F.binary_cross_entropy_with_logits(out['later_completion'][valid],later_done[valid]))/3
    cost=F.smooth_l1_loss(out['cost'][valid],d['cost'][valid]/20)
    loss=weighted+.3*future+.7*one+.3*two+.5*verifier+.1*cost
    accurate=(d['policy'].gather(1,out['logits'].argmax(-1)[:,None]).squeeze(-1)>0).float()
    healthy=d['completion']>0
    metrics={'loss':float(loss.detach()),'policy_ce':float(policy.mean().detach()),'weighted_policy_ce':float(weighted.detach()),
             'future_ce':float(future.detach()),'one_bce':float(one.detach()),'two_bce':float(two.detach()),
             'completion_bce':float(verifier.detach()),'cost_huber':float(cost.detach()),
             'progress_accuracy':float(accurate.mean()),'healthy_stop_accuracy':float(accurate[healthy].mean()) if healthy.any() else None}
    return loss,metrics


def raw_kind_pretraining(model,seed,updates):
    """Same public signal/noise labels as slotted admission; no filtering."""
    torch.manual_seed(seed+580)
    head=torch.nn.Linear(model.spec['dimension'],1)
    modules=[model.gru,model.project] if model.kind=='gru' else [model.event_encoder]
    parameters=[p for module in modules for p in module.parameters()]+list(head.parameters())
    optimizer=torch.optim.Adam(parameters,lr=.01)
    curves=[]
    for step in range(updates):
        kind=torch.randint(4,(256,));keys=torch.randint(8,(256,2))
        event=torch.cat((F.one_hot(kind,4),F.one_hot(keys[:,0],8),F.one_hot(keys[:,1],8),
                         torch.randint(2,(256,1)),torch.rand(256,1)*.7),-1).float()
        if model.kind=='gru':
            _,hidden=model.gru(event[:,None]);feature=model.project(hidden[0])
        else:feature=model.event_encoder(event)
        logits=head(feature).squeeze(-1);loss=F.binary_cross_entropy_with_logits(logits,(kind!=3).float())
        optimizer.zero_grad();loss.backward();optimizer.step()
        if step%300==0 or step+1==updates:
            curves.append({'update':step+1,'loss':float(loss.detach()),'kind_accuracy':float(((logits>=0)==(kind!=3)).float().mean())})
    return {'updates':updates,'curves':curves,'scope':'Auxiliary training-only classifier discarded; encoder retains learned weights. No admission filtering in control.'}


@torch.no_grad()
def evaluate(model,seed,n,h,count=64,horizon_extra=0):
    model.eval();worlds=[World(seed,i,n,h) for i in range(count)]
    state=model.initial(count);writes=np.zeros(count,dtype=int)
    diagnostics=[{'completion_probabilities':[],'stop_probabilities':[],'healthy_at_decision':[]} for _ in worlds]
    work={key:0 for key in ('read_candidates','retained_read_slots','operator_calls','latent_transitions')}
    for t in range(h):
        state,trace=model.observe(state,torch.from_numpy(np.stack([w.events[t] for w in worlds])))
        writes+=trace['gate'].numpy().astype(int)
    active=list(range(count))
    for _ in range(2*n+4+horizon_extra):
        if not active:break
        public=[worlds[i].candidates() for i in active]
        values=[torch.from_numpy(np.stack([p[k] for p in public])) for k in range(4)]
        out=model((state[0][active],state[1][active]),*values)
        chosen=out['logits'].argmax(-1).tolist();probability=out['logits'].softmax(-1)
        for key in work:work[key]+=model.last_work[key]
        remaining,receipts=[],[]
        for j,i in enumerate(active):
            stop=next(k for k,c in enumerate(public[j][4]) if c[0]=='stop')
            diagnostics[i]['completion_probabilities'].append(float(out['completion'][j].sigmoid()))
            diagnostics[i]['stop_probabilities'].append(float(probability[j,stop]))
            diagnostics[i]['healthy_at_decision'].append(bool(not worlds[i].bits.any()))
            receipt=worlds[i].act(public[j][4][chosen[j]])
            if not worlds[i].stopped:remaining.append(i);receipts.append(receipt)
        memory,occupied=state
        for t in range(max(map(len,receipts),default=0)):
            present=torch.tensor([t<len(r) for r in receipts])
            e=torch.from_numpy(np.stack([r[t] if t<len(r) else np.zeros(22,np.float32) for r in receipts]))
            updated,trace=model.observe((memory[remaining],occupied[remaining]),e,present)
            memory=memory.index_copy(0,torch.tensor(remaining),updated[0]);occupied=occupied.index_copy(0,torch.tensor(remaining),updated[1])
            writes[remaining]+=trace['gate'].numpy().astype(int)
        state=memory,occupied;active=remaining
    rows=[]
    for i,w in enumerate(worlds):
        row=w.finish(int(writes[i]));initial=row.pop('initial_events');row.pop('keys');row.pop('parents')
        row['public_initial_sha256']=hashlib.sha256(np.asarray(initial,np.float32).tobytes()).hexdigest()
        row.update(diagnostics[i]);row['physical_repair']=row['final_faults']==0
        row['healthy_nonstop_actions']=sum(not any(a['before']) and a['kind']!='stop' for a in row['actions'])
        row['harmful_regressions']=sum(not any(a['before']) and any(a['after']) for a in row['actions'])
        row['physical_action_cost']=sum(a['cost'] for a in row['actions'] if a['kind']!='stop')
        rows.append(row)
    return {'seed':seed,'entities':n,'history':h,'worlds':count,'horizon_extra':horizon_extra,
            'completion':float(np.mean([r['success'] for r in rows])),
            'physical_repair':float(np.mean([r['physical_repair'] for r in rows])),
            'final_faults':float(np.mean([r['final_faults'] for r in rows])),
            'actions':float(np.mean([len(r['actions']) for r in rows])),
            'cost':float(np.mean([r['cost'] for r in rows])),
            'physical_action_cost':float(np.mean([r['physical_action_cost'] for r in rows])),
            'work':work,'persistent_state_bytes_per_world':sum(t.numel()*t.element_size() for t in state)//count,'rows':rows}


def validation_metrics(model,p,count):
    return [evaluate(model,p['data']['validation_seed'],n,h,count) for n in (2,4,6) for h in (32,96)]


def fit(job):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    started=time.process_time();arm,seed,config,lr,folder,data_root,settings,smoke=job
    path=Path(folder);path.mkdir(parents=True,exist_ok=False)
    model=create(arm,config,seed)
    report={'arm':arm,'seed':seed,'config':config,'lr':lr,'parameters':sum(p.numel() for p in model.parameters()),
            'status':'RUNNING','stop_weight':1 if arm=='full_common' else 4,'source':git('rev-parse','HEAD')}
    before={name:p.detach().clone() for name,p in model.named_parameters()}
    try:
        if not isinstance(model,DenseController):report['writer']=writer_fit(model,seed,5 if smoke else settings['training']['writer_updates'])
        else:report['kind_pretraining']=raw_kind_pretraining(model,seed,5 if smoke else settings['training']['writer_updates'])
        train=torch.load(Path(data_root)/'train.pt',weights_only=True);validation=torch.load(Path(data_root)/'validation.pt',weights_only=True)
        if not isinstance(model,DenseController):
            train=retained_bank(model,train);validation=retained_bank(model,validation)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=.0001)
        generator=torch.Generator().manual_seed(seed+590)
        curves=[];max_updates=3 if smoke else settings['training']['updates'];t0=time.process_time()
        for update in range(1,max_updates+1):
            model.train();ids=torch.randint(len(train['events']),(min(64,len(train['events'])),),generator=generator)
            loss,metrics=task_loss(model,train,ids,report['stop_weight'])
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            if update==1 or update%settings['training']['log_every']==0 or update==max_updates:
                model.eval()
                with torch.no_grad():_,v=task_loss(model,validation,torch.arange(min(128,len(validation['events']))),report['stop_weight'])
                curves.append({'update':update,'train':metrics,'validation':v,'controller_cpu_seconds':time.process_time()-t0})
                write(path/'progress.json',{'curves':curves})
            if time.process_time()-started>360:
                report['status']='CAPPED';break
        else:report['status']='COMPLETE'
        report.update(updates=update,curves=curves,controller_cpu_seconds=time.process_time()-t0)
        model.eval();torch.save(model.state_dict(),path/'agent.pt');report['checkpoint_sha256']=digest(path/'agent.pt')
        report['validation']=validation_metrics(model,settings,2 if smoke else 64)
        report['parameter_change_l1']={name:float((p.detach()-before[name]).abs().sum()) for name,p in model.named_parameters()}
    except Exception:
        import traceback
        report['status']='FAILED';report['error']=traceback.format_exc();torch.save(model.state_dict(),path/'FAILED.pt')
    report['total_cpu_seconds']=time.process_time()-started
    write(path/'report.json',report)
    return report


def rank(report):
    if report['status']!='COMPLETE':return (float('inf'),)*3
    cells=report['validation']
    return (-sum(c['completion'] for c in cells),sum(c['final_faults'] for c in cells),sum(c['physical_action_cost'] for c in cells))


def run(output,smoke=False):
    torch.set_num_threads(1);p=json.loads(PROTOCOL.read_text());settings=p['prospective'];out=Path(output)
    out.mkdir(parents=True,exist_ok=False)
    if not smoke and git('status','--porcelain','--untracked-files=no'):raise RuntimeError('commit source before main outcomes')
    configs=configurations(settings['models']['target_parameters']);write(out/'configurations.json',configs)
    dataset=out/'_datasets';dataset.mkdir()
    offset=10_000_000 if smoke else 0
    for name,key,count in [('train','train_seed',settings['data']['train_snapshots']),('validation','validation_seed',settings['data']['validation_snapshots'])]:
        value=data(settings['data'][key]+offset,24 if smoke else count)
        torch.save(value,dataset/f'{name}.pt')
    write(out/'data-fingerprints.json',{name:digest(dataset/f'{name}.pt') for name in ('train','validation')})
    reports=[];started=time.monotonic();source=git('rev-parse','HEAD')
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs=[]
        for arm in settings['arms']:
            for index,lr in enumerate(settings['training']['learning_rates'][:1] if smoke else settings['training']['learning_rates']):
                jobs.append((arm,settings['training']['tuning_seed']+offset,configs[arm],lr,str(out/f'tune-{arm}-{index}'),str(dataset),settings,smoke))
        for future in as_completed([pool.submit(fit,j) for j in jobs]):
            report=future.result();reports.append(report);write(out/'tuning.json',reports)
            print('tune',report['arm'],report['lr'],report['status'],rank(report),'cpu',round(report['total_cpu_seconds'],1),flush=True)
        selected={arm:min([r for r in reports if r['arm']==arm],key=rank)['lr'] for arm in settings['arms']}
        if any(min(rank(r) for r in reports if r['arm']==arm)[0]==float('inf') for arm in settings['arms']):raise RuntimeError('incomplete tuning')
        write(out/'selection.json',{'selected_lrs':selected,'source':source,'test_not_opened':True})
        jobs=[]
        for arm in settings['arms']:
            for seed in settings['training']['initialization_seeds'][:1] if smoke else settings['training']['initialization_seeds']:
                jobs.append((arm,seed+offset,configs[arm],selected[arm],str(out/f'{arm}-{seed+offset}'),str(dataset),settings,smoke))
        futures=iter(jobs);pending={}
        from concurrent.futures import wait,FIRST_COMPLETED
        finished=[]
        def launch():
            try:job=next(futures)
            except StopIteration:return False
            pending[pool.submit(fit,job)]=job
            return True
        launch();launch()
        while pending:
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future);report=future.result();finished.append(report)
                write(out/'fits.json',finished)
                print('fit',report['arm'],report['seed'],report['status'],rank(report),'cpu',round(report['total_cpu_seconds'],1),flush=True)
                if sum(r['total_cpu_seconds'] for r in reports+finished)<120*60:launch()
    # All validation selection is now closed. Test all recorded complete fits.
    for report in finished:
        if report['status']!='COMPLETE':continue
        model=create(report['arm'],report['config'],report['seed']).eval()
        directory=out/f"{report['arm']}-{report['seed']}"
        model.load_state_dict(torch.load(directory/'agent.pt',weights_only=True))
        cells=[evaluate(model,settings['data']['test_seed']+offset,n,h,2 if smoke else 64) for n in (2,4,6) for h in (32,96,128)]
        write(directory/'test.json',{'cells':cells})
        print('test',report['arm'],report['seed'],round(np.mean([c['completion'] for c in cells]),4),flush=True)
    write(out/'summary.json',{'study':'DEV-059','source':source,'smoke':smoke,'protocol_sha256':digest(PROTOCOL),
        'fits':len(finished),'complete_fits':sum(r['status']=='COMPLETE' for r in finished),
        'expected_main_fits':5 if smoke else 50,'status':'SMOKE_ONLY' if smoke else 'COMPLETE' if len(finished)==50 and all(r['status']=='COMPLETE' for r in finished) else 'INCOMPLETE',
        'elapsed_seconds':time.monotonic()-started,'cpu_seconds':sum(r['total_cpu_seconds'] for r in reports+finished),
        'dataset_fingerprints':'data-fingerprints.json','scope':'Fixed sample/update budget; actual CPU, state and supervision differences disclosed; no convergence/C5/global superiority claim.'})
    write(out/'manifest.json',{str(f.relative_to(out)):digest(f) for f in out.rglob('*') if f.is_file() and '_datasets' not in f.parts and f.name!='manifest.json'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();run(args.output,args.smoke)
