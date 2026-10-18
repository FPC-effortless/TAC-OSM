"""Prospective DEV059-R teacher-assisted checkpoint repair; no runtime rules."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from tac_osm.development.learned_agent import LearnedAgent,AgentConfig
from run_fully_learned_dev058 import loss_for
from run_reliability_controls_dev059 import retained_bank,evaluate
from run_learned_stack_dev055 import write,digest,git
PROTOCOL=ROOT/'research/dev059/repair/protocol.json'


def public_routing_targets(ds,slots):
    """Training labels only: latest public facts and public parent relations."""
    attention=torch.full((len(ds['bank']),13,2),slots,dtype=torch.long)
    graph=torch.zeros(len(ds['bank']),13,13)
    raw=ds['bank'].numpy();occupied=ds['occupied'].numpy();descriptors=ds['descriptors'].numpy();valid=ds['valid'].numpy()
    for i in range(len(raw)):
        parents={int(e[4:12].argmax()):int(e[12:20].argmax()) for e,o in zip(raw[i],occupied[i]) if o and e[1]>.5}
        for a,d in enumerate(descriptors[i]):
            if not valid[i,a]:continue
            if d[0]>.5:graph[i,a]=torch.from_numpy(valid[i].astype(np.float32)/valid[i].sum());continue
            key=int(d[3:].argmax())
            for head,kinds in enumerate(((0,2),(1,))):
                matches=[s for s in range(slots) if occupied[i,s] and raw[i,s,4+key]>.5 and any(raw[i,s,k]>.5 for k in kinds)]
                if matches:attention[i,a,head]=max(matches,key=lambda s:raw[i,s,21])
            parent=parents.get(key,key)
            matches=[j for j,other in enumerate(descriptors[i]) if valid[i,j] and other[0]<.5 and int(other[3:].argmax())==parent]
            graph[i,a,matches]=1/len(matches)
    return dict(ds,attention=attention,graph=graph)


def repair_loss(model,ds,ids):
    original,metrics=loss_for(model,ds,ids)
    d={k:v[ids] for k,v in ds.items()}
    state=model.encode_bank(d['bank'],d['occupied'])
    output=model(state,d['descriptors'],d['valid'],d['queries'],d['query_valid'])
    ce=-(d['policy']*output['logits'].log_softmax(-1)).sum(-1)
    extra=3*(d['completion']*ce).mean()
    metrics.update(extra_stop_ce=float(extra.detach()),original_loss=metrics['loss'])
    metrics['loss']=float((original+extra).detach())
    return original+extra,metrics


def cells(model,seed,histories):return [evaluate(model,seed,n,h,64) for n in (2,4,6) for h in histories]
def functional(values):return all(c['completion']>=(.8 if c['entities']==6 else .9) for c in values)
def faults(values):return sum(sum(r['final_faults'] for r in c['rows']) for c in values)
def missing(values):return sum(sum(not r['stopped'] for r in c['rows']) for c in values)


def run(output,data_root):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if git('status','--porcelain','--untracked-files=no'):raise RuntimeError('commit repair source before training')
    p=json.loads(PROTOCOL.read_text());out=Path(output);out.mkdir(parents=True,exist_ok=False);reports=[];models=[]
    repair_started=time.process_time()
    for seed in p['source_checkpoints']:
        t0=time.process_time();source=ROOT/f'results/dev058-main/r0-{seed}';original_report=json.loads((source/'report.json').read_text())
        model=LearnedAgent(AgentConfig(**original_report['config']),seed).eval()
        model.load_state_dict(torch.load(source/'agent.pt',weights_only=True))
        original=LearnedAgent(model.config,seed).eval();original.load_state_dict(model.state_dict())
        for name in ('admission','address'):
            for parameter in getattr(model,name).parameters():parameter.requires_grad_(False)
        train=public_routing_targets(retained_bank(model,torch.load(Path(data_root)/'train.pt',weights_only=True)),model.config.slots)
        optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=p['learning_rate'],weight_decay=.0001)
        sampler=torch.Generator().manual_seed(seed+590060);curves=[]
        for step in range(1,p['updates']+1):
            if time.process_time()-repair_started>300:
                torch.save(model.state_dict(),out/f'CAPPED-{seed}.pt')
                write(out/'summary.json',{'status':'FITTING_BUDGET_EXHAUSTED','reports':reports,'test_opened':False})
                return
            model.train();ids=torch.randint(len(train['bank']),(p['batch'],),generator=sampler)
            loss,metrics=repair_loss(model,train,ids)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            if step==1 or step%150==0 or step==p['updates']:curves.append(dict(update=step,metrics=metrics))
        model.eval();folder=out/str(seed);folder.mkdir();torch.save(model.state_dict(),folder/'agent.pt')
        old=cells(original,p['validation']['seed'],p['validation']['histories']);new=cells(model,p['validation']['seed'],p['validation']['histories'])
        report={'seed':seed,'source':git('rev-parse','HEAD'),'config':model.configuration(),'original_checkpoint_sha256':digest(source/'agent.pt'),
                'checkpoint_sha256':digest(folder/'agent.pt'),'curves':curves,'cpu_seconds':time.process_time()-t0,
                'validation':{'original':old,'repaired':new},'validation_gate':functional(new) and faults(new)<=faults(old)}
        write(folder/'report.json',report);reports.append(report);models.append((original,model));print('repair validation',seed,report['validation_gate'],sum(c['completion'] for c in old)/6,sum(c['completion'] for c in new)/6,flush=True)
    accepted=all(r['validation_gate'] for r in reports)
    if accepted:
        for report,(original,model) in zip(reports,models):
            old=cells(original,p['test']['seed'],p['test']['histories']);new=cells(model,p['test']['seed'],p['test']['histories'])
            write(out/str(report['seed'])/'test.json',{'original':old,'repaired':new})
            report['test_gate']=functional(new) and faults(new)<=faults(old) and missing(new)<=missing(old)
            report['original_missing_stop']=missing(old);report['repaired_missing_stop']=missing(new)
            report['original_faults']=faults(old);report['repaired_faults']=faults(new)
            print('repair test',report['seed'],report['test_gate'],missing(old),missing(new),faults(old),faults(new),flush=True)
    status='VALIDATION_REJECTED' if not accepted else 'TEST_GUARD_FAILED' if not all(r['test_gate'] for r in reports) else 'FUNCTIONAL_GUARD_PASSED'
    if accepted and sum(r['original_missing_stop'] for r in reports)<=sum(r['repaired_missing_stop'] for r in reports):status='NO_STOPPING_IMPROVEMENT'
    write(out/'summary.json',{'study':'DEV-059-R','status':status,'protocol_sha256':digest(PROTOCOL),'source':git('rev-parse','HEAD'),
                            'reports':reports,'test_opened':accepted,'independent_initializations':2,'training':'Two inherited fits; no ten-seed architecture claim.'})
    write(out/'manifest.json',{str(f.relative_to(out)):digest(f) for f in out.rglob('*') if f.is_file() and f.name!='manifest.json'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);parser.add_argument('--data-root',default='results/dev059-main/_datasets')
    args=parser.parse_args();run(args.output,args.data_root)
