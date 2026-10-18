"""Post-hoc DEV058 teacher, admission and termination audit. No retraining."""
import argparse
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from tac_osm.development.learned_agent import LearnedAgent,AgentConfig
from tac_osm.development.learned_agent_world import World,teacher
from run_reliability_controls_dev059 import evaluate
from run_learned_stack_dev055 import write,digest


def wilson(success,total):
    z=1.959963984540054;p=success/total;d=1+z*z/total
    center=(p+z*z/(2*total))/d;radius=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return [center-radius,center+radius]


def public_reference(events,descriptors,valid):
    faults={};parents={}
    for e in events:
        kind=int(np.argmax(e[:4]));key=int(np.argmax(e[4:12]))
        if kind in (0,2):faults[key]=int(e[20])
        elif kind==1:parents[key]=int(np.argmax(e[12:20]))
    active_keys={int(np.argmax(d[3:])) for d,v in zip(descriptors,valid) if v and not d[0]}
    if not active_keys<=faults.keys():raise ValueError('exact public initial state required by reference')
    for i,(d,v) in enumerate(zip(descriptors,valid)):
        if not v:continue
        kind=int(d[:3].argmax());key=int(d[3:].argmax())
        if not any(faults[k] for k in active_keys) and kind==0:return i
        if kind==2 and faults[key] and (key not in parents or not faults[parents[key]]):return i
    raise AssertionError('no public safe-progress action')


def reference_cells(seed):
    cells=[]
    for n in (2,4,6):
        for h in (32,96,128):
            rows=[]
            for i in range(64):
                w=World(seed,i,n,h);writes=sum(int(e[:4].argmax())!=3 for e in w.events)
                for _ in range(2*n+4):
                    d,v,_,_,commands=w.candidates()
                    choice=public_reference(w.events,d,v)
                    private=int(teacher(w.bits,w.parents,commands).argmax())
                    assert choice==private,'privileged teacher differs from public reference'
                    receipt=w.act(commands[choice]);writes+=len(receipt)
                    if w.stopped:break
                row=w.finish(writes);row.pop('initial_events');row.pop('keys');row.pop('parents');rows.append(row)
            cells.append({'entities':n,'history':h,'rows':rows})
    return cells


def summarize_rows(rows):
    def mean(values):return float(np.mean(values))
    return {'worlds':len(rows),'completion':mean([r['success'] for r in rows]),
            'physical_repair':mean([r['final_faults']==0 for r in rows]),
            'actions_per_world':mean([len(r['actions']) for r in rows]),
            'cost':mean([r['cost'] for r in rows]),
            'action_cost_including_terminal':mean([r['action_cost'] for r in rows]),
            'observation_cost':mean([r['observation_cost'] for r in rows]),'write_cost':mean([r['write_cost'] for r in rows]),
            'residual_faults':sum(r['final_faults'] for r in rows),'missing_stop':sum(not r['stopped'] for r in rows),
            'completion_wilson95':wilson(sum(r['success'] for r in rows),len(rows)),
            'physical_repair_wilson95':wilson(sum(r['final_faults']==0 for r in rows),len(rows))}


@torch.no_grad()
def admission(model,cells):
    counts={'tp':0,'fp':0,'tn':0,'fn':0};kinds={str(k):{'total':0,'admitted':0} for k in range(4)}
    for cell in cells:
        events=[]
        for row in cell['rows']:
            events.extend(row['initial_events'])
            for a in row['actions']:events.extend(a['receipt'])
        e=torch.tensor(np.asarray(events,np.float32))
        predicted=model.admission(e).squeeze(-1)>=0
        actual=e[:,:4].argmax(-1)!=3
        counts['tp']+=int((predicted&actual).sum());counts['fp']+=int((predicted&~actual).sum())
        counts['tn']+=int((~predicted&~actual).sum());counts['fn']+=int((~predicted&actual).sum())
        for k in range(4):
            mask=e[:,:4].argmax(-1)==k;kinds[str(k)]['total']+=int(mask.sum());kinds[str(k)]['admitted']+=int((mask&predicted).sum())
    total=sum(counts.values());signal=counts['tp']+counts['fn']
    return dict(counts,total=total,signal_base_rate=signal/total,
                precision=counts['tp']/max(1,counts['tp']+counts['fp']),recall=counts['tp']/max(1,signal),
                skipped_fraction=(counts['tn']+counts['fn'])/total,by_public_kind=kinds,
                conclusion='Classification against explicit kind labels; high skip rate tracks noise base rate. Counts are LOGICAL admissions, not physical memory-traffic savings. All observations, admission/address/value encoders and dense memory update tensor operations still execute even with a closed gate.')


@torch.no_grad()
def failed_decisions(model,cells):
    details=[]
    for cell in cells:
        n,h=cell['entities'],cell['history'];worlds=[World(cell['seed'],i,n,h) for i in range(64)]
        state=model.initial(64);raw=torch.zeros(64,model.config.slots,22)
        def consume(indices,events,present=None):
            nonlocal state,raw
            sub=(state[0][indices],state[1][indices]);updated,trace=model.observe(sub,events,present)
            memory,occupied=state
            memory=memory.index_copy(0,torch.tensor(indices),updated[0]);occupied=occupied.index_copy(0,torch.tensor(indices),updated[1]);state=memory,occupied
            for j,i in enumerate(indices):
                if trace['gate'][j]:raw[i,trace['address'][j]]=events[j]
        for t in range(h):consume(list(range(64)),torch.from_numpy(np.stack([w.events[t] for w in worlds])))
        active=list(range(64))
        for step in range(2*n+4):
            if not active:break
            public=[worlds[i].candidates() for i in active]
            tensor=[torch.from_numpy(np.stack([p[k] for p in public])) for k in range(4)]
            result=model((state[0][active],state[1][active]),*tensor)
            chosen=result['logits'].argmax(-1).tolist();prob=result['logits'].softmax(-1)
            remaining=[];receipts=[]
            for j,i in enumerate(active):
                w=worlds[i];command=public[j][4][chosen[j]]
                old=cell['rows'][i]
                assert (command[0],int(command[1]))==(old['actions'][step]['kind'],old['actions'][step]['target'])
                if not old['success']:
                    stop=next(a for a,c in enumerate(public[j][4]) if c[0]=='stop')
                    target_key=int(w.keys[command[1]]) if command[1]>=0 else -1
                    top=int(result['read_logits'][j,chosen[j],0].argmax())
                    recorded=raw[i,top] if top<model.config.slots else None
                    matching=[s for s in range(model.config.slots) if state[1][i,s] and raw[i,s,4+target_key]>.5 and (raw[i,s,0]>.5 or raw[i,s,2]>.5)] if target_key>=0 else []
                    latest=max(matching,key=lambda s:float(raw[i,s,21])) if matching else None
                    details.append({'entities':n,'history':h,'index':i,'step':step,'command':list(command),
                        'healthy':bool(not w.bits.any()),'faults_before':w.bits.tolist(),
                        'completion_probability':float(result['completion'][j].sigmoid()),'stop_probability':float(prob[j,stop]),
                        'chosen_probability':float(prob[j,chosen[j]]),'top_status_read_slot':top,
                        'top_status_record':recorded.tolist() if recorded is not None else None,
                        'latest_status_record':raw[i,latest].tolist() if latest is not None else None,
                        'top_is_latest':top==latest if latest is not None else None})
                received=w.act(command)
                if not w.stopped:remaining.append(i);receipts.append(received)
            for t in range(max(map(len,receipts),default=0)):
                present=torch.tensor([t<len(r) for r in receipts]);events=torch.from_numpy(np.stack([r[t] if t<len(r) else np.zeros(22,np.float32) for r in receipts]))
                consume(remaining,events,present)
            active=remaining
    healthy=[r for r in details if r['healthy']]
    return {'decisions':details,'healthy_failed_world_decisions':len(healthy),
            'mean_completion_on_healthy_failed_world_decisions':float(np.mean([r['completion_probability'] for r in healthy])) if healthy else None,
            'mean_stop_probability_on_healthy_failed_world_decisions':float(np.mean([r['stop_probability'] for r in healthy])) if healthy else None,
            'healthy_top_read_not_latest':sum(r['top_is_latest'] is False for r in healthy),
            'interpretation':'Correlational post-hoc diagnostics. Top read score is not the entire top-k weighted representation; no causal over-caution diagnosis.'}


def run(directory,output):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    root=Path(directory);references=reference_cells(5581003);refrows=[r for c in references for r in c['rows']]
    result={'study':'DEV-059 retrospective DEV058 audit','reference':summarize_rows(refrows),
            'teacher_public_reference_actions_identical':True,'teacher_reference_cells':references,'checkpoints':{},
            'limitations':['Teacher is a safe-progress reference, not a proved globally optimal planner.','Wilson intervals are conditional on each trained checkpoint and independent generated world fixtures, not training-seed uncertainty.','All initial public records are paid; reference write costs use the same explicit signal-write convention.','Exact public history makes privileged and public reference actions identical here. Partial-observation teacher regret requires an information-matched public baseline.']}
    for seed in (7581,7582):
        path=root/f'r0-{seed}';report=json.loads((path/'report.json').read_text());test=json.loads((path/'test.json').read_text())
        assert digest(path/'agent.pt')==report['checkpoint_sha256']
        model=LearnedAgent(AgentConfig(**report['config'])).eval();model.load_state_dict(torch.load(path/'agent.pt',weights_only=True))
        rows=[r for c in test['cells'] for r in c['rows']]
        summary=summarize_rows(rows);summary['paired_cost_gap_vs_teacher']=float(np.mean([r['cost']-ref['cost'] for r,ref in zip(rows,refrows)]))
        summary['admission']=admission(model,test['cells']);summary['failure_diagnostic']=failed_decisions(model,test['cells'])
        slack=[evaluate(model,5581003,n,h,64,4) for n in (2,4,6) for h in (32,96,128)]
        summary['budget_slack_plus4']=summarize_rows([r for c in slack for r in c['rows']]);summary['budget_slack_cells']=slack
        result['checkpoints'][str(seed)]=summary
        print('audit',seed,summary['completion'],summary['physical_repair'],summary['admission']['precision'],summary['admission']['recall'],flush=True)
    write(output,result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',default='results/dev058-main');parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.directory,args.output)
