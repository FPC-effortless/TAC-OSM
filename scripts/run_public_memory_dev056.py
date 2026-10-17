"""Run the preregistered DEV-056 deadline without adaptive extensions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import traceback
from run_learned_stack_dev055 import ROOT,write,digest,git,fit,bootstrap
import numpy as np
import torch
from tac_osm.development.learned_stack import COSTS,FittedFilter,independent_transitions
from tac_osm.development.public_memory_deadline import (
    DeadlineModel,PublicHazardWorld,PublicFilter,HAZARDS,collect,oracle_transitions,public_features)

PROTOCOL=ROOT/'research/dev056/protocol.json'


@torch.no_grad()
def evaluate(policy,spec,family,tables,smoke=False):
    seed,count=spec
    if smoke: seed,count=seed+10_000_000,3
    worlds=[PublicHazardWorld(seed,i,family) for i in range(count)]
    rows=[{'seed':seed,'index':i,'family':family,'hazard':HAZARDS[family],
           'cost':0.,'acquisition_cost':0.,'fault_exposure':0,'final_faults':0,'actions':[]} for i in range(count)]
    neural=isinstance(policy,DeadlineModel)
    filters=None
    if neural:
        state=policy.initial(torch.zeros(count,11))
    else:
        filters=[PublicFilter(tables[policy],HAZARDS[family],policy=='misspecified_independent') for _ in worlds]
    writes,op_calls=0,0
    started=time.monotonic()
    for tick in range(24):
        events=[w.last for w in worlds]
        if neural:
            public=torch.tensor([public_features(e,HAZARDS[family]) for e in events],dtype=torch.float32)
            state=policy.update(state,public)
            writes+=policy.last_write_count
        else:
            for filt,event in zip(filters,events): filt.update(event,initial=(tick==0))
        if tick<16:
            actions=[int(np.random.default_rng(np.random.SeedSequence([seed,i,tick,5611])).integers(5)) for i in range(count)]
        else:
            if neural:
                expanded=tuple(s[:,None].expand(-1,5,*s.shape[1:]).reshape(-1,*s.shape[1:]) if s is not None else None for s in state)
                probabilities=policy.predict(expanded,torch.arange(5).repeat(count),torch.full((count*5,),tick)).sigmoid().view(count,5,2).numpy()
                op_calls+=policy.last_work['operator_calls']
            else: probabilities=np.array([f.predict() for f in filters])
            actions=(COSTS[None,:]+4*probabilities.sum(-1)).argmin(-1).tolist()
        for row,w,action in zip(rows,worlds,actions):
            event,cost,label=w.step(action)
            row['cost']+=cost
            row['fault_exposure']+=sum(label['after'])
            row['acquisition_cost']+=cost if tick<16 else 0
            row['final_faults']=sum(label['after'])
            row['actions'].append({'action':action,'cost':cost,'after':label['after'],'public':event.tensor()})
    return {'rows':rows,'cost':float(np.mean([r['cost'] for r in rows])),
            'fault_exposure':float(np.mean([r['fault_exposure'] for r in rows])),
            'final_faults':float(np.mean([r['final_faults'] for r in rows])),
            'memory_backend_sample_updates':writes if neural else None,
            'potential_memory_updates':24*count if neural else None,
            'actual_operator_calls':op_calls if neural else None,'seconds':time.monotonic()-started}


def summarize(p,reports,controls,selected):
    if any(reports[str(s)][a]['status']!='COMPLETE' for s in p['model_seeds'] for a in p['arms']):
        return {'status':'INCOMPLETE','decision':'RETIRE_MEMORY_NOVELTY_PURSUIT','reason':'Deadline did not establish superiority; failed cells retained.'}
    def arr(arm,metric='cost'):
        if arm in controls:
            a=np.array([[r[metric] for family in HAZARDS for r in controls[arm][family]['rows']]])
            return np.repeat(a,len(p['model_seeds']),axis=0)
        return np.array([[r[metric] for family in HAZARDS for r in reports[str(seed)][arm]['evaluation'][family]['rows']] for seed in p['model_seeds']])
    rng=np.random.default_rng(756100)
    result={'status':'COMPLETE_UNAUDITED','validation_selected_control':selected,'metrics':{},'headroom':{},'variants':{}}
    for arm in p['arms']+p['controls']:
        result['metrics'][arm]={m:float(arr(arm,m).mean()) for m in ('cost','fault_exposure','final_faults')}
    def guard(a,b):
        size=arr(a).shape[1]//2
        return all(arr(a,m)[s,start:start+size].sum()<=arr(b,m)[s,start:start+size].sum()
                   for m in ('fault_exposure','final_faults') for s in range(len(p['model_seeds'])) for start in (0,size))
    for ref in ('gru_matched',selected):
        delta=bootstrap(arr(ref)/2,arr('public_oracle')/2,rng)
        result['headroom'][ref]={**delta,'reliability_feasible':guard('public_oracle',ref),
                                 'adequate':bool(delta['mean']>=.1 and delta['interval'][0]>0 and guard('public_oracle',ref))}
    adequate=all(x['adequate'] for x in result['headroom'].values())
    accepted=[]
    for variant in ('mtsk','masked_mtsk'):
        comparisons={ref:{**bootstrap(arr(variant),arr(ref),rng,alpha=.05/4),
                          'fault_guard':guard(variant,ref)} for ref in ('gru_matched',selected)}
        passes=bool(adequate and all(x['interval'][1]<0 and x['fault_guard'] for x in comparisons.values()))
        result['variants'][variant]={'comparisons':comparisons,'accepted':passes}
        if passes: accepted.append(variant)
    result['decision']='DEVELOPMENT_CANDIDATE_REQUIRES_REPLICATION' if accepted else 'RETIRE_MEMORY_NOVELTY_PURSUIT'
    result['accepted_variants']=accepted
    result['adequate_public_headroom']=adequate
    result['scope']='Deadline decision for this program, not a universal impossibility claim or retirement of TAC-OSM.'
    return result


def run(output,smoke=False):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    p=json.loads(PROTOCOL.read_text())
    if not smoke and git('status','--porcelain','--untracked-files=no'): raise RuntimeError('commit source first')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();source=git('rev-parse','HEAD')
    data={};labels={}
    for split in ('train','validation'):
        seed,count=p['splits'][split]
        if smoke: seed,count=seed+10_000_000,8
        data[split],labels[split]=collect(seed,count)
    tables={'fitted_hmm':FittedFilter.fit(labels['train'],data['train'][1]),
            'misspecified_independent':independent_transitions(labels['train'],data['train'][1]),
            'public_oracle':oracle_transitions()}
    np.savez(out/'filters.npz',**tables)
    seed,count=p['splits']['baseline_selection']
    selection={a:{f:evaluate(a,[seed,count//2],f,tables,smoke) for f in HAZARDS}
               for a in ('fitted_hmm','misspecified_independent')}
    selected=min(selection,key=lambda a:np.mean([x['cost'] for x in selection[a].values()]))
    write(out/'baseline-selection.json',{'selected':selected,'validation':selection})
    controls={a:{f:evaluate(a,p['splits'][f],f,tables,smoke) for f in HAZARDS} for a in p['controls']}
    write(out/'controls.json',controls)
    reports={}
    for seed in p['model_seeds'][:1] if smoke else p['model_seeds']:
        reports[str(seed)]={}
        for arm in p['arms']:
            print('fit',seed,arm,flush=True)
            model=DeadlineModel(seed,arm)
            try:
                reference=DeadlineModel(seed,'gru_matched').trainable_count
                if arm!='mtsk_reset' and abs(model.trainable_count-reference)/reference>.02: raise AssertionError('parameter mismatch')
                gate_before=model.write_gate.weight.detach().clone() if arm=='masked_mtsk' else None
                training=fit(model,data['train'],data['validation'],seed,p['training'],smoke)
                initial=model.initial(data['train'][0][:1,0])
                training['retained_state_bytes']=sum(s.numel()*s.element_size() for s in initial if s is not None)
                if gate_before is not None:
                    training['write_gate_weight_change_l1']=float((model.write_gate.weight.detach()-gate_before).abs().sum())
                checkpoint=out/f'{seed}-{arm}.pt';torch.save(model.state_dict(),checkpoint)
                result={f:evaluate(model,p['splits'][f],f,tables,smoke) for f in HAZARDS}
                reports[str(seed)][arm]={'status':'COMPLETE','training':training,'evaluation':result,'checkpoint_sha256':digest(checkpoint)}
                print('done',seed,arm,round(training['seconds'],2),{f:x['cost'] for f,x in result.items()},flush=True)
            except Exception:
                reports[str(seed)][arm]={'status':'FAILED','updates':getattr(model,'completed_updates',0),'error':traceback.format_exc()}
                torch.save(model.state_dict(),out/f'{seed}-{arm}-FAILED.pt')
                print(reports[str(seed)][arm]['error'],flush=True)
            write(out/'reports.json',reports)
    summary={'status':'SMOKE_ONLY'} if smoke else summarize(p,reports,controls,selected)
    summary.update(source=source,protocol_sha256=digest(PROTOCOL),smoke=smoke,elapsed_seconds=time.monotonic()-started,
                   torch_version=torch.__version__,numpy_version=np.__version__)
    write(out/'summary.json',summary)
    write(out/'manifest.json',{f.name:digest(f) for f in sorted(out.iterdir()) if f.is_file()})
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();run(args.output,args.smoke)
