"""Registered paired-seed analysis and explicitly programmed public references."""
import argparse
import csv
import io
import json
from pathlib import Path
import numpy as np
from run_identity_dev0615 import PROTOCOL,score,write,IdentityWorld
from summarize_memory_dev061 import interval,signflip,holm


def metrics(cells):
    rows=[r for c in cells for r in c['rows']];m=score(cells)
    m.update(actions=float(np.mean([len(r['actions']) for r in rows])),physical_repair=float(np.mean([r['final_faults']==0 for r in rows])),early_stop_worlds=sum(r['stopped'] and r['final_faults']>0 for r in rows),missing_stop_worlds=sum(not r['stopped'] for r in rows),healthy_not_stopped=sum(not r['stopped'] and r['final_faults']==0 for r in rows),unnecessary_repairs=float(np.mean([sum(a['kind']=='repair' and not a['before'][a['target']] for a in r['actions']) for r in rows])),fifo_overflow_worlds=sum(r['writes']>64 for r in rows))
    tp=fp=fn=tn=0;probs=[];truth=[];gaps=[];boundaries=[]
    for r in rows:
        for a in r['actions']:
            stop=a['kind']=='stop';healthy=not any(a['before'])
            tp+=int(stop and healthy);fp+=int(stop and not healthy);fn+=int(not stop and healthy);tn+=int(not stop and not healthy)
        probs+=r.get('completion_probability',[]);truth+=r.get('healthy',[]);gaps+=r.get('top2_gaps',[]);boundaries+=r.get('epsilon_boundary_distance',[])
    m.update(stop_confusion=dict(tp=tp,fp=fp,fn=fn,tn=tn),stop_precision=tp/(tp+fp) if tp+fp else None,stop_recall=tp/(tp+fn) if tp+fn else None)
    if probs:
        p=np.array(probs);y=np.array(truth);m['completion_brier']=float(np.mean((p-y)**2));m['completion_ECE10']=float(sum(np.mean((p>=i/10)&(p<(i+1)/10+int(i==9)*1e-9))*abs(p[(p>=i/10)&(p<(i+1)/10+int(i==9)*1e-9)].mean()-y[(p>=i/10)&(p<(i+1)/10+int(i==9)*1e-9)].mean()) for i in range(10) if np.any((p>=i/10)&(p<(i+1)/10+int(i==9)*1e-9))))
        m['decisions']=len(gaps);m['top2_below_drift_0035']=float(np.mean(np.array(gaps)<.0035));m['top2_below_epsilon']=float(np.mean(np.array(gaps)<1e-6));m['minimum_epsilon_boundary_distance']=min(boundaries)
    return m


def public_choice(world):
    # Reference only: latest-state table explicitly parses public status/receipts.
    bits={};parents={}
    for e in world.events:
        key=int(e[4:12].argmax());kind=int(e[:4].argmax())
        if kind in (0,2):bits[key]=int(e[20])
        elif kind==1:parents[key]=int(e[12:20].argmax())
    commands=world.candidates()[4]
    for c in commands:
        kind,target=c
        if kind=='stop' and not any(bits.values()):return c
        if kind=='repair':
            key=int(world.keys[target]);parent=parents.get(key)
            if bits[key] and (parent is None or not bits[parent]):return c
    raise AssertionError('Public acyclic reference has no safe-progress action')


def reference(seed,n,h,count,family,horizon=None):
    rows=[]
    for i in range(count):
        w=IdentityWorld(seed,i,n,h,family)
        for _ in range(horizon or 3*n+6):
            w.act(public_choice(w))
            if w.stopped:break
        r=w.finish(0);r['economic_cost']=r['cost'];rows.append(r)
    return dict(seed=seed,entities=n,history=h,family=family,horizon=horizon or 3*n+6,rows=rows)


def run(directory,output):
    root=Path(directory);summary=json.loads((root/'summary.json').read_text());assert summary['status']=='COMPLETE' and summary['test_opened'];p=json.loads(PROTOCOL.read_text());seeds=p['seeds'];arms=p['arms'];tables={a:{} for a in arms};refs=[];records=[]
    for f in ('seen','unseen','fresh'):
        for n in (2,4,6):
            for h in (32,128):refs.append(reference(p['test_seed'],n,h,32,f))
    write(Path(output).with_name('public-reference.json'),{'cells':refs});ref=metrics(refs)
    for a in arms:
        for seed in seeds:
            cells=json.loads((root/f'group-{seed}'/a/'test.json').read_text())['cells'];m=metrics(cells[:18]);families={f:metrics(cells[i*6:(i+1)*6]) for i,f in enumerate(('seen','unseen','fresh'))};m['identity_families']=families
            m['passes_families']=all(v['completion']>=.95 for v in families.values());m['novel_matches_seen']=all(families[f]['completion']>=families['seen']['completion'] for f in ('unseen','fresh'));m['passes_combined']=m['passes_families'] and m['novel_matches_seen'];m['every_cell_passes']=all(score([c])['completion']>=.95 for c in cells[:18]);m['public_cost_regret']=m['cost']-ref['cost'];m['secondary']=[dict(family=c['family'],history=c['history'],horizon=c['horizon'],**metrics([c])) for c in cells[18:]];tables[a][seed]=m
            records.append(dict(arm=a,seed=seed,**{k:m[k] for k in ('completion','physical_repair','faults','cost','actions','passes_families','novel_matches_seen','passes_combined','every_cell_passes','public_cost_regret')}))
    aggregate={a:dict(seeds=len(v),passing_family_target=sum(m['passes_families'] for m in v.values()),passing_combined_target=sum(m['passes_combined'] for m in v.values()),every_cell_passes=sum(m['every_cell_passes'] for m in v.values()),completion_mean=float(np.mean([m['completion'] for m in v.values()])),cost_mean=float(np.mean([m['cost'] for m in v.values()])),faults=sum(m['faults'] for m in v.values()),per_seed=v) for a,v in tables.items()}
    comparisons={}
    for comparator in ('raw_gru_robust','raw_mlp_robust','no_casm_robust'):
        delta=[tables['split_robust'][s]['cost']-tables[comparator][s]['cost'] for s in seeds];comparisons[comparator]={'cost':interval(delta),'p':signflip(delta)}
    adjusted=holm({a:v['p'] for a,v in comparisons.items()})
    for a,v in comparisons.items():v['holm_p']=adjusted[a]
    factorial={k:interval([tables['split_robust'][s][k]-tables['split_bc'][s][k]-tables['fixed_robust'][s][k]+tables['fixed_bc'][s][k] for s in seeds]) for k in ('completion','cost')}
    main=aggregate['split_robust'];gates={'18_of20_combined':main['passing_combined_target']>=18,'zero_primary_faults':main['faults']==0,'public_cost':all(v['cost']<=ref['cost'] for v in main['per_seed'].values()),'cross_hardware_exact':'UNKNOWN until remote replay'}
    result=dict(status='COMPLETE',run=summary,reference=ref,arms=aggregate,comparisons=comparisons,factorial_interaction=factorial,gates=gates,promotion='NOT_ESTABLISHED',commercial='NOT_ESTABLISHED',scope='Paired independent training seeds; seed-world pairs are not independent replications. Exact public synthetic task; no hidden-state or commercial claim.')
    write(output,result);stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records);Path(output).with_suffix('.csv').write_text(stream.getvalue())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.directory,a.output)
