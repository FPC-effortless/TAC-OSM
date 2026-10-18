"""Public/privileged/shortcut references and paired-seed uncertainty analysis."""
import argparse
import csv
import io
import json
from pathlib import Path
import numpy as np
from data_belief_dev062 import BeliefWorld,COMMANDS
from run_learned_stack_dev055 import write
from summarize_memory_dev061 import interval,signflip,holm
from evaluate_uncertainty_dev060 import calibration


def metrics(cells):
    rows=[r for c in cells for r in c['rows']];result={'worlds':len(rows),'completion':float(np.mean([r['completion'] for r in rows])),'physical_repair':float(np.mean([r['physical_repair'] for r in rows])),'faults':sum(r['faults'] for r in rows),'cost':float(np.mean([r['cost'] for r in rows])),'actions':float(np.mean([len(r['actions']) for r in rows])),'inspections':float(np.mean([sum(a['kind']=='inspect' for a in r['actions']) for r in rows])),'early_stops':sum(r['stopped'] and r['faults']>0 for r in rows),'missing_stop':sum(not r['stopped'] for r in rows),'cost_components':{k:float(np.mean([r['costs'][k] for r in rows])) for k in ('observations','inspections','repairs','terminal')}}
    predictions=[];public=[];truth=[];stop=[];healthy=[]
    for r in rows:
        predictions+=r.get('posterior',[]);public+=r.get('public_posterior',[]);truth+=r.get('truth',[]);stop+=r.get('stop_probability',[]);healthy+=[not any(a['before']) for a in r['actions']]
    if predictions:
        p=np.array(predictions);q=np.array(public);y=np.eye(4)[truth];result['posterior_KL']=float(np.mean(np.sum(np.where(q>0,q*np.log(np.maximum(q,1e-30)/np.maximum(p,1e-30)),0),-1)));result['multiclass_Brier']=float(np.mean(np.sum((p-y)**2,-1)));result['public_multiclass_Brier']=float(np.mean(np.sum((q-y)**2,-1)));result['completion_calibration']=calibration(p[:,0],y[:,0]);result['stop_calibration']=calibration(stop,healthy)
    return result


def reference(seed,history,count,arm,error=.1,inspection=.1):
    rows=[]
    for i in range(count):
        w=BeliefWorld(seed,i,history,error,inspection);b=w.reference()
        for _ in range(10):
            if arm=='public':choice=int(b.policy().argmax())
            elif arm=='privileged':
                options=[j for j,(kind,target) in enumerate(COMMANDS) if kind=='repair' and w.bits[target] and (w.parents[target]<0 or not w.bits[w.parents[target]])];choice=options[0] if options else 0
            else:
                latest={}
                for e in w.events:
                    if e[0] or e[3] or e[4]:latest[int((w.identities@e[6:22]).argmax())]=int(e[38])
                options=[j for j,(kind,target) in enumerate(COMMANDS) if kind=='repair' and latest.get(target,0)];choice=options[0] if options else 0
            receipt=w.act(choice)
            # Public posterior is not labelled exact under privileged selection.
            b.update(choice,receipt)
            if w.stopped:break
        rows.append(w.finish())
    return dict(seed=seed,history=history,worlds=count,horizon=10,error=error,inspection=inspection,rows=rows)


def run(directory,output):
    root=Path(directory);summary=json.loads((root/'summary.json').read_text());assert summary['status']=='COMPLETE';spec=json.loads((root/'specs.json').read_text());refs={a:[reference(6261003,h,128,a) for h in (32,128)] for a in ('public','privileged','shortcut')};write(Path(output).with_name('public-reference.json'),refs);references={a:metrics(c) for a,c in refs.items()};tables={a:{} for a in spec};csvrows=[]
    for a in spec:
        for seed in range(92601,92621):
            r=json.loads((root/f'group-{seed}'/a/'test.json').read_text());m=metrics(r['cells'][:2]);m['fixed']=r['fixed'];m['secondary']=[metrics([c]) for c in r['cells'][2:]];m['passes_completion']=m['completion']>=.95;m['public_cost_regret']=m['cost']-references['public']['cost'];m['privileged_cost_regret']=m['cost']-references['privileged']['cost'];tables[a][seed]=m;csvrows.append(dict(arm=a,seed=seed,**{k:m[k] for k in ('completion','physical_repair','faults','cost','actions','inspections','passes_completion','public_cost_regret','privileged_cost_regret')}))
    aggregate={a:{'passing_seeds':sum(m['passes_completion'] for m in v.values()),'faults':sum(m['faults'] for m in v.values()),'mean_completion':float(np.mean([m['completion'] for m in v.values()])),'mean_cost':float(np.mean([m['cost'] for m in v.values()])),'median_fixed_KL':float(np.median([m['fixed']['posterior_KL'] for m in v.values()])),'median_expected_Brier_excess':float(np.median([m['fixed']['expected_Brier_excess'] for m in v.values()])),'per_seed':v} for a,v in tables.items()};comparisons={}
    for a in ('gru','mlp','no_casm'):
        delta=[tables['tac'][s]['cost']-tables[a][s]['cost'] for s in tables[a]];comparisons[a]={'cost':interval(delta),'p':signflip(delta)}
    adjusted=holm({a:comparisons[a]['p'] for a in ('gru','mlp')})
    for a,v in adjusted.items():comparisons[a]['holm_p']=v
    a=aggregate['tac'];gates={'18_of20_95_completion':a['passing_seeds']>=18,'median_KL_002':a['median_fixed_KL']<=.02,'median_Brier_excess_002':a['median_expected_Brier_excess']<=.02,'zero_primary_faults':a['faults']==0,'competitive':all(comparisons[k]['cost']['paired_seed_bootstrap95'][1]<=0 and comparisons[k]['holm_p']<.05 for k in ('gru','mlp')),'cross_host':'UNKNOWN until remote execution'}
    result={'status':'COMPLETE','run':summary,'references':references,'arms':aggregate,'comparisons':comparisons,'gates':gates,'promotion':'NOT_ESTABLISHED','commercial':'NOT_ESTABLISHED','scope':'Posterior-supervised bounded two-entity task, public action provenance, known-law teacher outside deployment. Information and bounded-planner limits differ from neural representation/policy errors. No general learned-physics or commercial claim.'};write(output,result);stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows);Path(output).with_suffix('.csv').write_text(stream.getvalue())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.directory,a.output)
