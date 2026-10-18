"""Public/privileged/shortcut references and paired-seed uncertainty analysis."""
import argparse
import csv
import io
import json
from pathlib import Path
import numpy as np
from run_learned_stack_dev055 import write
from summarize_memory_dev061 import interval,signflip,holm
from evaluate_uncertainty_dev060 import calibration


def metrics(cells):
    rows=[r for c in cells for r in c['rows']];result={'worlds':len(rows),'completion':float(np.mean([r['completion'] for r in rows])),'physical_repair':float(np.mean([r['physical_repair'] for r in rows])),'faults':sum(r['faults'] for r in rows),'cost':float(np.mean([r['cost'] for r in rows])),'actions':float(np.mean([len(r['actions']) for r in rows])),'inspections':float(np.mean([sum(a['kind']=='inspect' for a in r['actions']) for r in rows])),'early_stops':sum(r['stopped'] and r['faults']>0 for r in rows),'missing_stop':sum(not r['stopped'] for r in rows),'cost_components':{k:float(np.mean([r['costs'][k] for r in rows])) for k in ('observations','inspections','repairs','terminal')}}
    actions=[a for row in rows for a in row['actions']];stops=[a for a in actions if a['kind']=='stop'];healthy_decisions=sum(not any(a['before']) for a in actions);correct_stops=sum(not any(a['before']) for a in stops)
    result['stop_precision']=correct_stops/len(stops) if stops else None;result['stop_recall_at_healthy_decisions']=correct_stops/healthy_decisions if healthy_decisions else None;result['late_healthy_actions']=healthy_decisions-correct_stops;result['unnecessary_repairs']=sum(a['kind']=='repair' and not a['before'][a['target']] for a in actions)
    margins=[v for row in rows for v in row.get('top2_margin',[])];result['decisions_margin_below_00035']=sum(v<.0035 for v in margins);result['decisions_margin_below_000001']=sum(v<1e-6 for v in margins);result['decision_margins_count']=len(margins)
    predictions=[];public=[];truth=[];stop=[];healthy=[]
    for r in rows:
        predictions+=r.get('posterior',[]);public+=r.get('public_posterior',[]);truth+=r.get('truth',[]);stop+=r.get('stop_probability',[]);healthy+=[not any(a['before']) for a in r['actions']]
    if predictions:
        p=np.array(predictions);q=np.array(public);y=np.eye(12)[truth];kl=np.sum(np.where(q>0,q*np.log(np.maximum(q,1e-30)/np.maximum(p,1e-12)),0),-1);result['posterior_KL']=float(np.mean(kl));result['p95_KL']=float(np.quantile(kl,.95));result['p99_KL']=float(np.quantile(kl,.99));result['query_count']=len(kl);result['multiclass_Brier']=float(np.mean(np.sum((p-y)**2,-1)));result['public_multiclass_Brier']=float(np.mean(np.sum((q-y)**2,-1)));result['completion_calibration']=calibration(p[:,:3].sum(-1),y[:,:3].sum(-1));result['stop_calibration']=calibration(stop,healthy)
    return result


from analyze_belief_dev062_r1 import reference

def run(directory,output):
    root=Path(directory);summary=json.loads((root/'summary.json').read_text());assert summary['status']=='COMPLETE';specs=json.loads((root/'specs.json').read_text());histories=(32,128,512,1024);refs={a:[reference(6341003,h,64,a) for h in histories] for a in ('public','privileged','shortcut')};write(Path(output).with_name('public-reference.json'),refs);reference_metrics={a:{'overall':metrics(c),'strata':[metrics([v]) for v in c]} for a,c in refs.items()};tables={a:{} for a in specs};csvrows=[]
    for arm in specs:
        for seed in range(93401,93421):
            r=json.loads((root/f'group-{seed}'/arm/'test.json').read_text());m=metrics(r['cells'][:4]);m['strata']=[dict(history=h,**metrics([c])) for h,c in zip(histories,r['cells'][:4],strict=True)];m['secondary']=[metrics([c]) for c in r['cells'][4:]];m['fixed']=r['fixed'];m['passes_each_history']=all(v['completion']>=.95 for v in m['strata']);m['physical_guard']=all(n['faults']<=p['faults'] for c,ref in zip(r['cells'][:4],refs['public'],strict=True) for n,p in zip(c['rows'],ref['rows'],strict=True));tables[arm][seed]=m;csvrows.append({'arm':arm,'seed':seed,**{k:m[k] for k in ('completion','physical_repair','faults','cost','actions','inspections','early_stops','missing_stop','passes_each_history','physical_guard')},'fixed_p95_KL':r['fixed']['p95_KL'],'fixed_p99_KL':r['fixed']['p99_KL']})
    aggregate={a:{'passing_seeds_each_history':sum(v['passes_each_history'] for v in rows.values()),'passing_seeds_by_history':{str(h):sum(v['strata'][i]['completion']>=.95 for v in rows.values()) for i,h in enumerate(histories)},'physical_guard_seeds':sum(v['physical_guard'] for v in rows.values()),'faults':sum(v['faults'] for v in rows.values()),'mean_completion':float(np.mean([v['completion'] for v in rows.values()])),'mean_cost':float(np.mean([v['cost'] for v in rows.values()])),'median_fixed_p95_KL':float(np.median([v['fixed']['p95_KL'] for v in rows.values()])),'median_fixed_p99_KL':float(np.median([v['fixed']['p99_KL'] for v in rows.values()])),'per_seed':rows} for a,rows in tables.items()};primary='tac_rel_belief';comparisons={}
    for name in ('gru_rel_belief','mlp_rel_belief','no_casm_rel_belief'):
        delta=[tables[primary][seed]['cost']-tables[name][seed]['cost'] for seed in tables[primary]];comparisons[name]={'cost':interval(delta),'p':signflip(delta)}
    adjusted=holm({name:comparisons[name]['p'] for name in ('gru_rel_belief','mlp_rel_belief')})
    for name,value in adjusted.items():comparisons[name]['holm_p']=value
    effects={}
    for name,coefficients in {'relative_clock':{'tac_rel_belief':.5,'tac_rel_latent':.5,'tac_abs_belief':-.5,'tac_abs_latent':-.5},'belief_bottleneck':{'tac_rel_belief':.5,'tac_abs_belief':.5,'tac_rel_latent':-.5,'tac_abs_latent':-.5},'interaction':{'tac_rel_belief':1,'tac_abs_belief':-1,'tac_rel_latent':-1,'tac_abs_latent':1}}.items():
        delta=[sum(coef*tables[arm][seed]['cost'] for arm,coef in coefficients.items()) for seed in tables[primary]];effects[name]={'cost':interval(delta),'p':signflip(delta),'scope':'Secondary paired-seed factorial effect on this fixed family; not general architectural necessity.'}
    adjusted=holm({name:v['p'] for name,v in effects.items()})
    for name,value in adjusted.items():effects[name]['holm_p']=value
    p=aggregate[primary];gates={'18of20_each_history':p['passing_seeds_each_history']>=18,'physical_no_excess_all_seeds':p['physical_guard_seeds']==20,'median_fixed_p95_KL_005':p['median_fixed_p95_KL']<=.05,'median_fixed_p99_KL_01':p['median_fixed_p99_KL']<=.1,'economic_competitive':all(comparisons[k]['cost']['paired_seed_bootstrap95'][1]<=0 and comparisons[k]['holm_p']<.05 for k in ('gru_rel_belief','mlp_rel_belief')),'interventions':'PENDING_SEPARATE_AUDIT','cross_host':'UNKNOWN_UNTIL_REMOTE_REPLAY'};result={'status':'COMPLETE','run':summary,'primary':primary,'references':reference_metrics,'arms':aggregate,'comparisons':comparisons,'factorial_effects':effects,'gates':gates,'commercial':'NOT_ESTABLISHED','scope':'Posterior-supervised joint health/relation belief. 20seeds share64physical starts across four histories. No autonomous discovery, symbolic CASM or real repair economics.'};write(output,result);stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(csvrows[0]),lineterminator='\n');writer.writeheader();writer.writerows(csvrows);Path(output).with_suffix('.csv').write_text(stream.getvalue());print('GATES',json.dumps(gates),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.directory,a.output)
