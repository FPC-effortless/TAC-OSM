"""Derive registered DEV061 decisions from paired seed/world evidence."""
import argparse
import csv
import io
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from run_structured_memory_dev061 import PROTOCOL,raw_data,prepare,create
from memory_metrics_dev061 import visibility
from run_learned_stack_dev055 import write


def interval(values,unit='seed'):
    x=np.asarray(values,float);rng=np.random.default_rng(861061)
    means=x[rng.integers(len(x),size=(2000,len(x)))].mean(-1)
    return {'difference':float(x.mean()),f'paired_{unit}_bootstrap95':np.quantile(means,[.025,.975]).tolist()}


def signflip(values):
    x=np.asarray(values,float);x=x[x!=0]
    if not len(x):return 1.
    return float(np.mean([abs(np.dot(s,x))>=abs(x.sum())-1e-12 for s in itertools.product((-1,1),repeat=len(x))]))


def holm(values):
    ordered=sorted(values,key=values.get);last=0.;result={}
    for i,k in enumerate(ordered):last=max(last,min(1.,values[k]*(len(ordered)-i)));result[k]=last
    return result


def metrics(cells):
    rows=[r for c in cells for r in c['rows']];actions=[a for r in rows for a in r['actions']]
    mean=lambda x:float(np.mean(x))
    counts={k:0 for k in ('tp','fp','fn','tn')};lag=[];censored=[]
    p=[];labels=[]
    for r in rows:
        for a in r['actions']:
            healthy=not any(a['before']);stop=a['kind']=='stop';counts['tp' if stop and healthy else 'fp' if stop else 'fn' if healthy else 'tn']+=1
        healthy=[i for i,a in enumerate(r['actions']) if not any(a['before'])]
        if healthy:
            if r['success']:lag.append(len(r['actions'])-1-healthy[0])
            else:censored.append(len(r['actions'])-healthy[0])
        elif r['final_faults']==0 and not r['success']:
            # Repair on the final allowed action leaves no next stop opportunity.
            censored.append(0)
        if r.get('completion_probability'):
            p+=r['completion_probability'];labels+=[not any(a['before']) for a in r['actions']]
    decisions=len(actions);queries=sum(r.get('query_reads',0) for r in rows)
    result={'worlds':len(rows),'completion':mean([r['success'] for r in rows]),'physical_repair':mean([r['final_faults']==0 for r in rows]),
       'faults_per_world':mean([r['final_faults'] for r in rows]),'faults':sum(r['final_faults'] for r in rows),
       'economic_cost':mean([r.get('economic_cost',r['cost']-r['write_cost']) for r in rows]),'logical_write_cost':mean([r['write_cost'] for r in rows]),'action_cost_including_terminal':mean([r.get('action_cost',0.) for r in rows]),'observation_cost':mean([r.get('observation_cost',0.) for r in rows]),
       'actions':len(actions)/len(rows),'unnecessary_repairs':sum(a['kind']=='repair' and not a['before'][a['target']] for a in actions)/len(rows),
       'healthy_nonstop':counts['fn']/len(rows),'early_stop_worlds':sum(r['stopped'] and r['final_faults']>0 for r in rows),
       'missing_stop_worlds':sum(not r['stopped'] for r in rows),'stop_confusion':counts,
       'stop_precision':counts['tp']/(counts['tp']+counts['fp']) if counts['tp']+counts['fp'] else None,
       'stop_recall_per_healthy_decision':counts['tp']/(counts['tp']+counts['fn']) if counts['tp']+counts['fn'] else None,
       'healthy_to_stop_lag_mean_observed':mean(lag) if lag else None,'lag_observed':len(lag),'lag_censored':len(censored),
       'lag_censored_lower_bounds':censored,'decisions':decisions}
    if queries:
        result.update(stale_rate=sum(r['stale_reads'] for r in rows)/queries,latest_rate=sum(r['latest_hits'] for r in rows)/queries,wrong_rate=sum(r['wrong_reads'] for r in rows)/queries)
    if p:
        from evaluate_uncertainty_dev060 import calibration
        result['completion_calibration']=calibration(p,labels)
        result['stop_probability_calibration']=calibration([v for r in rows for v in r['stop_probability']],labels)
    return result


@torch.no_grad()
def run(directory,diagnostic,output):
    torch.set_num_threads(1)
    protocol=json.loads(PROTOCOL.read_text());fixed_public=raw_data(protocol['data']['fixed_query_seed'],protocol['data']['fixed_test_snapshots'])
    root=Path(directory);summary=json.loads((root/'summary.json').read_text());assert summary['status']=='COMPLETE' and summary['test_opened']
    runs=json.loads((root/'fits.json').read_text());d=json.loads(Path(diagnostic).read_text())
    tables={};records={};seedrows=[]
    reference={name:metrics(cells[:12]) for name,cells in d['references'].items() if not name.endswith('validation')}
    for fit in runs:
        arm,seed=fit['arm'],fit['seed'];test=json.loads((root/f'{arm}-{seed}/test.json').read_text());clean=test['cells'][:12]
        m=metrics(clean);m['pass']=m['completion']>.95;m['every_cell_pass']=all(np.mean([r['success'] for r in c['rows']])>.95 for c in clean)
        m['fixed_brier']=test['fixed_current_brier'];m['fixed_error']=test['fixed_current_error']
        if 'queries' in test['fixed_rows'][0]:
            model=create(arm,fit['config'],seed).eval();model.load_state_dict(torch.load(root/f'{arm}-{seed}/agent.pt',weights_only=True))
            fixed=prepare(model,fixed_public);state=model.training_state(fixed);out=model(state,fixed['descriptors'],fixed['valid'],fixed['queries'],fixed['query_valid'])
            checked=visibility(state[0][:,:,-22:],state[1],fixed_public['events'],fixed_public['present'],fixed['queries'],fixed['query_valid'],out['state_read_logits'][:,:,0].argmax(-1))
            q=sum(r['queries'] for r in checked);m['fixed_stale']=sum(r['stale'] for r in checked)/q
            m['fixed_latest']=sum(r['latest_hits'] for r in checked)/q;m['fixed_latest_available']=sum(r['latest_available'] for r in checked)/q
            m['retained_status_available']=sum(r['latest_available'] for r in test['fixed_rows'])/q
            m['preferred_slot_hit']=sum(r['latest_hits'] for r in test['fixed_rows'])/q
            action_valid=fixed['valid']&~fixed['descriptors'][:,:,0].bool()
            action_reads=visibility(state[0][:,:,-22:],state[1],fixed_public['events'],fixed_public['present'],fixed['descriptors'][:,:,3:],action_valid,out['read_logits'][:,:,0].argmax(-1))
            aq=sum(r['queries'] for r in action_reads);m['exploratory_action_head0_status_stale']=sum(r['stale'] for r in action_reads)/aq
            m['exploratory_action_head0_status_latest']=sum(r['latest_hits'] for r in action_reads)/aq

        m['public_regret']=m['economic_cost']-reference['public']['economic_cost'];m['private_regret']=m['economic_cost']-reference['privileged']['economic_cost']
        tables.setdefault(arm,{})[seed]=m;records.setdefault(arm,{})[seed]=test
        seedrows.append(dict(arm=arm,seed=seed,**{k:m[k] for k in ('completion','pass','every_cell_pass','physical_repair','faults','economic_cost','public_regret','actions','unnecessary_repairs','early_stop_worlds','missing_stop_worlds','fixed_brier')}))
    aggregates={}
    for arm,rows in tables.items():
        aggregate={'seeds':len(rows),'seeds_above95':sum(r['pass'] for r in rows.values()),'seeds_passing_every_cell':sum(r['every_cell_pass'] for r in rows.values()),'per_seed':rows,'OOD':{}}
        from audit_evidence_gaps_dev059 import wilson
        aggregate['seed_pass_wilson95']=wilson(aggregate['seeds_above95'],len(rows))
        for key in ('completion','economic_cost','faults_per_world','public_regret','actions','unnecessary_repairs','fixed_brier'):
            vals=[r[key] for r in rows.values()];aggregate[key]={'mean':float(np.mean(vals)),'sd':float(np.std(vals,ddof=1)),'min':float(min(vals)),'max':float(max(vals))}
        for j,family in enumerate(('unseen_ids','noisy_initial','noisy_receipts')):
            aggregate['OOD'][family]={seed:metrics(records[arm][seed]['cells'][12+j*3:15+j*3]) for seed in rows}
        aggregate['per_cell_seed_pass_fraction']=[float(np.mean([np.mean([r['success'] for r in records[arm][s]['cells'][i]['rows']])>.95 for s in rows])) for i in range(12)]
        aggregates[arm]=aggregate
    baseline=tables['generic'];names=['entity','entity_aging','entity_retention','entity_aging_retention'];comp={}
    for arm in names:
        seeds=sorted(tables[arm]);m=tables[arm]
        comp[arm]={'primary_exact_p':signflip([int(m[s]['pass'])-int(baseline[s]['pass']) for s in seeds])}
        for k in ('completion','economic_cost','faults_per_world','unnecessary_repairs','fixed_brier','fixed_stale'):
            comp[arm][k]=interval([m[s][k]-baseline[s][k] for s in seeds])
        comp[arm]['cost_vs_public']=interval([m[s]['economic_cost']-reference['public']['economic_cost'] for s in seeds])
        weight=np.array([(2,4,6)[i%3] for i in range(protocol['data']['fixed_test_snapshots'])]);weight=weight/weight.mean()
        delta=np.array([[r['brier'] for r in records[arm][s]['fixed_rows']] for s in seeds])-np.array([[r['brier'] for r in records['generic'][s]['fixed_rows']] for s in seeds])
        comp[arm]['world_paired_brier']=interval(delta.mean(0)*weight,'world')
        comp[arm]['world_paired_cost']={}
        for comparator in ('generic','structured_gru','public_mlp'):
            x=np.array([[r['economic_cost'] for c in records[arm][s]['cells'][:12] for r in c['rows']] for s in seeds])
            y=np.array([[r['economic_cost'] for c in records[comparator][s]['cells'][:12] for r in c['rows']] for s in seeds]);delta=(x-y).mean(0)
            comp[arm]['world_paired_cost'][comparator]=interval(delta,'world')
            comp[arm]['world_paired_cost'][comparator]['scope']='World bootstrap after averaging paired seeds; complements seed intervals, not independent seed-world replicates.'
    adjusted=holm({a:comp[a]['primary_exact_p'] for a in names})
    refval=metrics(d['references']['public_validation'])['economic_cost'];genericval=np.mean([metrics(r['validation'])['economic_cost'] for r in runs if r['arm']=='generic'])
    headroom=genericval-refval
    for arm in names:
        c=comp[arm];a=aggregates[arm];c['holm_p']=adjusted[arm]
        gates={'retention_label_integrity':'retention' not in arm,'reliability_9_of10':a['seeds_above95']>=9,'every_cell_9_of10':a['seeds_passing_every_cell']>=9,
               'primary_advantage':adjusted[arm]<.05 and sum(r['pass'] for r in tables[arm].values())>sum(r['pass'] for r in baseline.values()),
               'stale_decrease_0.01':c['fixed_stale']['difference']<=-.01,'brier_decrease_0.01':c['fixed_brier']['difference']<=-.01,
               'unnecessary_decrease_0.10':c['unnecessary_repairs']['difference']<=-.1,'no_fault_increase':c['faults_per_world']['paired_seed_bootstrap95'][1]<=0,
               'cost_better_than_public':c['cost_vs_public']['paired_seed_bootstrap95'][1]<-.05,'headroom':headroom>=.1,
               'competitive_vs_controls':a['completion']['mean']>=max(aggregates[k]['completion']['mean'] for k in ('structured_gru','public_mlp')) and a['economic_cost']['mean']<=min(aggregates[k]['economic_cost']['mean'] for k in ('structured_gru','public_mlp'))}
        c['gates']=gates;c['promote']=all(gates.values())
    interaction={k:interval([tables['entity_aging_retention'][s][k]-tables['entity_aging'][s][k]-tables['entity_retention'][s][k]+tables['entity'][s][k] for s in sorted(baseline)]) for k in ('completion','economic_cost','fixed_brier','faults_per_world')}
    frozen={s:{mode:metrics(cells) for mode,cells in v['modes'].items()} for s,v in d['frozen'].items()}
    result={'status':'COMPLETE','run':summary,'references':reference,'headroom_validation_economic':headroom,'arms':aggregates,'comparisons':comp,'factorial_interaction':interaction,'frozen_reads':frozen,
       'read_metric_scope':'Registered stale metrics concern top-ranked auxiliary current-state query head0. Additional action-head0 status diagnostics exclude stop and assume that role; neither establishes what information the policy used. Forced frozen reads are conditional-format interventions, with null stop reads.',
       'unvalidated':['continuous identity association NOT_SUPPORTED','unseen cancellation schema NOT_SUPPORTED','noisy stress is not a belief/POMDP validation','CASM-specific superiority not isolated by public-feature MLP','C5 indexed retrieval open'],
       'integrity_findings':{'retention':'Original left-associated STE can perturb stored timestamps and break exact CF label lookup. Both retention mechanism claims unqualified before heldout analysis. Separate DEV061-R1 correction registered; original evidence preserved.','factorial':'As implemented; does not validate intended retention target alignment.'},
       'decision':'PROMOTE' if any(c['promote'] for c in comp.values()) else 'NO_PROMOTION'}
    # NumPy comparisons can produce np.bool_; preserve their value as JSON bool.
    result=json.loads(json.dumps(result,default=lambda value:value.item() if isinstance(value,np.generic) else value,allow_nan=False))
    write(output,result)
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(seedrows[0]));writer.writeheader();writer.writerows(sorted(seedrows,key=lambda r:(r['seed'],r['arm'])))
    Path(output).with_suffix('.csv').write_text(stream.getvalue())


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--directory',required=True);a.add_argument('--diagnostic',required=True);a.add_argument('--output',required=True);args=a.parse_args();run(args.directory,args.diagnostic,args.output)
