"""Read-only FIT/schedule algebra. No Torch, model, optimizer or DEV reads."""
from collections import Counter
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo',type=Path,required=True,help='Frozen TAC-OSM checkout')
parser.add_argument('--fit',type=Path,required=True,help='Authenticated FIT.json only; never DEV')
parser.add_argument('--output',type=Path,required=True,help='New JSON path; existing outputs are never overwritten')
args=parser.parse_args()
ROOT=args.repo.resolve()
HERE=ROOT/'research/mathematical-engineering/langacquire097'
FIT=args.fit.resolve()
OUT=args.output.resolve()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
canonical=lambda value:json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
register=json.loads((HERE/'REGISTER.json').read_text())
assert sha(FIT)==register['data']['FIT.json_sha256']=='510b83352b0bb3b14791009735ff7680e7bb6bf622319c9c0391fb923c7eeeb1'
rows=json.loads(FIT.read_text())
assert len(rows)==152 and Counter(r['domain'] for r in rows)=={'lexical':128,'arc':24}
assert all(r['partition']=='FIT' for r in rows)
seeds=register['seeds'];steps=register['training']['steps'];batch_size=register['training']['batch_size']
assert seeds==[1701,1702,1703,1704,1705] and steps==256 and batch_size==4
profiles={f:json.loads((HERE/'profiles'/f'PROFILE_{f}.json').read_text()) for f in register['families']}
lengths={r['id']:{'domain':r['domain'],'acquisition':len(r['acquisition_bytes']),'query':len(r['query_bytes']),'target':len(r['target_bytes'])} for r in rows}
for x in lengths.values():
 x['full']=x['acquisition']+x['query']+x['target']-1
 x['acquisition_internal_labels']=x['acquisition']-1
assert all(x['target']==4 and x['full']==99 for x in lengths.values() if x['domain']=='lexical')
groups={d:[i for i,r in enumerate(rows) if r['domain']==d] for d in ('lexical','arc')}

def fraction_record(value):
 return {'value':float(value),'exact_numerator':str(value.numerator),'exact_denominator':str(value.denominator)}

def distributions(values):
 return {'minimum':min(values),'maximum':max(values),'mean':float(np.mean(values)),
         'distribution':{str(k):v for k,v in sorted(Counter(values).items())}}

summaries=[]
for seed in seeds:
 rng=np.random.default_rng(seed)
 planned=[]
 for _ in range(steps):
  batch=[]
  for d in ('lexical','arc'):batch.extend(int(i) for i in rng.choice(groups[d],size=2,replace=False))
  planned.append(batch)
 ids=[[rows[i]['id'] for i in batch] for batch in planned]
 schedule_hash=hashlib.sha256(canonical(ids)).hexdigest()
 matches=[]
 for f,profile in profiles.items():
  for objective in ('full','query'):
   p=next(r for r in profile['profiles'] if r['seed']==seed and r['objective']==objective)
   assert p['schedule_sha256']==schedule_hash
   matches.append({'family':f,'objective':objective,'schedule_sha256':p['schedule_sha256']})
 totals={d:Counter() for d in groups};freq=Counter();batches=[]
 per_step={objective:{d:[] for d in groups} for objective in ('full','query')}
 lexical_target_full=[];lexical_target_query=[];all_target_full=[];normalizer_ratios=[]
 case_coefficients={objective:{r['id']:Fraction(0) for r in rows} for objective in ('full','query')}
 for step,batch in enumerate(planned):
  mass={d:Counter() for d in groups}
  for i in batch:
   r=rows[i];l=lengths[r['id']];d=r['domain'];freq[r['id']]+=1
   mass[d]['examples']+=1
   for name in ('full','target','acquisition_internal_labels','query'):mass[d][name]+=l[name]
  full=sum(v['full'] for v in mass.values());target=sum(v['target'] for v in mass.values())
  assert mass['lexical']['examples']==mass['arc']['examples']==2
  assert mass['lexical']['full']==198 and mass['lexical']['target']==8
  for d in groups:
   totals[d].update(mass[d])
   per_step['full'][d].append(Fraction(mass[d]['full'],full))
   per_step['query'][d].append(Fraction(mass[d]['target'],target))
  lexical_target_full.append(Fraction(8,full));lexical_target_query.append(Fraction(8,target))
  all_target_full.append(Fraction(target,full));normalizer_ratios.append(Fraction(full,target))
  for i in batch:
   r=rows[i];l=lengths[r['id']]
   case_coefficients['full'][r['id']]+=Fraction(l['full'],full*steps)
   case_coefficients['query'][r['id']]+=Fraction(l['target'],target*steps)
  batches.append({'step':step+1,'episode_ids':ids[step],'mass_by_domain':{d:dict(v) for d,v in mass.items()},
                  'full_normalizer':full,'query_normalizer':target,
                  'arc_full_loss_coefficient_numerator':mass['arc']['full'],
                  'arc_query_loss_coefficient_numerator':mass['arc']['target']})
 seen_total=sum(v['full'] for v in totals.values());query_total=sum(v['target'] for v in totals.values())
 assert seen_total==next(r for r in profiles['tac']['profiles'] if r['seed']==seed)['planned_tokens_seen']
 avg=lambda values:sum(values,Fraction(0))/steps
 per_step_summary={o:{d:fraction_record(avg(v)) for d,v in by.items()} for o,by in per_step.items()}
 pooled={o:{d:fraction_record(Fraction(totals[d]['full' if o=='full' else 'target'],seen_total if o=='full' else query_total)) for d in groups} for o in ('full','query')}
 summaries.append({'seed':seed,'schedule_sha256':schedule_hash,'frozen_profile_bindings_verified':matches,
  'scheduled_examples':steps*batch_size,'scheduled_examples_by_domain':{d:totals[d]['examples'] for d in groups},
  'supervised_labels_by_domain_and_segment':{d:dict(v) for d,v in totals.items()},
  'full_supervised_targets':seen_total,'query_supervised_targets':query_total,
  'mean_per_step_loss_coefficient_mass':per_step_summary,'pooled_supervised_target_share':pooled,
  'arc_domain_mass_increase_query_minus_full':fraction_record(avg(per_step['query']['arc'])-avg(per_step['full']['arc'])),
  'lexical_future_target_loss_coefficient_mass':{'full':fraction_record(avg(lexical_target_full)),'query':fraction_record(avg(lexical_target_query))},
  'future_Y_mass_within_full_loss':fraction_record(avg(all_target_full)),
  'same_Y_byte_coefficient_multiplier_query_over_full':{'minimum':float(min(normalizer_ratios)), 'maximum':float(max(normalizer_ratios)), 'mean':float(avg(normalizer_ratios))},
  'per_step_arc_loss_share':{o:{'minimum':float(min(per_step[o]['arc'])),'maximum':float(max(per_step[o]['arc']))} for o in per_step},
  'exposure_repeat_counts':{d:{'unique_episodes_sampled':sum(freq[rows[i]['id']]>0 for i in indices),'available_episodes':len(indices),
                             'minimum_occurrences':min(freq[rows[i]['id']] for i in indices),'maximum_occurrences':max(freq[rows[i]['id']] for i in indices),
                             'mean_occurrences_per_available_episode':sum(freq[rows[i]['id']] for i in indices)/len(indices)} for d,indices in groups.items()},
  'episode_loss_coefficient_mass':{r['id']:{'domain':r['domain'],'occurrences':freq[r['id']], 'full':float(case_coefficients['full'][r['id']]),'query':float(case_coefficients['query'][r['id']])} for r in rows},
  'batches':batches})
summary={
 'mean_per_step_domain_mass':{o:{d:float(np.mean([s['mean_per_step_loss_coefficient_mass'][o][d]['value'] for s in summaries])) for d in groups} for o in ('full','query')},
 'pooled_target_counts_all_five_schedules':{d:{o:sum(s['supervised_labels_by_domain_and_segment'][d]['full' if o=='full' else 'target'] for s in summaries) for o in ('full','query')} for d in groups},
 'arc_domain_mass_increase_query_minus_full_mean':float(np.mean([s['arc_domain_mass_increase_query_minus_full']['value'] for s in summaries])),
 'lexical_Y_mass_mean':{o:float(np.mean([s['lexical_future_target_loss_coefficient_mass'][o]['value'] for s in summaries])) for o in ('full','query')},
 'future_Y_mass_full_mean':float(np.mean([s['future_Y_mass_within_full_loss']['value'] for s in summaries])),
 'same_Y_byte_coefficient_multiplier_mean':float(np.mean([s['same_Y_byte_coefficient_multiplier_query_over_full']['mean'] for s in summaries])),
}
audit={'schema':'langacquire097.objective-allocation-audit.v1','status':'READ_ONLY_FIT_SCHEDULE_MATHEMATICS_NOT_AN_OUTCOME',
 'source_bindings':{'FIT_json_sha256':sha(FIT),'train_py_sha256':sha(HERE/'train.py'),'objective_py_sha256':sha(HERE/'objective.py'),
                    'REGISTER_json_sha256':sha(HERE/'REGISTER.json'),'numpy_version':np.__version__},
 'permissions_observed':{'model_forwards':0,'optimizer_steps':0,'model_checkpoints_read':0,'DEV_inputs_or_outcomes_read':False,'frozen_files_changed':False},
 'exact_registered_loss':{
  'query_step':'(sum over 8 lexical Y-byte cross-entropies + sum over y_i+y_j ARC Y-byte cross-entropies)/(8+y_i+y_j)',
  'full_step':'(sum over 198 lexical next-byte cross-entropies + sum over n_i+n_j ARC next-byte cross-entropies)/(198+n_i+n_j)',
  'arc_y_i':'len(target_bytes) for the sampled ARC episode',
  'arc_n_i':'len(acquisition_bytes)+len(query_bytes)+len(target_bytes)-1',
  'full_segment_counts_per_episode':'A_internal=len(A)-1;Q_labels=len(Q);Y_labels=len(Y)',
  'reduction':'Each optimizer step minimizes a batch mean over active target bytes, not a domain-balanced or example-balanced mean.',
  'step_mean_vs_pool':'The mean of per-step domain shares is the exact allocation of the uniformly weighted registered batch losses. A pooled-byte share uses different batch weights and is reported separately.'},
 'FIT_length_distributions':{d:{key:distributions([l[key] for l in lengths.values() if l['domain']==d]) for key in ('acquisition','query','target','full')} for d in groups},
 'summary':summary,'per_seed':summaries,
 'interpretation':[
  'Two lexical and two ARC examples per batch balances example counts, not direct-loss coefficient mass. ARC targets are longer and receive most query-objective coefficient mass.',
  'Changing full to query is a bundled intervention: removing direct prefix losses, increasing all future-target coefficients, and shifting relative domain/episode weights. Any observed objective effect cannot uniquely isolate learning to write acquisition state.',
  'Lexical future Y bytes receive larger direct coefficients in query than full despite a smaller total lexical domain share; it would be incorrect to infer simple lexical target starvation from overall domain shares.',
  'These are coefficients determined by FIT lengths and the frozen schedule, not measured per-domain losses, gradient norms, optimizer updates or causes of any eventual outcome. AdamW, gradient clipping and gradient interactions preclude translating mass percentages into learning percentages.',
  'The same schedule applies to every model family/objective; this is an interpretation limitation of the registered objective contrast, not evidence of cross-arm data/order cheating.',
  'Shorter and longer ARC episodes receive unequal per-example weight; source/task-clustered exact-match evaluation is a different estimand from token-weighted training CE.',
  'Both objectives keep acquisition within the differentiable graph; query-only training does not add a gradient path absent from the full objective.',
  'No change to the frozen objectives, data, selection, thresholds or scorer is proposed for this trial. A future separately preregistered study could compare domain/example-balanced losses on the same observed episodes.'
 ]}
with OUT.open('x') as handle:json.dump(audit,handle,indent=2,sort_keys=True,allow_nan=False);handle.write('\n')
print(json.dumps({'path':str(OUT),'bytes':OUT.stat().st_size,'summary':summary,
 'per_seed':[{'seed':s['seed'],'full_targets':s['full_supervised_targets'],'query_targets':s['query_supervised_targets'],
 'arc_full_share':s['mean_per_step_loss_coefficient_mass']['full']['arc']['value'],'arc_query_share':s['mean_per_step_loss_coefficient_mass']['query']['arc']['value']} for s in summaries]},indent=2))
