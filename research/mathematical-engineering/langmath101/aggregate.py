"""Aggregate every preregistered LANGMATH101 fit; no checkpoint selection.

Consumes target-scored DEV records, never a judge, target archive or checkpoint.
Paired case effects average model seeds BEFORE source-cluster resampling.
All inference is exploratory on previously exposed DEV; no protected replication.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
CORE=('carry','reset','donor','direct_history','query_only')
SELECTIVE=('reset_preserve_clock','context_reset','lexical_reset','hidden_reset','fast_reset','slow_reset','banks_reset','top_level_reset','slot0_reset','upward_off','downward_off','color_relabel','code0','code1','force_byte','force_discrete','force_latent')
DRAWS=10000

def read(path):return json.loads(Path(path).read_text())
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def key(row):return (row.get('lane'),row.get('architecture',row.get('arm')),row.get('seed'))
def metric_module():
 spec=importlib.util.spec_from_file_location('_langmath101_aggregate_metrics',HERE/'metrics.py')
 mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod);return mod

def holm(rows):
 """Holm step-down adjusted two-sided p values within declared outcome families."""
 families=defaultdict(list)
 for row in rows:
  if row.get('p_two_sided') is not None:families[row['family']].append(row)
 for family,rr in families.items():
  ordered=sorted(rr,key=lambda r:r['p_two_sided']);running=0.
  for rank,row in enumerate(ordered):
   running=max(running,min(1.,(len(ordered)-rank)*row['p_two_sided']))
   row.update(p_holm=running,holm_family_size=len(ordered),holm_family=family,
              positive_effect_after_holm=bool(running<.05 and row['equal_cluster_mean']>0 and row['CI95_exploratory'][0]>0))
 return rows

def paired(left,right,metric,sign,met):
 """Require the same complete case cohort and all paired seeds; do not drop cases."""
 seeds=sorted(left)
 if seeds!=sorted(right) or not seeds:return {'status':'INCONCLUSIVE_MISSING_PAIRED_SEEDS'}
 ids=set(left[seeds[0]])
 if not ids or any(set(left[s])!=ids or set(right[s])!=ids for s in seeds):return {'status':'INCONCLUSIVE_COHORT_MISMATCH'}
 values=[];clusters=[];seed_effects={str(s):[] for s in seeds}
 for case in sorted(ids):
  rows=[(left[s][case],right[s][case]) for s in seeds]
  if any(a.get('status','MEASURED')!='MEASURED' or b.get('status','MEASURED')!='MEASURED' or metric not in a or metric not in b for a,b in rows):
   return {'status':'INCONCLUSIVE_FULL_COHORT_UNAVAILABLE','original_cases':len(ids),'paired_seeds':seeds}
  cc={str(r.get('cluster',r.get('file'))) for pair in rows for r in pair}
  if len(cc)!=1:return {'status':'INCONCLUSIVE_CLUSTER_IDENTITY_MISMATCH'}
  effects=[sign*(float(a[metric])-float(b[metric])) for a,b in rows]
  if not np.isfinite(effects).all():raise ValueError('nonfinite paired scored metric')
  values.append(float(np.mean(effects)));clusters.append(next(iter(cc)))
  for s,e in zip(seeds,effects):seed_effects[str(s)].append(e)
 result=met.paired_cluster_effect(values,clusters,seed=101001,draws=DRAWS)
 result.update(status='MEASURED_EXPLORATORY',original_cases=len(ids),paired_seeds=seeds,
  per_seed_case_mean={s:float(np.mean(v)) for s,v in seed_effects.items()},
  seed_pooling='paired seed effects averaged within original case before resampling source clusters',
  benefit_direction='positive favors the named left condition/model',p_two_sided=None,p_holm=None)
 if result['clusters']<2:
  result['status']='INCONCLUSIVE_FEWER_THAN_TWO_SOURCE_CLUSTERS';return result
 # Paired source-cluster sign randomization provides the two-sided p value.
 # Enumerating small cohorts prevents bootstrap-degenerate certainty when only
 # a handful of source clusters exist. Cluster exchangeability is an assumption.
 names=sorted(set(clusters));v=np.asarray(values)
 means=np.asarray([v[np.asarray([c==n for c in clusters])].mean() for n in names])
 if len(means)<=16:
  bits=np.arange(2**len(means),dtype=np.uint64)[:,None]
  signs=2*((bits>>np.arange(len(means),dtype=np.uint64)[None])&1).astype(float)-1
  null=(signs*means[None]).mean(1)
  result['p_two_sided']=float(np.mean(np.abs(null)>=abs(means.mean())-1e-15))
  result['p_randomizations']=len(null)
 else:
  rng=np.random.default_rng(101001);signs=rng.choice([-1.,1.],size=(DRAWS,len(means)))
  null=(signs*means[None]).mean(1)
  result['p_two_sided']=float((1+np.count_nonzero(np.abs(null)>=abs(means.mean())-1e-15))/(DRAWS+1))
  result['p_randomizations']=DRAWS
 result['p_method']='two-sided paired source-cluster sign randomization; exact through16clusters, otherwise10000draws; exchangeability assumed; exposed DEV exploratory'
 return result

def language_rows(results,directory):
 rows=results.get('language_windows')
 if rows is None and (directory/'LANGUAGE_SCORED.json').exists():rows=read(directory/'LANGUAGE_SCORED.json')
 return rows or []

def load_evaluation(path):
 directory=path.parent;r=read(path);seal=read(directory/'PREDICTIONS_SEALED.json');scores=read(directory/'SCORED_EPISODES.json')
 meta=seal['checkpoint_metadata'];identity=(meta['lane'],meta['arm'],meta['seed'])
 if r.get('arm')!=identity[1] or r.get('seed')!=identity[2] or r.get('lane',identity[0])!=identity[0]:raise ValueError('result/checkpoint cohort mismatch: '+str(path))
 if r['checkpoint_sha256']!=seal['checkpoint_sha256']:raise ValueError('checkpoint digest mismatch')
 if seal.get('before_target_scoring') is not True or seal.get('evaluation_optimizer_steps')!=0 or seal.get('model_digest_before')!=seal.get('model_digest_after') or seal.get('protected_TEST_access') is not False:raise ValueError('unqualified prediction provenance')
 for name,sha in seal['raw_prediction_files'].items():
  if digest(directory/name)!=sha:raise ValueError('sealed prediction bytes differ: '+name)
 if meta.get('stage')!='SEALED_BEFORE_DEV':raise ValueError('final preregistered checkpoint required')
 conditions={c['condition'] for c in seal['conditions']}
 if set(CORE+SELECTIVE)!=conditions:raise ValueError('complete registered intervention matrix required')
 by_condition=defaultdict(dict)
 for row in scores:
  c=row['condition'];i=row['id']
  if i in by_condition[c]:raise ValueError('duplicate scored episode')
  by_condition[c][i]=row
 if set(by_condition)!=conditions:raise ValueError('scored/sealed condition mismatch')
 original_ids=set(by_condition['carry'])
 if any(set(v)!=original_ids for v in by_condition.values()):raise ValueError('condition omitted originally selected cases')
 lr=language_rows(r,directory)
 if not lr:raise ValueError('per-window source-scored language rows required')
 lang={}
 sealed_ids={x['id'] for x in seal['language']['records']}
 for row in lr:
  if row['id'] in lang:raise ValueError('duplicate language case')
  lang[row['id']]={**row,'cluster':row['file'],'status':'MEASURED'}
 if set(lang)!=sealed_ids:raise ValueError('language scored/sealed cohort mismatch')
 return identity,{'results':r,'seal':seal,'episodes':dict(by_condition),'language':lang,'directory':str(directory)}

def subset(rows,domain=None,group=None,modality=None):
 return {i:r for i,r in rows.items() if (domain is None or r.get('domain')==domain) and (group is None or r.get('group')==group) and (modality is None or r.get('modality')==modality)}

def episode_summary(rows):
 measured=[r for r in rows.values() if r.get('status')=='MEASURED'];complete=len(measured)==len(rows)
 return {'original_cases':len(rows),'measured_cases':len(measured),'status':'MEASURED' if complete else 'INCONCLUSIVE_FULL_COHORT',
  'exact_successes':sum(r['exact_success'] for r in measured),'exact_rate':float(np.mean([r['exact_success'] for r in measured])) if complete and rows else None,
  'first_byte_NLL':float(np.mean([r['first_byte_NLL'] for r in measured])) if complete and rows else None}

def flatten(record):
 return {k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list,tuple)) else v for k,v in record.items()}
def write_csv(path,rows):
 keys=sorted({k for row in rows for k in row})
 with path.open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(flatten(r) for r in rows)

def aggregate(args):
 if args.output.exists():raise FileExistsError('retain original aggregate outputs')
 manifest=read(args.manifest);planned=manifest['arms'];expected={key(r) for r in planned}
 if len(expected)!=len(planned):raise ValueError('duplicate planned cohort')
 cohorts={};problems=[];extra=[]
 for path in sorted(args.evaluations.rglob('RESULTS.json')):
  try:
   identity,value=load_evaluation(path)
   if identity not in expected:extra.append({'cohort':identity,'path':str(path)});continue
   if identity in cohorts:raise ValueError('duplicate evaluations for planned cohort '+str(identity))
   cohorts[identity]=value
  except Exception as e:problems.append({'path':str(path),'error':type(e).__name__,'message':str(e)})
 missing=sorted(expected-set(cohorts));lanes=sorted({k[0] for k in expected});arms=sorted({k[1] for k in expected});seeds=sorted({k[2] for k in expected})
 # Source identities must agree across every model, lane and repeated seed.
 identities={json.dumps(v['seal']['source_identity'],sort_keys=True) for v in cohorts.values()}
 if len(identities)>1:problems.append({'error':'SourceIdentityMismatch','message':'evaluations do not share an identical original DEV source cohort'})
 met=metric_module();allseed=[];episode_table=[];comparisons=[];cost_table=[]
 for row in planned:
  k=key(row);base={'lane':k[0],'arm':k[1],'seed':k[2]};v=cohorts.get(k)
  if v is None:allseed.append({**base,'status':'MISSING_OR_INVALID_EVALUATION'});continue
  for record in v['results']['language']:allseed.append({**base,'status':'MEASURED','checkpoint_sha256':v['seal']['checkpoint_sha256'],**record})
  for condition,rr in v['episodes'].items():
   for domain in ('lexical','arc'):
    for group in [None]+sorted({r['group'] for r in rr.values() if r['domain']==domain}):
     episode_table.append({**base,'condition':condition,'domain':domain,'group':group or 'ALL',**episode_summary(subset(rr,domain,group))})
  for condition in v['seal']['conditions']:
   cc=condition['costs'];cost_table.append({**base,'condition':condition['condition'],'measured_cases':len(cc),
    'model_seconds':sum(c.get('model_seconds',0) for c in cc),'evaluation_wall_seconds':condition['seconds'],
    'maximum_tensor_state_bytes':max((c.get('tensor_bytes',0) for c in cc),default=0),
    'maximum_raw_prefix_bytes':max((c.get('non_tensor_raw_prefix_bytes',0) for c in cc),default=0),
    'maximum_cached_token_ID_count':max((c.get('non_tensor_cached_token_ID_count',0) for c in cc),default=0),
    'recurrent_updates':sum(c.get('recurrent_updates',0) for c in cc),
    'candidate_recurrent_calls':sum(c.get('candidate_recurrent_calls',0) for c in cc),
    'not_full_FLOPs':True,'container_overhead':'NOT_MEASURED'})
 def comparison(lane,left,right,lc='carry',rc='carry',kind='architecture'):
  relevant=[(lane,a,s) for a in (left,right) for s in seeds]
  common={'lane':lane,'kind':kind,'left':left,'right':right,'left_condition':lc,'right_condition':rc}
  if len(identities)!=1:
   comparisons.append({**common,'family':kind+'.availability','status':'INCONCLUSIVE_SOURCE_IDENTITY_MISMATCH'});return
  if any(k not in cohorts for k in relevant):
   comparisons.append({**common,'family':kind+'.availability','status':'INCONCLUSIVE_MISSING_PLANNED_COHORT'});return
  if kind!='intervention':
   modalities=sorted({r.get('modality') for s in seeds for r in cohorts[lane,left,s]['language'].values()})
   for modality in [None]+modalities:
    ll={s:subset(cohorts[lane,left,s]['language'],modality=modality) for s in seeds};rr={s:subset(cohorts[lane,right,s]['language'],modality=modality) for s in seeds}
    comparisons.append({**common,'domain':'language','group':modality or 'ALL','metric':'NLL_nats_per_byte','family':kind+'.language.NLL',**paired(ll,rr,'NLL_nats_per_byte',-1,met)})
  for domain in ('lexical','arc'):
   groups=sorted({r['group'] for s in seeds for r in cohorts[lane,left,s]['episodes'][lc].values() if r['domain']==domain})
   for group in [None]+groups:
    ll={s:subset(cohorts[lane,left,s]['episodes'][lc],domain,group) for s in seeds};rr={s:subset(cohorts[lane,right,s]['episodes'][rc],domain,group) for s in seeds}
    for metric,sign in [('first_byte_NLL',-1),('exact_success',1)]:
     comparisons.append({**common,'domain':domain,'group':group or 'ALL','metric':metric,'family':kind+'.'+domain+'.'+metric,**paired(ll,rr,metric,sign,met)})
 for lane in lanes:
  for arm in arms:
   if arm!='native098':comparison(lane,arm,'native098')
   for condition in CORE[1:]+SELECTIVE:comparison(lane,arm,arm,rc=condition,kind='intervention')
  for depth in (1,2,3,4,8):
   a,b=f'stack_assoc_{depth}',f'stack_recurrent_{depth}'
   if a in arms and b in arms:comparison(lane,a,b,kind='depth_associative_control')
  for kind in ('assoc','recurrent'):
   h,s=f'hierarchy_{kind}_3',f'stack_{kind}_3'
   if h in arms and s in arms:comparison(lane,h,s,kind='hierarchy_depth3_control')
 holm(comparisons)
 fulltask_wins=[r for r in comparisons if r.get('kind')=='architecture' and r.get('metric')=='exact_success' and r.get('group')=='ALL' and r.get('positive_effect_after_holm') and all(v>0 for v in r.get('per_seed_case_mean',{}).values())]
 complete=not missing and not problems and not extra and len(identities)==1
 stages={
 '001':{'status':'COMPLETE_INTERNAL_DEV' if complete else 'INCOMPLETE','scope':'all planned final-checkpoint cohorts; original10M attribution retained separately'},
 '002':{'status':'MEASURED_EXPLORATORY' if complete else 'INCOMPLETE','scope':'full paired state and bank boundary interventions; unavailable dimensions retained'},
 '003':{'status':'BOUNDED_INTERNAL_COMPARISON' if complete else 'INCOMPLETE','limit':'No authenticated strict hidden-history necessity benchmark; memory necessity is unestablished'},
 '004':{'status':'BOUNDED_INTERNAL_COMPARISON' if complete else 'INCOMPLETE','depths':[1,2,4,8],'additional_depth3_control':True},
 '005':{'status':'BOUNDED_INTERNAL_COMPARISON' if complete else 'INCOMPLETE','limit':'Architectural temporal priors and causal paths do not establish learned conceptual abstraction'},
 '006':{'status':'BOUNDED_INTERNAL_COMPARISON' if complete else 'INCOMPLETE','limit':'Byte, FIT-only BPE, discrete, latent and adaptive arms; surrogate compute charged; no full FLOP advantage'},
 '007':{'status':'UNESTABLISHED','reason':'Genuine unseen rule-family and novel-composition labels absent; exact lexical/ARC and surface controls cannot establish structure discovery'},
 '008':{'status':'MEASURED_PARTIAL_COSTS' if complete else 'INCOMPLETE','full_model_FLOPs':'INCONCLUSIVE_NOT_CLAIMED','energy':'NOT_MEASURED','production_decode_superiority':'NOT_ESTABLISHED'},
 '009':{'status':'BLOCKED_EXTERNAL_ASSESSOR','reason':'No independent external assessor or protected TEST; collaborating agents and internal replays are not independent replication'},
 '010':{'status':'CONDITIONAL_COST_AND_REPLICATION_REQUIRED' if fulltask_wins and complete else 'NO_INTEGRATION_GATE_ESTABLISHED','statistical_complete_task_wins':len(fulltask_wins),'integrated_model_constructed':False,'reason':'Requires reproducible Holm-supported complete-task wins plus prospectively acceptable measured costs; NLL alone never opens integration'}}
 report={'schema':'langmath101.aggregate.v1','status':'COMPLETE_INTERNAL_EXPOSED_DEV' if complete else 'INCOMPLETE_NO_COHORT_SELECTION',
  'manifest_sha256':digest(args.manifest),'planned_cohorts':len(expected),'valid_evaluations':len(cohorts),'missing_cohorts':missing,'invalid_inputs':problems,'unplanned_inputs':extra,
  'allseed_language':allseed,'allseed_episodic':episode_table,'paired_comparisons':comparisons,'recorded_evaluation_costs':cost_table,'stages':stages,
  'statistics':{'bootstrap_draws':DRAWS,'clustering':'language source file; original lexical source cluster; original ARC task','seed_pooling':'same original case averaged over all planned seeds, not independent seed copies','multiplicity':'Holm within comparison-kind/domain/metric family across all lanes and groups','CI':'unadjusted exploratory95%; p_holm controls declared family','qualification':'previously exposed public DEV only; not independent or protected'},
  'claim_limits':['No architecture superiority claim','No simultaneous parameter/data/full-compute parity claim','Counted dense arithmetic is not complete training FLOPs','No best seed/checkpoint/architecture selection','No universal IEEE/AD or learned-capability proof']}
 args.output.mkdir(parents=True)
 with (args.output/'REPORT.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
 write_csv(args.output/'ALL_SEEDS_LANGUAGE.csv',allseed);write_csv(args.output/'ALL_SEEDS_EPISODIC.csv',episode_table);write_csv(args.output/'PAIRED_COMPARISONS.csv',comparisons);write_csv(args.output/'RECORDED_COSTS.csv',cost_table)
 lines=['# LANGMATH101 internal DEV research report','',report['status'],'',f'Valid final-checkpoint evaluations: {len(cohorts)}/{len(expected)}. All planned cohorts are retained; no best-run selection.','',f'Paired comparisons use {DRAWS:,} source-cluster bootstrap draws after averaging repeated seeds within each original case. Holm corrections use the declared comparison-kind/domain/outcome families. Confidence intervals remain exploratory.','','| Stage | Status |','|---|---|']
 lines += [f'| {s} | {v["status"]} |' for s,v in stages.items()]
 lines += ['','All seed scores, unavailable cohorts, paired effects and recorded costs are in the companion CSV and JSON files.','',f'Complete-task architecture comparisons meeting exploratory corrected statistical criteria: {len(fulltask_wins)}. Integration additionally requires acceptable prospective costs and replication; no integrated model is created.','','The data lack a genuine unseen rule-family/composition evaluation. Independent protected replication remains blocked by the absent external assessor. No production decoding or full FLOP superiority follows from these measurements.']
 (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'status':report['status'],'planned':len(expected),'valid':len(cohorts),'output':str(args.output)}))
 return 0 if complete else 1

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--evaluations',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 raise SystemExit(aggregate(p.parse_args()))
