#!/usr/bin/env python3
"""Strict fifteen-original-checkpoint aggregator. Five seeds are the unit of pairing."""
from __future__ import annotations
import argparse,json,math,statistics
from pathlib import Path
from frozen_compare097 import STUDIES,SEEDS,LENGTHS,DOCS,ANCHORS

def gather(root):
 rows={}
 for p in root.rglob('RESULTS.json'):
  r=json.loads(p.read_text())
  if r.get('study')!='LANGPARALLEL097':raise ValueError('nonregistered original score')
  a=r['original']['arm'];s=r['original']['seed']
  if a not in STUDIES or s not in SEEDS or (a,s) in rows:raise ValueError('invalid duplicate or unregistered arm/seed')
  study,tag,nparams,runid=STUDIES[a]
  if (r['original']['source_training_study'],r['original']['training_run_id'],r['original']['training_updates'],r['original']['learned_parameters'])!=(study,runid,320,nparams):
   raise ValueError('original source/parameter contract')
  if r.get('optimizer_steps_in_this_study')!=0 or r.get('gpu_minutes')!=0 or r.get('protected_TEST')!='UNOPENED' or r.get('full_model_streaming') is not False:
   raise ValueError('source/compute/streaming honesty failure')
  if r.get('matched_original_training_FLOPs') is not False or r.get('no_architecture_superiority_claim') is not True:
   raise ValueError('invalid compute/superiority claim')
  if len(r['original']['checkpoint_sha256'])!=64 or len(r['original']['original_sample_order_sha256'])!=64:
   raise ValueError('original checkpoint/train receipt')
  if r['source_commit']!='2771df165f092a4fbb9f46b5b2e7450f19010adb' or {k:v['git_blob'] for k,v in r['source_docs'].items()}!=DOCS:
   raise ValueError('original Rust Book identity')
  for n in LENGTHS:
   t=r['score'][str(n)]
   if t['n']!=len(DOCS)*len(ANCHORS) or set(t['per_document'])!=set(DOCS) or not (math.isfinite(t['mean_nll']) and 0<=t['accuracy']<=1):
    raise ValueError('missing/invalid real target set')
   for doc,d in t['per_document'].items():
    if d['n']!=len(ANCHORS) or d['targets']!=list(ANCHORS) or d['sha256']!=r['source_docs'][doc]['sha256']:
     raise ValueError('unaligned original source labels')
  timing=r['timing']
  if (timing['batch'],timing['window'],len(timing['times_process_seconds']))!=(4,128,5) or not all(t>0 and math.isfinite(t) for t in timing['times_process_seconds']):
   raise ValueError('invalid batch timing')
  rows[a,s]=r
 expected={(a,s) for a in STUDIES for s in SEEDS}
 if set(rows)!=expected:raise ValueError('incomplete original fifteen-checkpoint matrix '+str(sorted(expected-set(rows))))
 for seed in SEEDS:
  if len({rows[a,seed]['original']['original_sample_order_sha256'] for a in STUDIES})!=1:
   raise ValueError('different real FIT sampling within seed')
 for n in LENGTHS:
  if len({json.dumps(rows[a,s]['source_docs'],sort_keys=True) for a,s in rows})!=1:
   raise ValueError('different real source inputs across arms/seeds')
  for doc in DOCS:
   if len({json.dumps(rows[a,s]['score'][str(n)]['per_document'][doc]['targets']) for a,s in rows})!=1:
    raise ValueError('different label positions')
 summary={'study':'LANGPARALLEL097','status':'FROZEN_ORIGINAL_15_CHECKPOINTS_SOURCE_GROUNDED_NO_FIT',
  'historical_model_seeds':len(SEEDS),'trained_models_in_this_study':0,
  'original_checkpoints':{f'{a}-{s}':rows[a,s]['original']['checkpoint_sha256'] for a,s in sorted(rows)},
  'publisher':'rust-lang/book original MIT or Apache-2.0 text',
  'source_docs':rows['flat_parallel',SEEDS[0]]['source_docs'],
  'source_exposure':'PREVIOUSLY_EXPOSED_DEVELOPMENT_NOT_PROTECTED_TEST',
  'per_seed':{},'per_architecture':{},'paired_by_seed':{},
  'parallelism':'Hillis-Steele native memory + wave + concept scan; layered adds affine-scan residual',
  'no_true_carried_state':True,'full_training_FLOPs_matched':False,
  'protected_TEST':'UNOPENED','gpu_minutes':0,'superiority_qualified':False}
 for a in STUDIES:
  summary['per_architecture'][a]={}
  summary['per_seed'][a]={str(s):{'score':rows[a,s]['score'],'timing':rows[a,s]['timing']} for s in SEEDS}
  for n in LENGTHS:
   xs=[rows[a,s]['score'][str(n)]['mean_nll'] for s in SEEDS]
   acc=[rows[a,s]['score'][str(n)]['accuracy'] for s in SEEDS]
   summary['per_architecture'][a][str(n)]={'mean_nll':statistics.mean(xs),'mean_accuracy':statistics.mean(acc),
      'seed_std_nll':statistics.stdev(xs),'target_count_per_seed':len(ANCHORS)*len(DOCS)}
  times=[rows[a,s]['timing']['median_process_ms_per_observed_byte'] for s in SEEDS]
  summary['per_architecture'][a]['cpu_full_window']={'mean_ms_per_byte':statistics.mean(times),'median_ms_per_byte':statistics.median(times),
      'unit':'2-thread hosted CPU batch4 window128 full inference (not incremental stream)','per_seed':dict(zip(map(str,SEEDS),times))}
 for n in LENGTHS:
  summary['paired_by_seed'][str(n)]={}
  for first,second in (('layered_h48_preferred','flat_parallel'),('layered_h48_preferred','layered_h59'),('layered_h59','flat_parallel')):
   delta=[rows[first,seed]['score'][str(n)]['mean_nll']-rows[second,seed]['score'][str(n)]['mean_nll'] for seed in SEEDS]
   mean=statistics.mean(delta);margin=2.776445*statistics.stdev(delta)/math.sqrt(len(SEEDS))
   summary['paired_by_seed'][str(n)][first+'_minus_'+second]={'mean_nll_delta':mean,
      '95pct_descriptive_t_interval':[mean-margin,mean+margin],
      'first_better_seed_count':sum(d<0 for d in delta),'per_seed':dict(zip(map(str,SEEDS),delta))}
 summary['paired_runtime']={}
 for first,second in (('layered_h48_preferred','flat_parallel'),('layered_h48_preferred','layered_h59'),('layered_h59','flat_parallel')):
  ratios=[rows[first,seed]['timing']['median_process_ms_per_observed_byte']/rows[second,seed]['timing']['median_process_ms_per_observed_byte'] for seed in SEEDS]
  summary['paired_runtime'][first+'_over_'+second]={'mean_ratio':statistics.mean(ratios),
    'median_ratio':statistics.median(ratios),'ratios_per_seed':dict(zip(map(str,SEEDS),ratios))}
 return summary
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 if a.output.exists():raise FileExistsError('immutable aggregate')
 m=gather(a.root);a.output.parent.mkdir(parents=True,exist_ok=True)
 a.output.write_text(json.dumps(m,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'study':m['study'],'paired':m['paired_by_seed'],'runtime':m['paired_runtime']}))
if __name__=='__main__':main()
