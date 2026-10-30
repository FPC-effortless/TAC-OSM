#!/usr/bin/env python3
"""Read-only cold-checkout verification of all 25 original fitted models.

Never score missing arms selectively. Recompute new publisher DEV next-byte
log probabilities from exact original checkpoint bytes without fitting.
"""
from __future__ import annotations
import argparse,hashlib,json,math,statistics,sys
from pathlib import Path
import torch
REG=Path(__file__).resolve().parent
sys.path.insert(0,str(REG))
from model094 import ARMS,SEEDS,PARAMS,make_model
from run094 import STEPS,BATCH,LENGTH,EVAL_START,EVAL_STOP,DEV_BLOBS,source_hash,read_new_dev,state_hash,evaluate,counts
def digest(raw):return hashlib.sha256(raw).hexdigest()
def verify(directory):
 torch.set_num_threads(2)
 docs=read_new_dev(directory/'DEV_SOURCE')
 allrows=[];sched_by_seed={}
 for seed in SEEDS:
  for arm in ARMS:
   folder=directory/f'{arm}-{seed}'
   cp_path=folder/'MODEL.pt';result_path=folder/'RESULTS.json'
   if not cp_path.is_file() or not result_path.is_file():raise ValueError('missing original fitted evidence '+str(folder))
   result=json.loads(result_path.read_text());raw=cp_path.read_bytes()
   if result['checkpoint_sha256']!=digest(raw):raise ValueError('checkpoint hash '+str(folder))
   cp=torch.load(cp_path,map_location='cpu',weights_only=True)
   if cp['study']!='LANGHEAD094' or result['study']!='LANGHEAD094' or cp['seed']!=seed or cp['arm']!=arm or result['seed']!=seed or result['arm']!=arm or cp['steps']!=STEPS:
    raise ValueError('original arm/seed/step mismatch '+str(folder))
   if cp['model_source_sha256']!=source_hash() or result['source_code_sha256']!=source_hash():
    raise ValueError('original model source checksum mismatch')
   if cp['source_sha256']!='911981cc63fd5436e0495cb067784e53475971131c811b7026e07d97a886d2d4':
    raise ValueError('original CPython FIT source mismatch')
   sched=cp['schedule']
   if sched['targets']!=STEPS*BATCH*LENGTH or sched!=result['real_sample_schedule']:raise ValueError('actual schedule not replayable')
   if seed in sched_by_seed and sched!=sched_by_seed[seed]:raise ValueError('within-seed data exposure differs by arm')
   sched_by_seed[seed]=sched
   m=make_model(arm).eval()
   m.load_state_dict(cp['state'],strict=True)
   if counts(m)!={'trainable':PARAMS[arm],'deployed':PARAMS[arm]} or counts(m)!=result['counts']:
    raise ValueError('original parameter count mismatch')
   if state_hash(m)!=result['weights_sha256']:raise ValueError('trained original weights hash mismatch')
   states=cp['optimizer']['state']
   if len(states)!=len(list(m.parameters())) or any(int(t['step'])!=STEPS for t in states.values()):
    raise ValueError('original optimizer parameters/step count mismatch')
   if not result['new_dev_after_checkpoint_seal'] or result['protected_TEST']!='NOT_ACCESSED':
    raise ValueError('evidence scope or order invalid')
   old=sorted(result['dev'],key=lambda a:a['owner'])
   if len(old)!=len(DEV_BLOBS) or {a['owner'] for a in old}!=set(DEV_BLOBS):raise ValueError('missing registered evaluation owners')
   for owner in old:
    if owner['source_git_blob']!=DEV_BLOBS[owner['owner']] or owner['sha256']!=digest(docs[owner['owner']]):
     raise ValueError('actual publisher source identity')
    tokens=owner['tokens']
    if len(tokens)!=EVAL_STOP-EVAL_START or any(t['ordinal']!=EVAL_START+i for i,t in enumerate(tokens)):
     raise ValueError('missing/extra original target rows')
   recomputed,_=evaluate(m,docs)
   for expected,actual in zip(old,sorted(recomputed,key=lambda a:a['owner'])):
    if expected['owner']!=actual['owner'] or expected['sha256']!=actual['sha256']:
     raise ValueError('independent original owner mismatch')
    for x,y in zip(expected['tokens'],actual['tokens']):
     if x['ordinal']!=y['ordinal'] or x['correct']!=y['correct'] or abs(x['nll']-y['nll'])>3e-5:
      raise ValueError('original independent predictor cold replay diverged')
   allrows.append(result)
 paired={}
 per_seed={}
 for row in allrows:
  seed=row['seed'];per_seed.setdefault(seed,{})[row['arm']]=sum(x['mean_nll'] for x in row['dev'])/len(row['dev'])
 for arm in ARMS:
  rows=[r for r in allrows if r['arm']==arm]
  diff=[per_seed[s][arm]-per_seed[s]['original'] for s in SEEDS]
  mu=statistics.mean(diff);sd=statistics.stdev(diff);half=2.776445*sd/math.sqrt(len(SEEDS))
  paired[arm]={
   'mean_nll':statistics.mean(per_seed[s][arm] for s in SEEDS),
   'mean_accuracy':statistics.mean(sum(d['accuracy'] for d in r['dev'])/len(r['dev']) for r in rows),
   'paired_mean_nll_delta_vs_original':mu,'paired_seed_95pct_t_interval':[mu-half,mu+half],
   'seed_deltas':diff,'seed_wins':sum(d<0 for d in diff),
   'mean_training_process_seconds':statistics.mean(r['train_cpu_seconds'] for r in rows),
   'mean_dev_cpu_ms_per_byte':statistics.mean(r['latency']['process_seconds']/r['latency']['targets']*1000 for r in rows),
   'per_owner_mean_nll':{o:statistics.mean(next(z['mean_nll'] for z in r['dev'] if z['owner']==o) for r in rows) for o in DEV_BLOBS}
  }
 return {'study':'LANGHEAD094','status':'INTERNAL_EXPLORATORY_EQUAL_EXPOSURE_UNMATCHED_FLOPs',
  'training_runs':len(allrows),'fit_publisher':'original licensed CPython 3.13','evaluation_publisher':'rust-lang/book pinned October 2026',
  'new_document_owners':list(sorted(docs)),'separate_from_prior_exposed_090_092_docs':True,
  'new_evaluation_owners_count':len(docs),'targets_per_arm':len(SEEDS)*len(docs)*(EVAL_STOP-EVAL_START),
  'training_FLOPs_matched':False,'source_identity':'VERIFIED_PINNED_GIT_BLOB',
  'original_checkpoint_optimizer_and_all_new_dev_token_replay':'PASS',
  'cross_publisher_generalization':'INTERNAL_EXPLORATORY_SINGLE_OTHER_PUBLISHER_NOT_INDEPENDENTLY_QUALIFIED','protected_TEST':'UNOPENED',
  'gpu_minutes':0,'no_architecture_superiority_claim':True,'paired':paired}
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 if a.output.exists():raise FileExistsError('immutable aggregate destination')
 r=verify(a.directory);a.output.parent.mkdir(parents=True,exist_ok=True)
 a.output.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({k:{'NLL':v['mean_nll'],'delta':v['paired_mean_nll_delta_vs_original']} for k,v in r['paired'].items()}))
if __name__=='__main__':main()
