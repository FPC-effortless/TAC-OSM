#!/usr/bin/env python3
"""Recompute saved screen summaries only; no torch, model, source text or network."""
import hashlib,json,zipfile,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def verify(path):
 raw=path.read_bytes()
 if hashlib.sha256(raw).hexdigest()!='e64641737ccb3d4f6b14d6bcc8835d5c7c092a060457346636222be31b002a30':raise ValueError('original artifact ZIP checksum')
 with zipfile.ZipFile(path) as z:r=json.loads(z.read('RESULTS.json'))
 if len(r['events'])!=210 or len(r['policy_rows'])!=5376 or r['new_training_steps']!=0 or r['DEV_scoring'] or r['protected_TEST']:raise ValueError('original cohort/boundary')
 for a in r['age_summary']:
  rr=[x for x in r['events'] if x['age']==a['age']]
  if len(rr)!=42:raise ValueError('same age coverage')
  for k,v in a.items():
   if k not in ('age','events') and abs(sum(x[k] for x in rr)/len(rr)-v)>1e-12:raise ValueError('event summary changed')
 for k,v in r['policy_mean'].items():
  if abs(sum(x[k] for x in r['policy_rows'])/len(r['policy_rows'])-v)>1e-12:raise ValueError('policy summary changed')
 for x in r['events']:
  j=x['historical_clock'];b=x['wrong_clock']
  if not 5<=abs(j-b)<=16 or b-4<=j<b or b>=x['clock']:raise ValueError('wrong event contamination')
 for c in r['source_checks']:
  if c['event_sum_error']>1e-5 or c['independent_PNDS_step_error']>1e-6 or any(c['lookup_calls'][k]!=224 for k in ('random','entropy','survival_proxy')) or c['lookup_calls']['always']!=896:raise ValueError('source/budget gate')
 passed=[]
 for a in r['age_summary']:
  if a['age']<128:continue
  rr=[x for x in r['events'] if x['age']==a['age']]
  means=[sum(x['old_event_removal_minus_full_nll'] for x in rr if x[k]==v)/sum(x[k]==v for x in rr) for k in ('seed','owner') for v in sorted({x[k] for x in rr})]
  if a['old_event_removal_minus_full_nll']>=.001 and a['old_event_removal_minus_full_nll']>a['wrong_normmatched_removal_minus_full_nll'] and min(means)>0:passed.append(a['age'])
 if passed!=r['candidate_fit_gate']['passed_ages']:raise ValueError('scientific gate recomputation')
 return {'status':'PASS_ORIGINAL_ARTIFACT_AND_RECOMPUTED_SUMMARIES','events':210,'distinct_historical_positions':len({(x['owner'],x['historical_clock']) for x in r['events']}),'candidate_fit_gate':r['candidate_fit_gate'],'new_model_executions':0,'new_training_steps':0}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,default=ROOT/'research/evidence/lacuna_survival001/ORIGINAL_ARTIFACT.zip');a=p.parse_args();print(json.dumps(verify(a.archive),indent=2))
