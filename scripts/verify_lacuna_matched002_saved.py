#!/usr/bin/env python3
"""Validate original002 summaries and prefix-selected cohort; no model inference."""
import argparse,hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def verify(path):
 if hashlib.sha256(path.read_bytes()).hexdigest()!='df59a1a9fc7baa67fa5c48d419f8d6db5b0ca9836efe924bd552146adc134f61':raise ValueError('original002ZIP hash')
 with zipfile.ZipFile(path) as z:r=json.loads(z.read('RESULTS.json'))
 docs={};matched={};measured=[x for x in r['rows'] if x['status']=='MEASURED']
 for x in r['availability']:
  owner=x['owner']
  if owner not in ('Doc/tutorial/appendix.rst','Doc/tutorial/classes.rst'):raise ValueError('unregistered owner')
  if owner not in docs:docs[owner]=(ROOT/'research/source-bundles/LANGUAGE-SOURCE023'/owner).read_bytes()
  raw=docs[owner];t=x['clock'];lo=x['bin_begin'];hi=x['bin_end_exclusive'];ctx=raw[t-3:t+1]
  best=max((j for j in range(max(4,t-hi+1),t-lo+1) if raw[j-4:j]==ctx),default=None)
  if x['available']!=(best is not None):raise ValueError('prefix cohort availability changed')
  matched[(owner,t,lo)]=best
 for x in measured:
  raw=docs[x['owner']];j=x['historical_clock'];b=x['wrong_clock'];t=x['clock']
  if j!=matched[(x['owner'],t,x['bin_begin'])] or x['historical_continuation_matches_nextbyte']!=int(raw[j]==raw[t+1]):raise ValueError('cohort or historical reference label')
  if not 5<=abs(b-j)<=64 or b-4<=j<b or b>=t or raw[b-4:b]==raw[t-3:t+1]:raise ValueError('contaminated wrongcontext')
 for a in r['summary']:
  rr=[x for x in measured if x['bin_begin']==a['bin_begin']]
  if len(rr)!=a['measurement_count'] or len({(x['owner'],x['clock']) for x in rr})!=a['distinct_source_query_cells']:raise ValueError('coverage count')
  for k,v in a['metrics'].items():
   if abs(v-sum(x[k] for x in rr)/len(rr))>1e-12:raise ValueError('summary mismatch')
 if len(measured)!=570 or r['new_training_steps'] or r['DEV_scoring'] or r['protected_TEST'] or r['provisional_positive_bins']:raise ValueError('original outcome/boundary')
 return {'status':'PASS_ORIGINAL002_SUMMARIES_AND_PREFIX_COHORT','measurement_rows':570,'source_query_bin_cells':190,'new_model_executions':0,'new_training_steps':0,'provisional_positive_bins':[]}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,default=ROOT/'research/evidence/lacuna_matched002/ORIGINAL_ARTIFACT.zip');a=p.parse_args();print(json.dumps(verify(a.archive),indent=2))
