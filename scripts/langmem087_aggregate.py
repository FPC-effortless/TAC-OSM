#!/usr/bin/env python3
"""Read-only aggregation of9original evaluations, no model fit or source text read."""
import argparse,json,hashlib
from pathlib import Path
LESIONS=('fast_zero','slow_zero','both_zero','permuted_read','old_slow_zero','old_slow_permuted','current_slow_write_zero')
def aggregate(directory):
 paths=sorted(directory.rglob('EVAL_RECEIPT.json'))
 if len(paths)!=9:raise ValueError('all9original checkpoint evaluations required')
 records=[];fits=[]
 for path in paths:
  r=json.loads(path.read_text());fit=json.loads((path.parent/'TRAINED/FIT_RECEIPT.json').read_text())
  if r['study']!='LANGMEM087' or r['seed']!=fit['seed'] or r['trained_steps']!=fit['trained_steps'] or r['checkpoint_sha256']!=fit['checkpoint_sha256'] or r['trained_model_hash']!=fit['trained_model_hash']:raise ValueError('source model identity')
  if not fit['cold_replay_bitwise'] or fit['unseen_DEV_accessed_before_seal'] is not False or not r['original_graph_default_bitwise_parity'] or not r['weights_unchanged_during_evaluation'] or not r['all_three_durations_fitted_before_workflow_DEV_acquisition']:raise ValueError('replay/causal gate')
  if r['model_counts']['trainable']!=100521 or r['model_counts']['deployed']!=100521 or r['nextbyte_targets_per_checkpoint']!=3584 or len(r['rows'])!=3584 or r['persistent_tensor_bytes_per_owner']!=1920:raise ValueError('model/cohort budget')
  if r['actual_source_future_poison_pairs']!=2 or r['original_prefix_max_abs_error']>3e-5 or r['oldslow_split_max_abs_error']>1e-6 or r['evaluator_old_keys_fed_to_model'] or r['gpu_minutes'] or r['protected_TEST']!='NOT_ACCESSED':raise ValueError('actual source diagnostic gate')
  h=hashlib.sha256()
  for x in r['rows']:h.update(x['owner'].encode()+b':'+str(x['clock']).encode()+b':'+bytes([x['actual_next_byte']])+b';')
  if h.hexdigest()!=r['evaluation_nextbyte_target_sha256']:raise ValueError('row target checksum')
  for key,value in r['metrics'].items():
   if key.endswith('_minus_full_nll'):continue
   if abs(sum(x[key] for x in r['rows'])/3584-value)>1e-10:raise ValueError('metric recomputation')
  records.append(r);fits.append(fit)
 if sorted((r['seed'],r['trained_steps']) for r in records)!=[(s,n) for s in (1701,1702,1703) for n in (160,640,1280)]:raise ValueError('complete seed/duration grid')
 for field in ('evaluation_nextbyte_target_sha256','fit_source_sha256','model_source_sha256','diagnostic_source_sha256'):
  if len({r[field] for r in records})!=1:raise ValueError('source/cohort identity')
 if len({json.dumps(r['new_document_hashes'],sort_keys=True) for r in records})!=1:raise ValueError('new publisher source equality')
 by_duration=[];owner_rows=[];strata=[]
 for n in (160,640,1280):
  group=[r for r in records if r['trained_steps']==n];metrics={k:sum(r['metrics'][k] for r in group)/3 for k in group[0]['metrics']}
  by_duration.append({'updates':n,'targets_seen_per_seed':n*1024,'metrics':metrics,'query_proxy':[r['autonomous_query_metrics'] for r in group]})
  for owner in sorted({x['owner'] for x in group[0]['rows']}):
   for r in group:
    rows=[x for x in r['rows'] if x['owner']==owner];m={k:sum(x[k] for x in rows)/len(rows) for k in ('full_nll','slow_zero_nll','old_slow_zero_nll','old_slow_permuted_nll','current_slow_write_zero_nll')}
    owner_rows.append({'updates':n,'seed':r['seed'],'owner':owner,'targets':len(rows),'metrics':m})
  for lo,hi in ((256,512),(512,1024),(1024,2048)):
   rows=[x for r in group for x in r['rows'] if lo<=x['clock']<hi];strata.append({'updates':n,'source_clock_begin':lo,'source_clock_end_exclusive':hi,'NOT_historical_age':True,'full_nll':sum(x['full_nll'] for x in rows)/len(rows),'old_slow_zero_minus_full_nll':sum(x['old_slow_zero_nll']-x['full_nll'] for x in rows)/len(rows)})
 paired=[]
 for seed in (1701,1702,1703):
  rs={r['trained_steps']:r for r in records if r['seed']==seed}
  for n in (640,1280):paired.append({'seed':seed,'updates':n,'full_nll_minus160':rs[n]['metrics']['full_nll']-rs[160]['metrics']['full_nll'],'old_slow_lesion_delta_change_from160':rs[n]['metrics']['old_slow_zero_minus_full_nll']-rs[160]['metrics']['old_slow_zero_minus_full_nll']})
 return {'study':'LANGMEM087','stage':'A_duration_only','independently_qualified':False,'seed_count':3,'checkpoint_count':9,'distinct_new_source_positions':3584,'same_source_positions_across_durations_and_seeds':True,'new_document_hashes':records[0]['new_document_hashes'],'by_duration':by_duration,'per_seed':[{k:r[k] for k in ('seed','trained_steps','checkpoint_sha256','metrics','autonomous_query_metrics','model_full_inference_wall_seconds','model_observed_steps','ring_query_wall_seconds','ring_queries','peak_RSS_bytes','oldslow_split_max_abs_error','original_prefix_max_abs_error')} for r in records],'by_owner':owner_rows,'by_stream_clock':strata,'paired_duration':paired,'fit_receipts':fits,'gpu_minutes':0,'protected_TEST':'NOT_ACCESSED','limits':['Duration contrast deliberately changes training exposure; no trained architecture superiority','Oldslow lesion is immediate one-query read attribution with other history channels fixed; not particular distant byte or semantic recall','Only two new Rust owners, shared source targets acrossseed/duration; no iid byte significance','Bias/context/write-cap/dropout/auxiliary supervision/read-before-write stages not executed','Full graph IEEE/AD and independent qualification OPEN']}
def main():
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError('original aggregate exists')
 r=aggregate(a.results);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps(r,indent=2,sort_keys=True))
if __name__=='__main__':main()
