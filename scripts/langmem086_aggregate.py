#!/usr/bin/env python3
"""Require full successful training and original sealed first-evaluation receipts."""
import argparse,json
from pathlib import Path

def aggregate(directory):
    paths=sorted(directory.rglob('EVAL_RECEIPT.json'))
    if len(paths)!=3:raise ValueError('three completed original fits/evaluations required')
    records=[];fits=[]
    for path in paths:
        r=json.loads(path.read_text());fit=json.loads((path.parent/'TRAINED/FIT_RECEIPT.json').read_text())
        if r['study']!='LANGMEM086' or fit['study']!='LANGMEM086' or r['seed']!=fit['seed']:raise ValueError('source original identities')
        if r['checkpoint_sha256']!=fit['checkpoint_sha256'] or r['trained_model_hash']!=fit['trained_model_hash']:raise ValueError('sealed actual model')
        if fit['trained_steps']!=160 or not fit['cold_replay_bitwise'] or fit['unseen_DEV_accessed_before_seal'] is not False:raise ValueError('FIT before DEV and replay')
        if r['model_counts']['trainable']!=100521 or r['model_counts']['deployed']!=100521:raise ValueError('actual100k count')
        if not r['weights_unchanged_during_evaluation'] or r['evaluator_old_keys_fed_to_model'] or not r['model_receives_only_observed_byte_and_own_state']:raise ValueError('autonomous interface')
        if r['nextbyte_targets_per_seed']!=1792 or len(r['rows'])!=1792 or r['persistent_tensor_bytes_per_owner']!=1920:raise ValueError('real source targets/state')
        if r['actual_source_future_poison_pairs']!=2 or r['original_prefix_max_abs_error']>3e-5 or r['gpu_minutes'] or r['protected_TEST']!='NOT_ACCESSED':raise ValueError('real source gates')
        records.append(r);fits.append(fit)
    if sorted(r['seed'] for r in records)!=[1701,1702,1703] or len({r['checkpoint_sha256'] for r in records})!=3:raise ValueError('three original learned seeds')
    for field in ('evaluation_nextbyte_target_sha256','fit_source_sha256'):
        if len({r[field] for r in records})!=1:raise ValueError('same real source')
    if len({json.dumps(r['new_document_hashes'],sort_keys=True) for r in records})!=1:raise ValueError('same new publisher bytes')
    if len({tuple((x['owner'],x['clock'],x['actual_next_byte']) for x in r['rows']) for r in records})!=1:raise ValueError('same actual target rows')
    metrics={k:sum(r['metrics'][k] for r in records)/3 for k in records[0]['metrics']}
    for lesion in ('fast_zero','slow_zero','both_zero','permuted_read'):metrics[lesion+'_minus_full_nll']=metrics[lesion+'_nll']-metrics['full_nll']
    return {'study':'LANGMEM086','seed_count':3,'new_source_nextbyte_targets_per_seed':1792,'same_targets_across_seeds':True,
        'evaluation_nextbyte_target_sha256':records[0]['evaluation_nextbyte_target_sha256'],'new_document_hashes':records[0]['new_document_hashes'],
        'trained_parameters_per_seed':100521,'state_tensor_bytes_per_owner':1920,'shared_fixed_buffer_bytes':8192,'clock_bytes':8,
        'all_original_models_trained_sealed_before_each_new_DEV_read':True,'evaluator_key_fed_to_model':False,
        'metrics':metrics,'per_seed':[{k:r[k] for k in ('seed','checkpoint_sha256','metrics','autonomous_query_metrics','original_prefix_max_abs_error','evaluation_cpu_seconds','model_full_inference_wall_seconds','ring_query_wall_seconds','peak_evaluation_RSS_bytes')} for r in records],
        'fit_receipts':[{k:r[k] for k in ('seed','trained_steps','targets_seen','sample_receipt','model_counts','fit_cpu_seconds','peak_RSS_bytes','fit_loss_first','fit_loss_last','cold_replay_bitwise','actual_FIT_future_poison_before_first_update')} for r in fits],
        'gpu_minutes':0,'protected_TEST':'NOT_ACCESSED','independently_qualified':False,'limits':records[0]['limits']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable result exists')
    r=aggregate(a.results);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps(r,indent=2,sort_keys=True))
if __name__=='__main__':main()
