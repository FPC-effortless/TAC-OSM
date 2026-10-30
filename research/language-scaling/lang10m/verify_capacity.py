"""Audit saved capacity outputs and deterministic FIT windows without model inference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from capacity_profile import SHAPES,PRECISIONS,sample,sha,atomic


def verify(result,tokens,baseline,output):
    r=json.loads(result.read_text());old=json.loads(baseline.read_text())
    if r['status']!='PASS_CAPACITY_MATRIX':raise ValueError('capacity gates did not all pass')
    if r['model_sha256']!=sha(Path(__file__).resolve().parent/'model.py'):raise ValueError('profile/model source mismatch')
    if len(r['physical_T4s'])!=2 or len({p[0] for p in r['physical_T4s']})!=2:raise ValueError('distinct physical pair missing')
    if r['optimizer_steps']!=0 or sha(tokens)!=r['token_arrays_sha256']:raise ValueError('optimizer/data identity mismatch')
    expected={(b,t,p) for b,t in SHAPES for p in PRECISIONS}
    if {(x['batch'],x['context'],x['precision']) for x in r['results']}!=expected or len(r['results'])!=20:raise ValueError('incomplete matrix')
    with np.load(tokens,allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
    before={(x['batch'],x['context'],x['seed']):x for x in old['results'] if x['precision']=='fp32'}
    matched=0;loss_deltas=[];amp_zero=[]
    for row in r['results']:
        if row['status']!='PASS_NO_UPDATE_PROFILE' or row['optimizer_steps']!=0 or row['trainable_parameters']!=10001188:raise ValueError('failed/mismatched case')
        if row['weight_sha256_before']!=row['weight_sha256_after'] or not row['all_gradients_finite']:raise ValueError('weights or gradients changed/invalid')
        _,_,windows=sample(arrays,row['batch'],row['context'],row['seed'])
        if windows!=row['source_windows']:raise ValueError('actual source windows mismatch')
        previous=before[row['batch'],row['context'],row['seed']]
        matched+=int(row['weight_sha256_before']==previous['weight_sha256_before'])
        if row['precision']=='fp32':
            loss_deltas.append(abs(float(np.mean(row['untrained_FIT_loss_numerical_diagnostic_NOT_LANGUAGE_QUALITY']))-
                                   float(np.mean(previous['untrained_FIT_loss_numerical_diagnostic_NOT_LANGUAGE_QUALITY']))))
        elif row['zero_gradient_parameter_tensors_on_fixture']:
            amp_zero.append({k:row[k] for k in ('batch','context','zero_gradient_parameter_tensors_on_fixture')})
    report={'variant':'LANG10M001','verification':'PASS20_ORIGINAL_NO_UPDATE_CASES_AND_SOURCE_WINDOWS',
        'cases_verified':20,'physical_T4_UUIDs':[p[0] for p in r['physical_T4s']],
        'weights_unchanged_cases':20,'original_initial_weight_hash_matches':matched,
        'max_FP32_untrained_diagnostic_loss_change_from_baseline':max(loss_deltas),
        'AMP_cases_with_zero_gradient_tensors':amp_zero,
        'AMP_backward_precision_training_readiness':'OPEN: unscaled AMP has zero gradient tensors where FP32 has none; GradScaler/full-gradient parity not measured',
        'neural_optimizer_updates':0,'new_predictions_during_verification':0,
        'profile_sha256':sha(result),'model_sha256':r['model_sha256'],
        'pilot_unique_tokens':1170938,'adequate_full_pretraining_corpus':False,
        'trained_language_quality':'NOT_MEASURED','independently_qualified':False,
        'timed_profile_GPU_minutes_excluding_platform_startup':old['allocated_GPU_minutes_upper_bound']+r['allocated_GPU_minutes_upper_bound'],
        'combined_quota_ceiling_GPU_minutes':12,'quota_reservation':'60s initial allocation debit +300s repair platform timeout, both with2 GPUs'}
    atomic(output,report);print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True)
    p.add_argument('--tokens',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();verify(a.result,a.tokens,a.baseline,a.output)
