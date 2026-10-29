#!/usr/bin/env python3
"""LANGMEM085 fail-closed crosspublisher original Rust 3-source-checkpoint aggregate."""
import argparse,json,math,statistics
from pathlib import Path
SEEDS=(1701,1702,1703)
AGES=('8','32','128','256')
ARMS=('source','overwrite','partial025','residual_clip035','selective')
EXPECTED={
  'src/ch02-00-guessing-game-tutorial.md':'bf8769bd079c4b6ae75183f6e22a6b33ed82fdac17cdec52377edae56ab291f0',
  'src/ch04-01-what-is-ownership.md':'873724c6862ad0cc447becf0e818eb39a324c5d4bfa26ef721286aae1941c0ba'}
def summarize(root):
    files=sorted(root.rglob('RUST_RESULTS.json'))
    if len(files)!=3:raise ValueError('exact three source replay models')
    records=[json.loads(x.read_text()) for x in files]
    if sorted(r.get('seed') for r in records)!=list(SEEDS):
        raise ValueError('original 3 seeds missing')
    for r in records:
        if (r.get('study')!='LANGMEM085' or r.get('source_role')!='FIRST_NEW_RUST_PUBLISHER_DEV'
            or not r.get('no_new_fits') or r.get('optimizer_updates')!=0 or
            r.get('original_pretrained_parameters')!=499627 or
            r.get('original_producer_FP32_state_bytes')!=4160 or
            r.get('bank_FP32_bytes')!=256 or
            r.get('proposed_tensors_total_FP32_bytes')!=4416 or
            r.get('historical_cells_per_seed')!=956 or
            r.get('arm_rows_per_seed')!=4780 or len(r.get('rows',[]))!=4780 or
            r.get('source_steps')!=1280 or r.get('original_prefix_error',1)>3e-5 or
            r.get('strict_decoy_no_overlap_count')!=4780 or
            r.get('source_documents_sha256')!=EXPECTED or
            r.get('gpu_minutes')!=0 or r.get('protected_TEST')!='NOT_ACCESSED'):
            raise ValueError('not genuine original Rust publisher source/status/byte budget')
        if set(r.get('metrics',{}))!=set(ARMS):
            raise ValueError('registered five arms absent')
        for row in r['rows']:
            if (row['arm'] not in ARMS or row['age'] not in (8,32,128,256)
                or row['clock']%4 or not 128<=row['clock']<640 or
                row['anchor']!=row['clock']-row['age'] or row['anchor']<4 or
                not row['decoy_clean'] or
                not 5<=row['hard_wrong_age_diff']<=16 or
                not 5<=row['uniform_wrong_age_diff']<=16 or
                row['wrong_pool_size']<=0 or
                not 0<=row['historical_byte']<=255 or
                any(row[k] not in (0,1) for k in ('correct','hard_wrong_correct','uniform_wrong_correct','FIT_prior_correct'))):
                raise ValueError('future/target leakage or wrong source clock in control')
    if len({r['original_checkpoint_sha256'] for r in records})!=3:
        raise ValueError('three original 078 checkpoints not distinct')
    if len({r['historical_byte_cohort_sha256'] for r in records})!=1:
        raise ValueError('source target SHA changed between original models')
    ids=[tuple((q['owner'],q['clock'],q['anchor'],q['age'],q['arm'],q['historical_byte']) for q in r['rows']) for r in records]
    if ids[0]!=ids[1] or ids[0]!=ids[2]:
        raise ValueError('actual source owner/clock/byte cohorts differ')
    output={}
    for arm in ARMS:
        output[arm]={}
        for age in AGES:
            group=[row for r in records for row in r['rows'] if row['arm']==arm and str(row['age'])==age]
            count=len(group)
            if count!=3*{'8':256,'32':256,'128':254,'256':190}[age]:
                raise ValueError('original source age count mismatch')
            output[arm][age]={'source_cells_per_seed':count//3,
                'correct':statistics.mean(x['correct'] for x in group),
                'hard_wrong':statistics.mean(x['hard_wrong_correct'] for x in group),
                'uniform_wrong':statistics.mean(x['uniform_wrong_correct'] for x in group),
                'FIT_prior':statistics.mean(x['FIT_prior_correct'] for x in group),
                'ring_available':statistics.mean(x['raw_ring_available'] for x in group),
                'ring_correct':statistics.mean(x['raw_ring_correct'] for x in group),
                'mean_eta':statistics.mean(x['eta'] for x in group),
                'paired_correct_minus_hard':statistics.mean(x['correct']-x['hard_wrong_correct'] for x in group),
                'paired_correct_minus_uniform':statistics.mean(x['correct']-x['uniform_wrong_correct'] for x in group),
                'unique_count_total_over_three_seeds':sum(x['context_occurrences']==1 for x in group),
                'unique_correct_total_over_three_seeds':sum(x['correct'] for x in group if x['context_occurrences']==1),
                'conflicting_count_total_over_three_seeds':sum(x['context_distinct_continuations']>1 for x in group)}
    return {'study':'LANGMEM085','status':'FIRST_CROSS_PUBLISHER_8X8_SELECTIVE_WRITE_CAUSAL_CONTROLS',
       'model_seed_count':3,'new_publisher_documents_count':2,
       'first_rust_source_historical_labels_per_seed':956,
       'original_source_sha256':records[0]['historical_byte_cohort_sha256'],
       'original_new_publisher_sha256':EXPECTED,
       'source_frozen_checkpoint_shas':{str(r['seed']):r['original_checkpoint_sha256'] for r in records},
       'all_clean_control_target_position_exclusions_pass':True,
       'no_new_model_fits':True,'gpu_minutes':0,'protected_TEST':'NOT_ACCESSED',
       'correct_address_evaluator_supplied_not_bank_encoded':True,
       'additional_bank_bytes':256,'original_model_state_bytes':4160,
       'metrics':output,'per_seed_control_summary':[{
           'seed':r['seed'],'cpu_process_seconds':r['cpu_process_seconds'],
           'write_full_eta_fraction':r['eta_full_write_fraction'],
           'operator_update_wall_seconds':r['operator_update_wall_seconds']}
           for r in records],
       'independently_qualified':False,
       'limitations':['Two original Rust Book chapters only; other publisher but no statistically independent large corpus',
          'Original slow state frozen PSF pretrained encoder plus external exact historical key index',
          'Same deterministic byte/key operators over 3 seeds; model seed independence limited',
          'Raw ring has capacity240 payload bytes vs candidate memory 256B, both need selection/index',
          'Selectivity and delta source updates have different FLOPs, no model next-byte quality proof']}
def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('original result already exists')
    r=summarize(a.results)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print('LANGMEM085_RUST_AGGREGATE '+json.dumps({'metrics':{arm:{age:{k:round(v[k],6) for k in ('correct','hard_wrong','FIT_prior','mean_eta')} for age,v in d.items()} for arm,d in r['metrics'].items()},'source_sha':r['original_source_sha256']},sort_keys=True))
if __name__=='__main__':main()
