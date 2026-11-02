"""Emit explicitly incomplete prospective registrations from count-only designs."""
import json,sys,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE));from verify import inventory

def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def h(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def run():
 rows=inventory()
 source_order=json.loads((HERE/'evidence/SOURCE_ORDER.json').read_text())
 protocol={'study_id':'LANGSCALE001','status':'DESIGN_FROZEN_EXECUTION_BLOCKED','source_lineage':'LANGCAP105; originals unchanged',
  'prediction_boundary':'Observed bytes through t only; predict byte t+1; state per unique file stream; reset per 128-byte chunk',
  'primary_question':'Does reallocating matched TAC parameters to larger temporal state improve its 1M-to-10M real-code FIT learning curve?',
  'interpretation_limit':'The same-width arm trades head capacity for state; the scaled-memory arm also changes width. Neither isolates an abstract state-size-only cause.',
  'arms':rows,'seeds':[10301,10302,10303],
  'fit':{'optimizer':'AdamW','lr':0.0003,'betas':[0.9,0.95],'eps':1e-8,'weight_decay':0.1,'batch_size':4,'sequence_length':128,'steps':2560,'precision':'FP32','warmup':False,'gradient_clipping':False,'state_carry_across_batches':False},
  'horizon':{'compare_fit_learning_curves_at':[256,512,1024,2048,2560],'additional_training_after_2560':'requires new registration','checkpoints':'retain every listed milestone, plus first and final; never overwrite original LANGCAP105 artifacts'},
  'evaluation':{'original_langcap105_dev':'ARCHIVE_ONLY_NO_RESCORING_OR_TUNING','new_validation':'NOT_YET_SOURCE_QUALIFIED','sealed_test':'UNOPENED_NOT_YET_SOURCE_QUALIFIED','model_selection':'NONE_UNTIL_INDEPENDENT_EVALUATION_REGISTRATION'},
  'information_threat_model':['No DEV or TEST files in trainer mounts','No future bytes or target metadata passed into model forward','Authentic source receipt and per-file chunk boundaries required','Seed-shared data coordinates; no cross-file carried state','Independent evaluator and signing key remain outside actor access'],
  'scope':'New controlled follow-up, not a continuation or relabeling of original checkpoints','new_optimizer_updates':0}
 save(HERE/'PROTOCOL.json',protocol)
 shared={'feature':{'inputs':'Long[B,128] observed bytes,0..255; no target IDs'},'target':{'next_byte':'same original file t+1'},
  'optimizer':protocol['fit'],'tuning':{'trials':0,'policy':'fixed historical recipe for attribution; no per-arm tuning'},
  'selection':{'policy':'no DEV-based selection; freeze final 2560-step weights'},'evaluation':protocol['evaluation']}
 for scale in ('small','medium'):
  target=1_000_000 if scale=='small' else 10_000_000
  entries=[]
  for row in rows:
   if row['scale']!=scale:continue
   for seed in protocol['seeds']:
    entries.append({'architecture':row['arm'],'seed':seed,'trainable_parameters':row['trainable_parameters'],'deployed_parameters':row['deployed_parameters'],
      'nontrainable_state_bytes':4*row.get('recurrent_state_floats',0),'dataset_sha256':'94fc488af0e975c0643e1d566d0468a4091199fdae02b159c8669a01bebef842',
      'train_split_sha256':source_order['fit_file_identity_sha256'],'dev_split_sha256':None,'test_split_sha256':None,'training_sample_order_sha256':source_order['orders'][str(seed)]['coordinates_sha256'],
      'feature_contract_sha256':h(shared['feature']),'target_contract_sha256':h(shared['target']),
      'optimizer_schedule_sha256':h(shared['optimizer']),'tuning_policy_sha256':h(shared['tuning']),
      'selection_policy_sha256':h(shared['selection']),'evaluation_contract_sha256':h(shared['evaluation']),
      'hardware_profile':'Planned Kaggle two observed NVIDIA T4 GPUs; independent seed/model shards, FP32',
      'budget':{'steps':2560,'examples_seen':10240,'tokens_seen':1310720,'batch_size':4,'history_events':128,
         'planned_training_flops':None,'max_training_flops':None,'tuning_trials':0,'tuning_flops':0,
         'prefit_profile_method':'OPEN: partial Torch kernel audit is NOT full CE/backward/AdamW accounting',
         'prefit_profile_training_flops':None}})
  manifest={'schema_version':'tac-osm.parity.v1','study_id':'LANGSCALE001-'+scale.upper(),
   'study_type':'architecture_comparison','status':'INCOMPLETE_BLOCKED_NOT_TRAINING_AUTHORIZATION','registered_before_first_fit':True,
   'model_scale_policy':'declared-target-v2','parameter_scale':{'target_trainable_parameters':target,'target_deployed_parameters':target,'relative_tolerance':0.05,'registered_before_first_fit':True,
   'rationale':'Controlled state-budget reallocation against the original 1M/10M TAC, GRU and Transformer geometries; no dummy padding.'},
   'real_world_source':{'name':'Original CPython source bytes at 60403a5409ff2c3f3b07dd2ca91a7a3e096839c7','sha256':entries[0]['dataset_sha256'],'license':'Python Software Foundation License Version 2; original source license review inherited, fresh evaluator qualification OPEN','observed_real_outcomes':True},
   'literature_gap':{'unresolved_TAConly_question':protocol['primary_question'],
     'what_is_not_being_rediscovered':'Not a new memory algorithm claim; test whether fixed temporal geometry explains poor scaling of this exact TAC implementation.',
     'established_work':[{'paper':'Parallelizing Linear Transformers with the Delta Rule over Sequence Length, NeurIPS 2024','url':'https://arxiv.org/abs/2406.06484','established_result':'Associative matrix memory and efficient delta-rule training predate this allocation study; update law is unchanged here.'},
       {'paper':'Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality, ICML 2024','url':'https://proceedings.mlr.press/v235/dao24a.html','established_result':'Structured recurrent state and hardware-efficient sequence training are established; parameter count is not a full compute measurement.'}]},
   'arms':entries,'incomplete_fields_are_intentional':'Null new validation/TEST hashes and full FLOPs are missing evidence, not invented values. Canonical preflight MUST reject this draft.'}
  save(ROOT/'research/mathematical-engineering'/('langscale001-'+scale)/'FAIR_COMPARISON.json',manifest)
 proposal={'schema':'tac-osm.phone-research.v1','run_id':'langscale001','variant':'System 1','kind':'training','issue_number':694,
  'estimated_cpu_minutes':600,'requires_cuda':False,'gpu_speedup_expected':True,'estimated_gpu_minutes':180,'gpu_strategy':'seed_shard',
  'seeds':protocol['seeds'],'real_source_sha256':'94fc488af0e975c0643e1d566d0468a4091199fdae02b159c8669a01bebef842',
  'study_manifest':'research/mathematical-engineering/langscale001-small/FAIR_COMPARISON.json','workload':None,'diagnostic':None,'dataset_sources':[],'budget_gpu_minutes':240}
 save(ROOT/'research/phone_queue/langscale001.json',proposal)
 nofit={**proposal,'run_id':'langscale001-nofit','kind':'source_audit','estimated_cpu_minutes':10,'requires_cuda':False,'gpu_speedup_expected':False,'estimated_gpu_minutes':0,'gpu_strategy':None,'study_manifest':None,'budget_gpu_minutes':0,'diagnostic':'scientific'}
 save(ROOT/'research/phone_queue/langscale001-nofit.json',nofit)
if __name__=='__main__':run()
