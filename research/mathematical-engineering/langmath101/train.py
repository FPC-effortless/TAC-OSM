"""Source-gated real-data pilot worker; never opens DEV or protected TEST."""
import argparse
from dataclasses import fields, is_dataclass
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('langmath101_common',HERE/'common.py');C=importlib.util.module_from_spec(sp);sys.modules[sp.name]=C;sp.loader.exec_module(C)

def state_payload(value):
 """Complete safe tensor/container snapshot, including adaptive cache metadata."""
 if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
 if is_dataclass(value) and not isinstance(value,type):
  return {'dataclass':type(value).__module__+'.'+type(value).__qualname__,
   'fields':{f.name:state_payload(getattr(value,f.name)) for f in fields(value)}}
 if isinstance(value,dict):return {k:state_payload(v) for k,v in value.items()}
 if isinstance(value,tuple):return tuple(state_payload(v) for v in value)
 if isinstance(value,list):return [state_payload(v) for v in value]
 # Torch2.5 weights_only does not allow the reducer for empty bytes. Preserve
 # exact byte contents using plain bounded integer lists, without allowlists.
 if type(value) is bytes:return {'__bytes__':list(value)}
 if value is None or type(value) in (str,bool,int,float):return value
 raise TypeError('Unsupported replay-state field: '+str(type(value)))

def assert_state_equal(left,right,path='state'):
 """Assert every field of the complete source replay state, including types."""
 if isinstance(left,torch.Tensor):
  assert isinstance(right,torch.Tensor) and left.dtype==right.dtype and left.shape==right.shape and torch.equal(left,right),path+' tensor differs'
  return
 assert type(left) is type(right),path+' type differs'
 if isinstance(left,dict):
  assert left.keys()==right.keys(),path+' keys differ'
  for k in left:assert_state_equal(left[k],right[k],path+'.'+str(k))
 elif isinstance(left,(tuple,list)):
  assert len(left)==len(right),path+' length differs'
  for i,(a,b) in enumerate(zip(left,right)):assert_state_equal(a,b,path+'.'+str(i))
 else:assert left==right,path+' value differs'

def semantic_state_payload(model,state):
 """Validate each live model's BPE guard, then rebase only object id/version.

 The guard detects cache reuse after a model object/parameter version changes;
 these process-local values legitimately differ across a checkpoint reload.
 Every persistent tensor, token, byte, owner, clock, dtype/device and tokenizer
 identity remains in the exact semantic comparison and in both raw snapshots.
 """
 snapshot=state_payload(state)
 guard=getattr(state,'cache_guard',())
 if guard:
  assert getattr(model,'mode',None)=='a5_fixed_bpe','Only fixed-BPE states carry a volatile cache guard'
  expected=[('tokenizer',model.tokenizer.receipt()['sha256'])]
  for name,p in model.named_parameters():
   try:version=p._version
   except RuntimeError:version=None
   expected.append((name,id(p),version,str(p.dtype),str(p.device)))
  assert_state_equal(guard,tuple(expected),'model_local_BPE_cache_guard')
  snapshot['fields']['cache_guard']=(guard[0],)+tuple((row[0],row[3],row[4]) for row in guard[1:])
 return snapshot

def train(a):
 if a.output.exists():raise FileExistsError('Original outputs must be preserved')
 manifest=C.read(a.manifest);gate=C.local('owner_two_lane');gate.validate_program(manifest,'preflight',root=C.ROOT)
 assert C.digest(C.RECIPE)==manifest['training_recipe_sha256'],'Runtime recipe differs from frozen profile'
 row=next(r for r in manifest['arms'] if (r['architecture'],r['seed'],r['lane'])==(a.arm,a.seed,a.lane))
 profile=C.read(C.ROOT/manifest['operator_profile_path']);pr=next(r for r in profile['models'] if (r['architecture'],r['seed'])==(a.arm,a.seed))
 actual_source=C.source_hashes();assert actual_source==profile['source_hashes'],'Source changed after profiling'
 bundle=C.local('data').load_fit(a.fit);spec=manifest['source_schedules'][str(a.seed)];fullplan=C.read(C.ROOT/spec['path'])
 assert C.sha(C.ROOT/spec['path'])==spec['sha256']
 tokenizer_path=a.tokenizers/str(a.seed)/'TOKENIZER.json'
 tokenizer_sha=C.sha(tokenizer_path)
 assert tokenizer_sha==profile['tokenizer_receipts_sha256'][str(a.seed)],'Actual tokenizer receipt differs from prefit profile'
 C.setup(a.seed);tokenizer=C.read(tokenizer_path);m=C.make_model(a.arm,tokenizer)
 assert m.counts()['trainable']==row['trainable_parameters'] and m.counts()['deployed']==row['deployed_parameters']
 assert C.state_digest(m)==pr['weights_before'],'Exact prefit initialization changed'
 a.output.mkdir(parents=True);initial=C.state_digest(m)
 metadata={'arm':a.arm,'seed':a.seed,'lane':a.lane,'recipe':C.RECIPE,'tokenizer_receipt':tokenizer,
  'implementation_sha256':profile['implementation_sha256'],'source_hashes':actual_source,
  'training_recipe_sha256':C.digest(C.RECIPE),'tokenizer_receipt_sha256':tokenizer_sha,
  'model_sha256':initial,'optimizer_steps':0,'stage':'INITIAL_BEFORE_DEV'}
 initial_sha=C.checkpoint(a.output/'INITIAL_MODEL.pt',m,None,metadata)
 opt=torch.optim.AdamW(m.parameters(),lr=C.RECIPE['learning_rate'],betas=tuple(C.RECIPE['betas']),eps=C.RECIPE['eps'],weight_decay=C.RECIPE['weight_decay'],foreach=False)
 meter=C.local('meter');journal=[];ledgers={};sample_order=[];losses=[];observed=0;charged=0;started=time.perf_counter();cpu=time.process_time()
 completed_updates=0;attempted_updates=0
 rule=row['budget'].get('dynamic_stop_rule');steps=row['budget']['steps'] if rule is None else rule['max_steps']
 cold_input=C.padded_batch(bundle,fullplan[0])['tokens'];cold_original=None
 try:
  for i,item in enumerate(fullplan[:steps]):
   if time.perf_counter()-started>C.RECIPE['max_arm_seconds']:raise TimeoutError('Preregistered per-fit CPU budget exhausted')
   batch=C.padded_batch(bundle,item);opt.zero_grad(set_to_none=True)
   with meter.ActualTrainingLedger() as trace:
    value=C.loss(m,batch);value.backward()
    for name,p in m.named_parameters():
     if p.grad is not None and not bool(torch.isfinite(p.grad).all()):raise ValueError('Nonfinite gradient: '+name)
    torch.nn.utils.clip_grad_norm_(m.parameters(),C.RECIPE['grad_clip'],error_if_nonfinite=True)
    attempted_updates+=1
    opt.step()
    completed_updates+=1
   r=trace.receipt();r['observed_tokens']=batch['counts']['observed'];journal.append(r);ledgers[r['operator_sha256']]=trace.operators
   charged+=r['counted_dense_units'];observed+=r['observed_tokens'];sample_order.append(item);losses.append(float(value.detach()))
   if rule and r['counted_dense_units']>rule['max_single_step_counted_units']:raise ValueError('Actual step exceeded frozen counted-work bound')
   if i%64==0:print(json.dumps({'arm':a.arm,'seed':a.seed,'lane':a.lane,'step':i+1,'loss':losses[-1],'charged':charged}),flush=True)
   if rule and charged>=rule['target_counted_units']:
    if i+1<rule['min_steps']:raise ValueError('Compute target reached before registered minimum exposure')
    break
  if rule and charged<rule['target_counted_units']:raise ValueError('Maximum steps reached before counted-work target')
  n=len(journal);assert n==len(sample_order)
  final=C.state_digest(m);assert final!=initial,'No actual weight change'
  # Every saved AdamW slot must carry a genuine positive step and finite moments.
  slots=opt.state_dict()['state'];assert len(slots)==len(list(m.parameters()))
  slot_steps=[]
  for slot in slots.values():
   assert 0<float(slot['step'])<=n
   assert all(bool(torch.isfinite(slot[key]).all()) for key in ('exp_avg','exp_avg_sq'))
   slot_steps.append(int(slot['step']))
  m.eval()
  with torch.inference_mode():
   y,state=m(cold_input,owners=('cold0','cold1','cold2','cold3'));cold_original={'logits':y.detach().numpy(),**{'state__'+k:v.detach().numpy() for k,v in state.tensors.items()}}
  meta={**metadata,'model_sha256':final,'optimizer_steps':n,'stage':'SEALED_BEFORE_DEV'}
  checkpoint_sha=C.checkpoint(a.output/'MODEL.pt',m,opt,meta)
  # Actual cold load of original final bytes, not merely a metadata comparison.
  loaded=C.load_model(torch.load(a.output/'MODEL.pt',weights_only=True))
  with torch.inference_mode():
   cy,cs=loaded(cold_input,owners=('cold0','cold1','cold2','cold3'))
   assert torch.equal(cy,y),'Cold source logits differ'
   original_state=state_payload(state);reloaded_state=state_payload(cs)
   assert_state_equal(semantic_state_payload(m,state),semantic_state_payload(loaded,cs))
   # Continue both original and reloaded models on the next four authentic FIT
   # observations from the frozen common schedule. No targets enter this test.
   next_input=C.padded_batch(bundle,fullplan[1])['tokens'][:,:4]
   ny,ns=m(next_input,state=state,owners=('cold0','cold1','cold2','cold3'))
   ncy,ncs=loaded(next_input,state=cs,owners=('cold0','cold1','cold2','cold3'))
   assert torch.equal(ny,ncy),'Cold source continuation logits differ'
   assert_state_equal(semantic_state_payload(m,ns),semantic_state_payload(loaded,ncs))
   continued_original=state_payload(ns);continued_reloaded=state_payload(ncs)
  np.savez_compressed(a.output/'COLD_FIT.npz',tokens=cold_input.numpy(),**cold_original)
  torch.save(original_state,a.output/'COLD_FIT_STATE.pt')
  torch.save(reloaded_state,a.output/'COLD_RELOADED_STATE.pt')
  np.savez_compressed(a.output/'COLD_CONTINUATION.npz',tokens=next_input.numpy(),logits=ny.detach().numpy())
  torch.save({'original':continued_original,'reloaded':continued_reloaded},a.output/'COLD_CONTINUATION_STATE.pt')
  receipt={'trainable_parameters':m.counts()['trainable'],'deployed_parameters':m.counts()['deployed'],
   'checkpoint_sha256':checkpoint_sha,'initial_checkpoint_sha256':initial_sha,'initial_weights_sha256':initial,
   'final_weights_sha256':final,'implementation_sha256':profile['implementation_sha256'],
   'training_recipe_sha256':C.digest(C.RECIPE),'tokenizer_receipt_sha256':tokenizer_sha,
   'actual_sample_order_sha256':C.digest(sample_order),'sample_order':sample_order,
   'actual_operator_journal_sha256':C.digest(journal),'operator_steps':journal,'operator_ledgers':ledgers,
   'actual_steps':n,'actual_examples_seen':4*n,'actual_tokens_seen':observed,'actual_dense_compute_tokens':4*272*n,
   'charged_operator_units':charged,'optimizer_cpu_seconds':time.process_time()-cpu,
   'elapsed_seconds':time.perf_counter()-started,'peak_memory_bytes':C.hardware()['peak_rss_bytes'],'hardware':C.hardware(),
   'arm':a.arm,'seed':a.seed,'lane':a.lane,'optimizer_slot_steps':slot_steps,'cold_replay':'EXACT_SEMANTIC_STATE',
   'cold_replay_scope':'all_persistent_content_exact; BPE_object_id_and_parameter_version_verified_per_model_then_rebased',
   'cold_state_encoding':'typed_containers_tensors_and_tagged_byte_lists',
   'cold_state_sha256':C.sha(a.output/'COLD_FIT_STATE.pt'),
   'cold_reloaded_state_sha256':C.sha(a.output/'COLD_RELOADED_STATE.pt'),
   'cold_continuation_state_sha256':C.sha(a.output/'COLD_CONTINUATION_STATE.pt'),
   'cold_continuation':'EXACT_LOGITS_AND_SEMANTIC_STATE_ON_NEXT_FOUR_FIT_OBSERVATIONS',
   'protected_TEST_access':False,'status':'SEALED_BEFORE_DEV','training_loss_first16':float(np.mean(losses[:16])),
   'training_loss_last16':float(np.mean(losses[-16:])),'full_model_FLOPs':'INCONCLUSIVE_NOT_CLAIMED'}
  C.save(a.output/'FIT_RECEIPT.json',receipt);C.save(a.output/'LOSSES.json',losses)
  print(json.dumps({'status':'SEALED_BEFORE_DEV','arm':a.arm,'seed':a.seed,'lane':a.lane,'steps':n,'elapsed':receipt['elapsed_seconds']}),flush=True)
 except Exception as e:
  C.local('failure').preserve(C,a.output,m,opt,metadata,journal,completed_updates,
                            attempted_updates,time.perf_counter()-started,e)
  raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--fit',type=Path,required=True);p.add_argument('--tokenizers',type=Path,required=True)
 p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--lane',required=True);p.add_argument('--output',type=Path,required=True);train(p.parse_args())
