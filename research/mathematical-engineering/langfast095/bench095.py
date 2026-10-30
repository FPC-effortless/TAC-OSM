#!/usr/bin/env python3
"""LANGFAST095 original real-source checkpoint no-fit native-byte CPU benchmark.

Both models get the EXACT same archived 094 trained 100,521p original TAC
weights. Compare all native logits and every state tensor on genuine source
bytes before timing; run paired CPU repetitions with explicit process time.
No new training, protected TEST, model selection or GPU.
"""
from __future__ import annotations
import argparse,hashlib,json,statistics,sys,time,random
from pathlib import Path
from dataclasses import replace
import torch
from torch.profiler import profile,ProfilerActivity
ROOT=Path(__file__).resolve().parents[3]
REG=ROOT/'research/mathematical-engineering/langfast095'
sys.path[:0]=[str(ROOT/'research/mathematical-engineering/langmem086'),str(REG)]
from compact_tac import CompactDualClippedTAC086
from optimized095 import OptimizedCompactTAC095
CHECKPOINTS=ROOT/'research/replay-bundles/LANGHEAD094'
OWNER='src/ch01-01-installation.md'
DOC_BLOB='a9e03f534aec9791f6b8200068376fa5678a576c'
SOURCE_BLOB='1e072193e6ba89a46d8bd33e65dd207843558402'
SEEDS=(1701,1702,1703)
def hash_bytes(raw):return hashlib.sha256(raw).hexdigest()
def git_blob(raw):return hashlib.sha1(('blob '+str(len(raw))+'\0').encode()+raw).hexdigest()
def evidence(seed):
 source=(ROOT/'research/mathematical-engineering/langmem086/compact_tac.py').read_bytes()
 if git_blob(source)!=SOURCE_BLOB:raise ValueError('original source code changed')
 path=CHECKPOINTS/f'original-{seed}'
 orig=(path/'MODEL.pt').read_bytes()
 receipt=json.loads((path/'RESULTS.json').read_text())
 if receipt['checkpoint_sha256']!=hash_bytes(orig):raise ValueError('original checkpoint mismatch')
 cp=torch.load(path/'MODEL.pt',map_location='cpu',weights_only=True)
 if cp['study']!='LANGHEAD094' or cp['arm']!='original' or cp['seed']!=seed or cp['steps']!=384:
  raise ValueError('original trained model wrong run or arm')
 if receipt['weights_sha256']!=digest_state(cp['state']):raise ValueError('original trained weights hash mismatch')
 original=CompactDualClippedTAC086().eval();optimized=OptimizedCompactTAC095().eval()
 # The original TiedHeadTAC093 wrapper always registers one fixed byte
 # permutation buffer even for its original arm. Verify it explicitly;
 # this is not a trained weight and was never read by the original model.
 original_fixed=cp['state'].get('shuffled')
 expected_fixed=(torch.arange(256)*73+41)%256
 if original_fixed is None or not torch.equal(original_fixed,expected_fixed):
  raise ValueError('original wrapper fixed buffer mismatch')
 actual_keys=set(cp['state'])-{'shuffled'}
 if not actual_keys or any(not key.startswith('core.') for key in actual_keys):
  raise ValueError('original learned core namespace')
 origstate={k[len('core.'):]:v for k,v in cp['state'].items() if k.startswith('core.')}
 if len(origstate)!=len(actual_keys):raise ValueError('original core key collision')
 original.load_state_dict(origstate,strict=True)
 optimized.load_state_dict(origstate,strict=True)
 if set(original.state_dict())!=set(optimized.state_dict()) or any(
  not torch.equal(x,optimized.state_dict()[k]) for k,x in original.state_dict().items()):
  raise ValueError('live original model weights differ')
 path=CHECKPOINTS/'DEV_SOURCE'/OWNER
 raw=path.read_bytes()
 if git_blob(raw)!=DOC_BLOB or len(raw)<512:raise ValueError('Rust publisher document blob changed')
 return original,optimized,raw,receipt
def digest_state(values):
 h=hashlib.sha256()
 for k,v in sorted(values.items()):
  h.update(k.encode()+b'\0'+v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def state_equal(a,b):
 if a.clock!=b.clock or a.owners!=b.owners or set(a.tensors)!=set(b.tensors):
  return False
 return all(torch.equal(a.tensors[k],b.tensors[k]) for k in a.tensors)
@torch.no_grad()
def verify(original,optimized,raw,seed):
 owner=('real-doc-'+str(seed),)
 orig=original.start(owner);opt=optimized.start(owner)
 max_logits=0.;max_state=0.
 for t in range(256):
  token=torch.tensor([[int(raw[t])]],dtype=torch.long)
  x,orig=original(token,state=orig,owners=owner)
  y,opt=optimized(token,state=opt,owners=owner)
  max_logits=max(max_logits,float((x-y).abs().max()))
  for k in orig.tensors:max_state=max(max_state,float((orig.tensors[k]-opt.tensors[k]).abs().max()))
  if not torch.equal(x,y) or not state_equal(orig,opt):
   raise AssertionError(f'model altered logits/state seed={seed} clock={t} max_logit={max_logits} max_state={max_state}')
 # Multi-byte full forward, lesion controls, batch owner independence.
 for lesion in (None,'fast_zero','slow_zero','both_zero','permuted_read'):
  own=('owner-A','owner-B')
  tokens=torch.tensor([list(raw[16:48]),list(raw[80:112])],dtype=torch.long)
  l1,s1=original(tokens,owners=own,lesion=lesion)
  l2,s2=optimized(tokens,owners=own,lesion=lesion)
  if not torch.equal(l1,l2) or not state_equal(s1,s2):raise AssertionError(f'lesion/owner state changed {lesion}')
 # Error-path equivalence: invalid state must be rejected (including NaN).
 owner=('err-owner',);a=original.start(owner);b=optimized.start(owner)
 for p,state in ((original,a),(optimized,b)):
  bad=replace(state,tensors={**state.tensors,'upper':torch.full_like(state.tensors['upper'],float('nan'))})
  try:p(torch.tensor([[4]]),state=bad,owners=owner)
  except ValueError:pass
  else:raise AssertionError('nonfinite state guard skipped')
  bad2=replace(state,tensors={k:v for k,v in state.tensors.items() if k!='basis'})
  try:p(torch.tensor([[4]]),state=bad2,owners=owner)
  except ValueError:pass
  else:raise AssertionError('shape key guard skipped')
 return dict(seed=seed,token_count=256,logits_max_abs_difference=max_logits,state_max_abs_difference=max_state,
             original_native_state_bytes=original.start(('one',)).tensor_bytes(),
             optimized_native_state_bytes=optimized.start(('one',)).tensor_bytes(),
             learned_parameters=sum(p.numel() for p in original.parameters()),
             state_identity='BITWISE_EQUAL',lesion_and_owner_checks='PASS',nonfinite_and_missing_key_rejection='PASS')
@torch.no_grad()
def decode(model,raw,seed,number=192):
 owner=('timing-'+str(seed),);state=model.start(owner)
 start=time.process_time()
 for j in range(number):
  token=torch.tensor([[int(raw[j])]],dtype=torch.long)
  _,state=model(token,state=state,owners=owner)
 return (time.process_time()-start)*1000/number
def prof(model,raw,seed):
 owner=('profile-'+str(seed),);state=model.start(owner)
 with torch.no_grad():
  with profile(activities=[ProfilerActivity.CPU],record_shapes=False) as p:
   for j in range(16):
    _,state=model(torch.tensor([[int(raw[j])]],dtype=torch.long),state=state,owners=owner)
 obj=sorted(p.key_averages(),key=lambda x:x.self_cpu_time_total,reverse=True)[:22]
 return [{'op':k.key,'self_cpu_us':float(k.self_cpu_time_total),'calls':k.count} for k in obj]
def run(output):
 if output.exists():raise FileExistsError('immutable output receipt')
 torch.set_num_threads(1)
 rows=[];timings=[];profiles={}
 for seed in SEEDS:
  a,b,raw,_=evidence(seed)
  rows.append(verify(a,b,raw,seed))
  # warm both exact source graphs before paired time; no checkpoint mutations
  decode(a,raw,seed,number=32);decode(b,raw,seed,number=32)
  trial=[]
  schedule=['original','optimized']*5
  random.Random(seed).shuffle(schedule)
  for arm in schedule:
   ms=decode(a if arm=='original' else b,raw,seed)
   trial.append({'arm':arm,'ms_per_byte':ms})
  timings.append({'seed':seed,'trials':trial,'median_original_ms':statistics.median(z['ms_per_byte'] for z in trial if z['arm']=='original'),
     'median_optimized_ms':statistics.median(z['ms_per_byte'] for z in trial if z['arm']=='optimized')})
  if seed==SEEDS[0]:
   profiles={'original':prof(a,raw,seed),'optimized':prof(b,raw,seed)}
 before=statistics.median(z['median_original_ms'] for z in timings)
 after=statistics.median(z['median_optimized_ms'] for z in timings)
 report={'study':'LANGFAST095','status':'NO_FIT_INTERNAL_BENCHMARK_NOT_MOBILE_QUALIFICATION',
  'real_document_git_blob':DOC_BLOB,'original_tac_source_blob':SOURCE_BLOB,
  'original_historical_checkpoint_count':len(SEEDS),'original_model_seed_rows':rows,
  'timing':timings,'profiler_first_seed':profiles,
  'median_original_ms':before,'median_optimized_ms':after,
  'relative_median_reduction_percent':(1-after/before)*100,
  'trained_new_models':0,'gpu_minutes':0,'protected_TEST':'UNOPENED',
  'bitwise_predictions_and_all_native_states':'PASS',
  'nonpersistent_buffer_warm_bytes':(4*4+8*8)*4}
 output.parent.mkdir(parents=True,exist_ok=True)
 output.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({k:report[k] for k in ('median_original_ms','median_optimized_ms','relative_median_reduction_percent','bitwise_predictions_and_all_native_states')}))
 return report
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path)
 args=parser.parse_args();run(args.output)
