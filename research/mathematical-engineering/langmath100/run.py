"""Frozen exact10M original replay and factorized state interventions; no fit.

Requires original099 artifact folders, source-pinned model.py and PROTOCOL.json.
Predictions are sealed before opening DEV_JUDGE; public DEV is not blind.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
SOURCE_SHA='ddb2000482e2aa87a4fbdec19cdfdd01249280263e2438be230799ac1758ce9f'
SEEDS=(9901,9902,9903)
CHECKPOINTS={9901:'b0cf0d278c785c124ceee2b782e104acc814ed87d3c18c44c789b7689b24ab83',
 9902:'fa7329e1038918f050ace1088f3c901ff4f2bfd38e15f19bba3749c9a1b6280a',
 9903:'cf856c624f5566dd6d59e251077a909e851074c7a1f3dbdb4f4d89047bc54ae3'}
FACTORS=('context','lexical','previous_key','fast','slow')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def weight_hash(m):
 h=hashlib.sha256()
 for n,t in sorted(m.state_dict().items()):
  h.update(n.encode());h.update(str(tuple(t.shape)).encode());h.update(t.detach().contiguous().numpy().tobytes())
 return h.hexdigest()
def load_source(path):
 assert sha(path)==SOURCE_SHA,'Exact original source changed'
 s=importlib.util.spec_from_file_location('langmath100_original099',path)
 m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
def state(M,d):return M.State099(tuple(d['owners']),d['clock'],d['tensors'])
def combine(M,ss):
 s=ss[0];assert all(x.clock==s.clock for x in ss)
 return M.State099(tuple(x.owners[0] for x in ss),s.clock,{k:torch.cat([x.tensors[k] for x in ss]) for k in s.tensors})
def split(M,s):return [M.State099((o,),s.clock,{k:v[i:i+1].clone() for k,v in s.tensors.items()}) for i,o in enumerate(s.owners)]
def close(a,b):return bool(torch.allclose(a,b,atol=3e-5,rtol=3e-5))
def state_close(a,b):return all(close(a.tensors[k],b.tensors[k]) for k in a.tensors)

@contextmanager
def suppress_banks(M,which):
 """Clamp matrix AND mass trajectories at the actual source affine scan."""
 original=M.affine_scan
 def intercepted(a,b,initial,*,parallel=True):
  if b.ndim==4 and b.shape[2]==2:
   a=a.clone();b=b.clone();initial=initial.clone()
   for j in which:a[:,:,j]=1;b[:,:,j]=0;initial[:,j]=0
  return original(a,b,initial,parallel=parallel)
 if which:M.affine_scan=intercepted
 try:yield
 finally:M.affine_scan=original

def intervene(M,s,name,donor=None):
 t={k:v.clone() for k,v in s.tensors.items()}
 if name.startswith('reset_'):
  mask=int(name[6:])
  for j,n in enumerate(FACTORS):
   if not mask&(1<<j):continue
   if n in ('context','lexical','previous_key'):t[n].zero_()
   else:
    bank=0 if n=='fast' else 1;t['memory'][:,bank]=0;t['mass'][:,bank]=0
 elif name.startswith('donor_'):
  if donor is None:return None
  if name=='donor_full':t={k:v.clone() for k,v in donor.tensors.items()}
  else:
   for j in ([0] if name=='donor_fast' else [1] if name=='donor_slow' else [0,1]):
    t['memory'][:,j]=donor.tensors['memory'][:,j];t['mass'][:,j]=donor.tensors['mass'][:,j]
 elif name.startswith('shuffle_'):
  for j in ([0] if name=='shuffle_fast' else [1] if name=='shuffle_slow' else [0,1]):
   t['memory'][:,j]=t['memory'][:,j].flip(-1);t['mass'][:,j]=t['mass'][:,j].flip(-1)
 elif name=='corrupt':t={k:v.roll(1,-1) for k,v in t.items()}
 elif name=='fresh_query':t={k:torch.zeros_like(v) for k,v in t.items()}
 return M.State099(s.owners,0 if name=='fresh_query' else s.clock,t)

def infer(M,m,rows,states,condition,donors,aux=None):
 n=len(rows);lp=np.full((n,256),np.nan,dtype=np.float32);gen=np.zeros((n,40),np.uint8)
 available=np.zeros(n,bool);groups=defaultdict(list);prepared=[]
 for i,(r,s) in enumerate(zip(rows,states)):
  if condition=='binding':current=aux[i]
  else:current=intervene(M,s,condition,donors[i])
  prepared.append(current)
  if current is not None:
   available[i]=True;groups[(current.clock,len(r['query_bytes']),r['generation_limit'])].append(i)
 off=[0] if condition=='off_fast' else [1] if condition=='off_slow' else [0,1] if condition=='off_both' else []
 with suppress_banks(M,off):
  for (_,_,limit),ids in groups.items():
   for start in range(0,len(ids),32):
    ii=ids[start:start+32];s=combine(M,[prepared[i] for i in ii]);o=s.owners
    q=torch.tensor([rows[i]['query_bytes'] for i in ii],dtype=torch.long)
    if condition=='direct_history':
     q=torch.tensor([rows[i]['acquisition_bytes']+rows[i]['query_bytes'] for i in ii],dtype=torch.long);s=m.start(o)
    wm=torch.zeros_like(q,dtype=torch.bool) if condition=='no_writes' else None
    if condition=='stateless_every_byte':
     for j in range(q.shape[1]):logits,s=m(q[:,j:j+1],owners=o)
    else:logits,s=m(q,state=s,owners=o,write_mask=wm)
    lp[ii]=logits[:,-1].log_softmax(-1).numpy();token=logits[:,-1].argmax(-1)
    for j in range(limit):
     gen[ii,j]=token.numpy().astype(np.uint8)
     if j+1<limit:
      if condition=='stateless_every_byte':logits,s=m(token[:,None],owners=o)
      else:logits,s=m(token[:,None],state=s,owners=o,write_mask=torch.zeros_like(token[:,None],dtype=torch.bool) if condition=='no_writes' else None)
      token=logits[:,-1].argmax(-1)
 return {'log_probs':lp,'generated':gen,'available':available}

def decode(raw):
 raw=bytes(raw)
 if b';' not in raw:raise ValueError('delimiter')
 p=raw.split(b';',1)[0]
 if len(p)<3 or p[0] not in b'123456' or p[1] not in b'123456':raise ValueError('dimensions')
 if len(p[2:])!=(p[0]-48)*(p[1]-48) or any(v not in b'0123456789' for v in p[2:]):raise ValueError('payload')
 return p

def language(M,m,folder,stage,seed,data):
 cases=read(data/'LANGUAGE_CASES.json');source_info=read(data/'EVAL_MANIFEST.json')['language_sources']
 raw={r['file']:(data/r['file']).read_bytes() for r in source_info};nn=[];cc=[]
 for start in range(0,len(cases),4):
  group=cases[start:start+4];x=torch.tensor([list(raw[r['file']][r['offset']:r['offset']+257]) for r in group])
  logits,_=m(x[:,:-1],owners=tuple(r['id'] for r in group));l=logits.log_softmax(-1)
  nn.extend((-l.gather(-1,x[:,1:,None]).squeeze(-1).mean(1)).tolist());cc.extend((logits.argmax(-1)==x[:,1:]).sum(1).tolist())
 with np.load(folder/f'{stage}-{seed}/language-original.npz',allow_pickle=False) as z:
  assert np.allclose(nn,z['nll'],atol=3e-5,rtol=3e-5),'language NLL replay'
  assert np.array_equal(cc,z['correct']),'language accuracy replay'
  maxdelta=float(np.max(np.abs(np.array(nn)-z['nll'])))
 return {'stage':stage,'cases':len(cases),'max_NLL_delta':maxdelta,
         'mean_NLL':float(np.mean(nn)),'correct':int(sum(cc)),'targets':len(cases)*256}

def timing(M,m,rows,output):
 values=[];o=('timing',)
 for n in (64,256,1024):
  # Original licensed observed bytes only; bounded constructed concatenations.
  observed=bytes(sum([r['acquisition_bytes']+r['query_bytes'] for r in rows],[]))[:n]
  x=torch.tensor([list(observed)]);_,s=m(x,owners=o);q=torch.tensor([[63]])
  for mode in ('carry','slow_off','both_off','fresh_byte','replay'):
   def fn():
    with suppress_banks(M,[1] if mode=='slow_off' else [0,1] if mode=='both_off' else []):
     return m(torch.cat((x,q),1),owners=o) if mode=='replay' else m(q,owners=o) if mode=='fresh_byte' else m(q,state=s,owners=o)
   for _ in range(3):fn()
   samples=[]
   for _ in range(15):
    st=time.perf_counter();fn();samples.append((time.perf_counter()-st)*1000)
   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as p:fn()
   values.append({'history_bytes':n,'mode':mode,'p50_ms':float(np.median(samples)),'p95_ms':float(np.quantile(samples,.95)),
    'source_bytes_per_second':1000/float(np.median(samples)), 'samples_ms':samples,
    'partial_ATen_FLOPs':sum(e.flops for e in p.key_averages()),
    'uncounted_operators':[e.key for e in p.key_averages() if not e.flops],
    'nominal_dense_MACs_per_byte':9921852,'tensor_state_bytes_allocated':11104,
    'cost_note':'Clamp keeps original projections/geometry; not a pruned fast decoder. Replay throughput here counts one newly predicted byte including history reprocessing.'})
 save(output/'TIMING.json',values)

def run(a):
 if a.output.exists():raise FileExistsError('Preserve original output directory')
 a.output.mkdir(parents=True)
 M=load_source(a.model);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 b=a.artifacts;data=b/'langtrain099-EVAL-only';orig=b/'langtrain099-complete-evaluation'
 manifest=read(data/'EVAL_MANIFEST.json')
 assert sha(data/'EVAL_MANIFEST.json')=='549a2fa587de07e6d9cc30944255d5d980c9974e7fb9d366d435567098465c31'
 for name,h in manifest['files'].items():assert sha(data/name)==h,name
 rows=read(data/'DEV_ACTOR.json');dm=read(data/'DEV_DONORS.json');index={r['id']:i for i,r in enumerate(rows)}
 conditions=[f'reset_{i:02d}' for i in range(32)]+['donor_full','donor_fast','donor_slow','donor_banks',
 'shuffle_fast','shuffle_slow','shuffle_banks','off_fast','off_slow','off_both','no_writes','fresh_query','stateless_every_byte','direct_history']
 seeds=[a.seed] if a.seed else list(SEEDS);replay=[];predictions=[]
 for seed in seeds:
  fit=b/f'langtrain099-original-seed-{seed}';receipt=read(fit/'FIT_RECEIPT.json')
  stages=('trained',) if a.smoke else ('initial','trained')
  for stage in stages:
   name='MODEL.pt' if stage=='trained' else 'INITIAL_MODEL.pt';path=fit/name
   expected=receipt['checkpoint_sha256' if stage=='trained' else 'initial_checkpoint_sha256'];assert sha(path)==expected
   if stage=='trained':assert expected==CHECKPOINTS[seed],'Original frozen checkpoint identity differs'
   checkpoint=torch.load(path,weights_only=True);meta=checkpoint['metadata'];m=M.ScaledContextualAssociativeTAC099().eval();m.load_state_dict(checkpoint['model'],strict=True)
   assert m.counts()['trainable']==m.counts()['deployed']==10000000
   assert weight_hash(m)==meta['model_sha256'];before=weight_hash(m)
   assert meta['source_hashes']['research/mathematical-engineering/langtrain099/model.py']==SOURCE_SHA
   if stage=='trained':
    slots=checkpoint['optimizer']['state'];assert len(slots)==17
    assert all(float(v['step'])==2048 for v in slots.values())
   del checkpoint
   with torch.inference_mode():
    if stage=='trained':
     cold=torch.tensor(read(fit/'COLD_FIT_INPUTS.json'));logits,cs=m(cold)
     with np.load(fit/'COLD_FIT_ORIGINAL.npz',allow_pickle=False) as z:
      assert np.allclose(logits.numpy(),z['logits'],atol=3e-5,rtol=3e-5),'cold FIT'
      assert all(np.allclose(v.numpy(),z[k],atol=3e-5,rtol=3e-5) for k,v in cs.tensors.items())
    payload=torch.load(orig/f'{stage}-{seed}/STATE.pt',weights_only=True)
    states=[state(M,s) for s in payload['states']];aux=[state(M,s) if s else None for s in payload['auxiliary_states']]
    # Independently regenerate every native state from original acquisition.
    for i,r in enumerate(rows):
     _,fresh=m(torch.tensor([r['acquisition_bytes']]),owners=states[i].owners)
     assert state_close(states[i],fresh),'acquisition state replay'
    donors=[]
    for i,d in enumerate(dm):
     if not d['available']:donors.append(None);continue
     j=index[d['donor_id']];assert j!=i and rows[i]['domain']==rows[j]['domain'] and states[i].clock==states[j].clock
     donors.append(states[j])
    with np.load(orig/f'{stage}-{seed}/original.npz',allow_pickle=False) as z:
     for ci,cond in enumerate(('reset_00','reset_31','donor_full','corrupt','binding')):
      got=infer(M,m,rows,states,cond,donors,aux);ok=got['available'];assert np.array_equal(ok,z['available'][ci])
      assert np.allclose(got['log_probs'][ok],z['log_probs'][ci,ok],atol=3e-5,rtol=3e-5),('original logit replay',seed,stage,cond)
      assert np.array_equal(got['generated'][ok],z['generated_bytes'][ci,ok]),('original generation replay',seed,stage,cond)
    lang={} if a.smoke else language(M,m,orig,stage,seed,data)
    replay.append({'seed':seed,'stage':stage,'checkpoint_sha256':expected,'source_sha256':SOURCE_SHA,'parameters':10000000,
      'recorded_training_updates':2048 if stage=='trained' else 0,'all_72_acquired_states_reproduced':True,
      'all_original_condition_predictions_reproduced':True,'language_replay':lang,'weight_sha256':before})
    if stage=='trained':
     run_conditions=conditions[:2] if a.smoke else conditions
     for cond in run_conditions:
      tic=time.perf_counter();got=infer(M,m,rows,states,cond,donors,aux)
      pathout=a.output/f'{seed}-{cond}.npz';np.savez_compressed(pathout,**got,ids=np.array([r['id'] for r in rows]))
      predictions.append({'seed':seed,'condition':cond,'file':pathout.name,'sha256':sha(pathout)})
      print(json.dumps({'seed':seed,'condition':cond,'seconds':round(time.perf_counter()-tic,3)}),flush=True)
     # Direct replay must agree with carry before any target score is opened.
     if not a.smoke:
      with np.load(a.output/f'{seed}-reset_00.npz') as x,np.load(a.output/f'{seed}-direct_history.npz') as y:
       assert np.allclose(x['log_probs'],y['log_probs'],atol=3e-5,rtol=3e-5)
       assert np.array_equal(x['generated'],y['generated'])
      timing(M,m,rows,a.output) if seed==9901 else None
    assert weight_hash(m)==before,'weights mutated'
   del m
 save(a.output/'REPRODUCTION.json',{'status':'PASS','rows':replay,'optimizer_steps':0,'source_sha256':SOURCE_SHA})
 save(a.output/'PREDICTIONS_SEALED.json',{'predictions':predictions,'protocol_sha256':sha(HERE/'PROTOCOL.json'),
      'before_new_target_scoring':True,'previously_exposed_DEV':True,'optimizer_steps':0})
 print(json.dumps({'status':'PREDICTIONS_SEALED','output':str(a.output),'sets':len(predictions)}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--smoke',action='store_true')
 run(p.parse_args())
