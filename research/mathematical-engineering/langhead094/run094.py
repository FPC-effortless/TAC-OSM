#!/usr/bin/env python3
"""LANGHEAD094: original real-FIT 384-step independent 5-seed x 4-architecture study.

Checkpoint is sealed before any evaluation documents are retrieved.
Same exposure/seed/optimizer/parameter parity, but training FLOPs not matched.
No architecture superiority, protected TEST, or device speed claims.
"""
from __future__ import annotations
import argparse,hashlib,json,random,resource,sys,time,urllib.request
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[3]
REG=ROOT/'research/mathematical-engineering/langhead094'
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'research/mathematical-engineering/langmem086'),str(REG)]
from scripts.replay_langmem086087 import fit_sources
from scripts.validate_research_parity import check_manifest
from model094 import ARMS,SEEDS,PARAMS,make_model
COMMIT='2771df165f092a4fbb9f46b5b2e7450f19010adb'
DEV_BLOBS={
 'src/ch01-01-installation.md':'a9e03f534aec9791f6b8200068376fa5678a576c',
 'src/ch02-00-guessing-game-tutorial.md':'f118eb5455f20aa14128056733a0182869d816d8',
 'src/ch03-01-variables-and-mutability.md':'ed15afc0fd46a011578cdf41b3e4f03a5e3257fc'}
STEPS=384;BATCH=4;LENGTH=64;EVAL_START=128;EVAL_STOP=640
def digest(b):return hashlib.sha256(b).hexdigest()
def counts(m):
 n=sum(p.numel() for p in m.parameters() if p.requires_grad)
 return {'trainable':n,'deployed':sum(p.numel() for p in m.parameters())}
def source_hash():return digest((REG/'model094.py').read_bytes())
def state_hash(m):
 h=hashlib.sha256()
 for k,v in sorted(m.state_dict().items()):
  h.update(k.encode()+b'\0'+v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def read_new_dev(local=None):
 out={}
 for name,blob in sorted(DEV_BLOBS.items()):
  if local is None:
   url='https://raw.githubusercontent.com/rust-lang/book/'+COMMIT+'/'+name
   with urllib.request.urlopen(url,timeout=30) as r:raw=r.read()
  else:raw=(local/name).read_bytes()
  if len(raw)<EVAL_STOP+2 or hashlib.sha1(('blob '+str(len(raw))+'\0').encode()+raw).hexdigest()!=blob:
   raise ValueError('pinned DEV Git blob identity failed: '+name)
  out[name]=raw
 return out
def train_batches(docs,seed):
 rng=np.random.default_rng(seed);names=sorted(docs)
 for step in range(STEPS):
  row=[]
  for _ in range(BATCH):
   owner=names[int(rng.integers(len(names)))];raw=docs[owner]
   start=int(rng.integers(len(raw)-LENGTH))
   row.append((owner,start,raw[start:start+LENGTH+1]))
  yield step,row
def replay_schedule(docs,seed):
 h=hashlib.sha256();b=hashlib.sha256()
 for _,row in train_batches(docs,seed):
  for owner,start,raw in row:
   h.update(f'{owner}:{start};'.encode());b.update(raw)
 return {'seed':seed,'steps':STEPS,'batch':BATCH,'context':LENGTH,'targets':STEPS*BATCH*LENGTH,
         'sample_order_sha256':h.hexdigest(),'sample_bytes_sha256':b.hexdigest()}
@torch.no_grad()
def evaluate(m,docs):
 m.eval();allrows=[];process=time.process_time()
 for owner,raw in sorted(docs.items()):
  state=m.start((owner,));tokens=[]
  for t in range(EVAL_STOP):
   logits,state=m(torch.tensor([[int(raw[t])]],dtype=torch.long),state=state,owners=(owner,))
   if t<EVAL_START:continue
   actual=int(raw[t+1]);lp=logits[0,0].log_softmax(-1)
   tokens.append({'ordinal':t,'nll':float(-lp[actual]),'correct':int(int(lp.argmax())==actual)})
  allrows.append({'owner':owner,'source_git_blob':DEV_BLOBS[owner],'sha256':digest(raw),'n':len(tokens),
      'mean_nll':sum(t['nll'] for t in tokens)/len(tokens),
      'accuracy':sum(t['correct'] for t in tokens)/len(tokens),'tokens':tokens})
 return allrows,{'process_seconds':time.process_time()-process,'targets':len(allrows)*(EVAL_STOP-EVAL_START)}
@torch.no_grad()
def causal_gate(m,prefix):
 x=torch.tensor([list(prefix[:12])]);poison=x.clone();poison[:,8:]=(poison[:,8:]+137)%256
 a,_=m(x,owners=('prefix-gate',));b,_=m(poison,owners=('prefix-gate',))
 if not torch.equal(a[:,:8],b[:,:8]):raise ValueError('future suffix influenced earlier predictions')
 st=m.start(('prefix-gate',));rows=[]
 for i in range(x.shape[1]):
  p,st=m(x[:,i:i+1],state=st,owners=('prefix-gate',));rows.append(p)
 err=float((a-torch.cat(rows,1)).abs().max())
 if err>3e-5:raise ValueError('native streaming parity')
 return {'poison':'PASS','streaming_max_abs_error':err}
def run(arm,seed,out):
 if arm not in ARMS or seed not in SEEDS:raise ValueError('unregistered arm/seed')
 if out.exists():raise FileExistsError('immutable result destination')
 torch.set_num_threads(2);torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
 m=make_model(arm)
 manifest=json.loads((REG/'FAIR_COMPARISON.json').read_text())
 if check_manifest(manifest,'preflight')['status']!='PASS_PREFIT_SINGLE_MODEL':raise ValueError('scientific preflight')
 if source_hash()!=manifest['implementation_sha256']:raise ValueError('registered implementation hash')
 expected=next(z for z in manifest['models'] if z['name']==arm and z['seed']==seed)
 if counts(m)!={'trainable':PARAMS[arm],'deployed':PARAMS[arm]} or expected['trainable_parameters']!=PARAMS[arm] or expected['deployed_parameters']!=PARAMS[arm]:
  raise ValueError('actual trained model parameter parity')
 docs,sha=fit_sources(ROOT)
 if sha!=manifest['real_world_source']['sha256'] or len(docs)!=14:raise ValueError('original publisher FIT')
 sched=replay_schedule(docs,seed)
 pre=causal_gate(m,next(iter(docs.values())))
 optimizer=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01)
 before=state_hash(m);steps=[];order=hashlib.sha256();by=hashlib.sha256();start=time.process_time()
 gradients=[]
 for step,rows in train_batches(docs,seed):
  for owner,offset,raw in rows:order.update(f'{owner}:{offset};'.encode());by.update(raw)
  x=torch.tensor([list(raw[:-1]) for _,_,raw in rows]);y=torch.tensor([list(raw[1:]) for _,_,raw in rows])
  optimizer.zero_grad(set_to_none=True)
  logits,_=m(x,owners=tuple(f'fit-{i}' for i in range(BATCH)))
  loss=F.cross_entropy(logits.reshape(-1,256),y.reshape(-1))
  if not torch.isfinite(loss):raise ValueError('nonfinite loss')
  loss.backward()
  if any(p.grad is None or not torch.isfinite(p.grad).all() for p in m.parameters()):
   raise ValueError('unconnected or nonfinite trained gradient')
  if arm=='tied_dot' and step in (0,STEPS-1):
   gradients.append(float(m.raw_strength.grad.abs().max()))
   if gradients[-1]==0:raise ValueError('trainable output head inactive')
  torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
  optimizer.step();steps.append(float(loss.detach()))
  if step in (0,STEPS-1):print(json.dumps({'arm':arm,'seed':seed,'step':step+1,'fit_loss':steps[-1]}),flush=True)
 cpu=time.process_time()-start
 if order.hexdigest()!=sched['sample_order_sha256'] or by.hexdigest()!=sched['sample_bytes_sha256']:raise ValueError('actual real-sample exposure parity')
 if state_hash(m)==before:raise ValueError('model not trained')
 out.mkdir(parents=True)
 cp={'study':'LANGHEAD094','arm':arm,'seed':seed,'steps':STEPS,'source_sha256':sha,
     'model_source_sha256':source_hash(),'state':m.state_dict(),'optimizer':optimizer.state_dict(),
     'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state()[1].tolist(),
     'python_rng':random.getstate(),'schedule':sched,'counts':counts(m),'losses':steps}
 cpfile=out/'MODEL.pt';torch.save(cp,cpfile);cpsha=digest(cpfile.read_bytes())
 loaded=torch.load(cpfile,map_location='cpu',weights_only=True)
 cold=make_model(arm).eval();cold.load_state_dict(loaded['state'],strict=True)
 if state_hash(cold)!=state_hash(m):raise ValueError('cold original weights replay')
 post=causal_gate(cold,next(iter(docs.values())))
 # New docs retrieved ONLY AFTER sealing complete model+optimizer+RNG checkpoint.
 new_docs=read_new_dev()
 source_directory=out/'DEV_SOURCE'
 for path,raw in new_docs.items():
  q=source_directory/path;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(raw)
 scores,timing=evaluate(cold,new_docs)
 receipt={'study':'LANGHEAD094','arm':arm,'seed':seed,'checkpoint_sha256':cpsha,
  'weights_sha256':state_hash(cold),'source_sha256':sha,'source_code_sha256':source_hash(),
  'counts':counts(cold),'train_steps':STEPS,'train_targets':sched['targets'],
  'real_sample_schedule':sched,'fitted_loss_first':steps[0],'fitted_loss_last':steps[-1],
  'train_cpu_seconds':cpu,'peak_rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  'extra_gradients':gradients,'causal_prefit':pre,'causal_postfit':post,
  'dev':scores,'latency':timing,'new_dev_after_checkpoint_seal':True,
  'gpu_minutes':0,'protected_TEST':'NOT_ACCESSED',
  'evidence':'internal_training_matched_data_exposure_params_NOT_matched_training_FLOPs'}
 (out/'RESULTS.json').write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'arm':arm,'seed':seed,'checkpoint_sha256':cpsha,'dev_mean_nll':sum(z['mean_nll'] for z in scores)/len(scores)}),flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=ARMS,required=True)
 ap.add_argument('--seed',type=int,choices=SEEDS,required=True);ap.add_argument('--out',type=Path,required=True)
 args=ap.parse_args();run(args.arm,args.seed,args.out)
if __name__=='__main__':main()
