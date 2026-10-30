#!/usr/bin/env python3
"""LANGPARALLEL097: no-fit authentic-publisher evaluation of parallel flat/layered TAC.

Read ONLY original 035 (flat), 039 (59D) and 040 (48D) trained weights,
with source/optimizer/sampler integrity. Original 500K model parameter exception
is historical; this does NOT create or fit a new 500K variant. Parallel scan
and layer math are unchanged. All scores are exploratory exposed DEV.
"""
from __future__ import annotations
import argparse,hashlib,io,json,math,statistics,sys,time,urllib.request
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'research/mathematical-engineering/context026'),str(ROOT/'research/mathematical-engineering/langlayer038'),str(ROOT/'research/mathematical-engineering/langlayer040')]
from context_transition import ContextTransition026
from recurrence_only import RecurrenceOnlyDepth038
from factorial_depth import FactorialDepthControl040
from scripts.langlayer035_sampling import fit_sources,schedule_receipt
STUDIES={'flat_parallel':('LANGLAYER035','tac_context026',500122,40785939365),
         'layered_h59':('LANGLAYER039','tac_recuronly038',499901,40795790808),
         'layered_h48_preferred':('LANGLAYER040','state48_no_graph',499623,40797105091)}
SEEDS=(1701,1702,1703,1704,1705)
COMMIT='2771df165f092a4fbb9f46b5b2e7450f19010adb'
DOCS={
'src/ch01-01-installation.md':'a9e03f534aec9791f6b8200068376fa5678a576c',
'src/ch02-00-guessing-game-tutorial.md':'f118eb5455f20aa14128056733a0182869d816d8',
'src/ch03-01-variables-and-mutability.md':'ed15afc0fd46a011578cdf41b3e4f03a5e3257fc'}
LENGTHS=(64,128)
ANCHORS=tuple(range(256,1280,128)) # 8 authentic distinct future targets/document.
def sha(b):return hashlib.sha256(b).hexdigest()
def blob(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def make(arm):
 if arm=='flat_parallel':return ContextTransition026(vocab=256,ffn_width=501).eval()
 if arm=='layered_h59':return RecurrenceOnlyDepth038(vocab=256).eval()
 if arm=='layered_h48_preferred':return FactorialDepthControl040(vocab=256,state_width=48,upper_graph=False).eval()
 raise ValueError('unknown historical arm')
def source_docs(out):
 source=out/'DEV_SOURCE'
 if source.exists():raise FileExistsError('source archive already exists')
 docs={}
 for name,oid in sorted(DOCS.items()):
  url=f'https://raw.githubusercontent.com/rust-lang/book/{COMMIT}/{name}'
  with urllib.request.urlopen(url,timeout=45) as response:raw=response.read()
  if blob(raw)!=oid or len(raw)<=max(ANCHORS)+1:raise ValueError('publisher Git blob not authentic '+name)
  dest=source/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
  docs[name]=raw
 return docs
def load_docs(source):
 docs={}
 for name,oid in sorted(DOCS.items()):
  raw=(source/name).read_bytes()
  if blob(raw)!=oid:raise ValueError('registered source blob mismatch '+name)
  docs[name]=raw
 return docs
def artifacts(root,arm,seed):
 study,tag,parameters,runid=STUDIES[arm]
 source=list(root.rglob('FIT_AND_EXPOSED_DEV_RECEIPT.json'))
 matches=[]
 for fp in source:
  r=json.loads(fp.read_text())
  if r.get('study')==study and r.get('arm')==tag and r.get('seed')==seed:matches.append((fp,r))
 if len(matches)!=1:raise ValueError(f'missing/duplicate trained checkpoint {arm}/{seed}: {len(matches)}')
 fp,receipt=matches[0]
 if receipt['updates']!=320 or receipt['target_bytes_seen']!=327680 or receipt['parameters']!={'trainable':parameters,'deployed':parameters}:
  raise ValueError('historical training budget or params mismatch')
 if not receipt['checkpoint_reloaded_identical_scored_DEV'] or receipt['protected_TEST']!='NOT_ACCESSED' or receipt['GPU_minutes']!=0:
  raise ValueError('historical qualification mismatch')
 original=fp.parent/receipt['checkpoint_filename']
 raw=original.read_bytes()
 if sha(raw)!=receipt['checkpoint_sha256']:raise ValueError('checkpoint SHA256 corruption')
 cp=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
 if (cp['study'],cp['arm'],cp['seed'],cp['updates'])!=(study,tag,seed,320) or cp['source_commit']!='60403a5409ff2c3f3b07dd2ca91a7a3e096839c7':
  raise ValueError('original checkpoint identity mismatch')
 expected=schedule_receipt(fit_sources(),seed,steps=320)
 if cp['sample_order_sha256']!=expected['sample_order_sha256'] or receipt['source_sample_order_sha256']!=expected['sample_order_sha256'] or receipt['source_bytes_sha256']!=expected['sample_bytes_sha256']:
  raise ValueError('authentic FIT sample order changed')
 if not isinstance(cp.get('optimizer'),dict) or not cp['optimizer'].get('state') or not isinstance(cp.get('rng_torch'),torch.Tensor) or 'rng_numpy' not in cp or 'rng_python' not in cp:
  raise ValueError('original optimizer RNG missing')
 model=make(arm)
 if sum(p.numel() for p in model.parameters())!=parameters:raise ValueError('actual trained size not exact')
 model.load_state_dict(cp['model'],strict=True)
 return model,{'arm':arm,'seed':seed,'source_training_study':study,'training_run_id':runid,'training_updates':320,'checkpoint_sha256':sha(raw),'original_sample_order_sha256':expected['sample_order_sha256'],'learned_parameters':parameters}
@torch.no_grad()
def score(model,docs):
 out={}
 for n in LENGTHS:
  all_nll=[];all_accuracy=[];cpu=[]
  by_doc={}
  for name,raw in sorted(docs.items()):
   contexts=[[int(x) for x in raw[pos-n:pos]] for pos in ANCHORS]
   labels=[int(raw[pos]) for pos in ANCHORS]
   x=torch.tensor(contexts,dtype=torch.long);y=torch.tensor(labels,dtype=torch.long)
   if x.shape!=(len(ANCHORS),n):raise ValueError('chronological publisher context')
   start=time.process_time();logits=model(x)[:,-1];cpu.append(time.process_time()-start)
   lp=F.log_softmax(logits.float().double(),-1)
   nll=-lp.gather(-1,y[:,None]).squeeze(-1)
   correct=logits.argmax(-1).eq(y)
   if not bool(torch.isfinite(nll).all()):raise ValueError('nonfinite genuine source NLL')
   by_doc[name]={'nll_sum':float(nll.sum()),'correct':int(correct.sum()),'n':len(ANCHORS),
     'sha256':sha(raw),'targets':list(ANCHORS)}
   all_nll.extend(nll.tolist());all_accuracy.extend(correct.tolist())
  out[str(n)]={'n':len(all_nll),'mean_nll':statistics.mean(all_nll),
    'accuracy':statistics.mean(all_accuracy),'process_seconds':sum(cpu),
    'per_document':by_doc,'window_reset':True,'carried_state':False}
 return out
@torch.no_grad()
def timing(model,docs):
 raw=next(iter(docs.values()))
 samples=torch.tensor([list(raw[pos-128:pos]) for pos in (256,384,512,640)],dtype=torch.long)
 # Warmups are excluded, fixed same inputs and CPU thread counts for both arms.
 for _ in range(2):model(samples)
 reps=[]
 for _ in range(5):
  t=time.process_time();logits=model(samples);reps.append(time.process_time()-t)
  if logits.shape!=(4,128,256):raise ValueError('causal output shape')
 return {'batch':4,'window':128,'repetitions':5,'times_process_seconds':reps,
         'median_process_ms_per_observed_byte':statistics.median(reps)*1000/512,
         'throughput_mode':'batch4 full window recomputation, NOT incremental persistent decode'}
def run(arm,seed,root,output,source_root):
 if output.exists():raise FileExistsError('immutable result')
 if arm not in STUDIES or seed not in SEEDS:raise ValueError('registered cohort only')
 torch.set_num_threads(2);torch.manual_seed(seed)
 model,original=artifacts(root,arm,seed)
 docs=load_docs(source_root)
 results={'study':'LANGPARALLEL096','experiment':'FROZEN_HISTORICAL_NO_NEW_FIT',
  'original':original,'source_commit':COMMIT,'evaluation_publisher':'rust-lang/book',
  'evaluation_source_published_before_current_study':True,'evaluation_previously_exposed_in_OTHER_studies':True,
  'source_docs':{n:{'git_blob':DOCS[n],'sha256':sha(raw),'source_bytes':len(raw)} for n,raw in docs.items()},
  'score':score(model,docs),'timing':timing(model,docs),
  'complete_original_native_parallel_scan':True,
  'upper_recurrence_scan':arm!='flat_parallel',
  'upper_graph':False,'full_model_streaming':False,
  'optimizer_steps_in_this_study':0,'gpu_minutes':0,'protected_TEST':'UNOPENED',
  'equal_original_fit_target_exposure':True,'matched_original_training_FLOPs':False,
  'no_architecture_superiority_claim':True}
 output.parent.mkdir(parents=True,exist_ok=True)
 output.write_text(json.dumps(results,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'arm':arm,'seed':seed,'nll_at_128':results['score']['128']['mean_nll'],
                    'cpu_ms_per_byte':results['timing']['median_process_ms_per_observed_byte']}))
def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=list(STUDIES),required=True)
 p.add_argument('--seed',choices=SEEDS,type=int,required=True)
 p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True)
 a=p.parse_args();run(a.arm,a.seed,a.root,a.output,a.source_root)
if __name__=='__main__':main()
