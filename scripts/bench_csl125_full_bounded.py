"""Short bounded full-geometry CPU no-fit inference timing."""
import json,statistics,time,sys,hashlib,resource,platform
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from tac_osm.csl125 import CSL125,CSL125Config

torch.set_num_threads(1);torch.manual_seed(20261105)
m=CSL125(CSL125Config()).eval()
xx=torch.randint(0,32768,(1,64));yy=torch.randint(0,32768,(1,8))
out={'kind':'CSL125_BOUNDED_FULL_VOCAB_INFERENCE','weights':'random','optimizer_steps':0,'params':m.parameter_count(),'threads':1,'torch':torch.__version__,'source_sha256':hashlib.sha256((ROOT/'src/tac_osm/csl125.py').read_bytes()).hexdigest(),'prompt_tokens':64,'decode_tokens':8,'rungs':{},'limits':'CPU PyTorch FP32 single-run no-fit test, very small sample, no absolute performance claims or GPU comparisons'}
with torch.inference_mode():
 for d in (2,8,20):
  fs=[]; ds=[]
  for rep in range(3):
   t=time.perf_counter();z,s=m(xx,depth=d,stream_ids=('bench',),emit_token_logits=True);fs.append(time.perf_counter()-t)
   t=time.perf_counter()
   for i in range(8):
    z,s=m(yy[:,i:i+1],depth=d,state=s,stream_ids=('bench',),emit_token_logits=True)
   ds.append(time.perf_counter()-t)
  out['rungs'][str(d)]={'prefill_samples_ms':[round(1000*t,3) for t in fs],'decode_8_samples_ms':[round(1000*t,3) for t in ds],'prefill_toks_per_s':round(64/statistics.median(fs),1),'decode_toks_per_s':round(8/statistics.median(ds),1)}
  print(d,out['rungs'][str(d)],flush=True)
out['rss_peak_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
Path('/mnt/data/CSL125_FULL_CPU_BENCHMARK_2026-11-05.json').write_text(json.dumps(out,indent=2)+'\n')