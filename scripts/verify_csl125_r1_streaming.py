"""Actual full 123.7M CSL-125-R1 batch/stream causal consistency, no fit."""
import hashlib,json,resource,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from tac_osm.csl125_r1 import CSL125R1,CSL125R1Config

torch.set_num_threads(1);torch.manual_seed(20261105)
m=CSL125R1(CSL125R1Config()).eval()
x=torch.randint(0,32768,(1,24));xp=x.clone();xp[:,16:]=(xp[:,16:]+311)%32768
rows=[]
with torch.no_grad():
    for depth in m.config.rungs:
        start=time.perf_counter()
        full,fs=m(x,depth=depth,stream_ids=('unit',))
        p1,s=m(x[:,:7],depth=depth,stream_ids=('unit',))
        p2,s=m(x[:,7:13],depth=depth,state=s,stream_ids=('unit',))
        p3,s=m(x[:,13:],depth=depth,state=s,stream_ids=('unit',))
        stacked=torch.cat([p1[depth]['token_logits'],p2[depth]['token_logits'],p3[depth]['token_logits']],dim=1)
        logit_err=(stacked-full[depth]['token_logits']).abs().max().item()
        poison,_=m(xp,depth=depth,stream_ids=('unit',))
        poison_err=(poison[depth]['token_logits'][:,:16]-full[depth]['token_logits'][:,:16]).abs().max().item()
        state_err=max(max((getattr(a,n)-getattr(b,n)).abs().max().item()
                          for n in ('fast','slow','fast_mass','slow_mass','conv_tail'))
                          for a,b in zip(s.layers,fs.layers))
        k_err=max([0.]+[(a.k-b.k).abs().max().item() for a,b in zip(s.attention,fs.attention) if a is not None])
        rows.append({'depth':depth,'max_abs_stream_logits':logit_err,'max_abs_state_error':state_err,
                     'max_abs_kv_error':k_err,'max_abs_future_poison_on_earlier_logits':poison_err,
                     'timing_seconds':round(time.perf_counter()-start,4),
                     'status':'PASS' if max(logit_err,state_err,k_err)<1e-3 and poison_err==0 else 'FAIL'})
        print(rows[-1],flush=True)
assert all(row['status']=='PASS' for row in rows)
result={'type':'CSL125_R1_FULL_UNTRAINED_CAUSAL_STREAMING','source_sha256':hashlib.sha256((ROOT/'src/tac_osm/csl125_r1.py').read_bytes()).hexdigest(),
        'parameters':m.parameter_count(),'torch':torch.__version__,'device':'cpu','precision':'fp32','depths':[2,8,20],
        'batch':1,'sequence':24,'chunks':[7,6,11],'tokenizer_fitted':False,'optimizer_updates':0,
        'rows':rows,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'status':'PASS', 'limits':'Synthetic source-only computation; no trained capability, full formal verification, CUDA, independent source authority or hardware relevance.'}
file=ROOT/'research/mathematical-engineering/csl125_r1/FULL_STREAMING.json'
file.write_text(json.dumps(result,indent=2)+'\n')
