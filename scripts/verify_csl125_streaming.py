"""Full 123.7M untrained model: streamed-vs-batched forward and causal-poison tests."""
import os, sys, json, hashlib, time, resource, statistics
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from tac_osm.csl125 import CSL125,CSL125Config

torch.set_num_threads(1);torch.manual_seed(20261105)
m=CSL125(CSL125Config()).eval()
x=torch.randint(0,32768,(1,24));xp=x.clone();xp[:,16:]=(xp[:,16:]+311)%32768
rows=[]
with torch.no_grad():
    for depth in (2,8,20):
        start=time.perf_counter();full,fs=m(x,depth=depth,stream_ids=('unit',))
        p1,s=m(x[:,:7],depth=depth,stream_ids=('unit',))
        p2,s=m(x[:,7:13],depth=depth,state=s,stream_ids=('unit',))
        p3,s=m(x[:,13:],depth=depth,state=s,stream_ids=('unit',))
        stacked=torch.cat([p1[depth]['token_logits'],p2[depth]['token_logits'],p3[depth]['token_logits']],dim=1)
        logits_err=(stacked-full[depth]['token_logits']).abs().max().item()
        pfull,_=m(xp,depth=depth,stream_ids=('unit',))
        poison_err=(pfull[depth]['token_logits'][:,:16]-full[depth]['token_logits'][:,:16]).abs().max().item()
        state_err=max(max((aa.fast-bb.fast).abs().max().item(),(aa.slow-bb.slow).abs().max().item()) for aa,bb in zip(s.layers,fs.layers))
        k_err=max([0.0]+[(aa.k-bb.k).abs().max().item() for aa,bb in zip(s.attention,fs.attention) if aa is not None])
        rows.append({'depth':depth,'max_abs_stream_logits':logits_err,'max_abs_stream_state':state_err,'max_abs_kv':k_err,'max_abs_future_poison_on_prior_logits':poison_err,'wall_seconds':time.perf_counter()-start,'pass':max(logits_err,state_err,k_err)<1e-3 and poison_err==0})
        print(rows[-1],flush=True)
assert all(r['pass'] for r in rows)
res={'kind':'CSL125_123699534_FULL_GEOMETRY_CAUSAL_STREAM_TEST','source_sha256':hashlib.sha256((ROOT/'src/tac_osm/csl125.py').read_bytes()).hexdigest(),'precision':'fp32','weights':'random_init','optimizer_steps':0,'sequence':24,'chunks':[7,6,11],'rows':rows,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'overall_status':'PASS','limits':'Numerical/causal software consistency only; no learned key binding, language quality, or GPU result.'}
Path('/mnt/data/CSL125_FULL_STREAMING_2026-11-05.json').write_text(json.dumps(res,indent=2)+'\n')