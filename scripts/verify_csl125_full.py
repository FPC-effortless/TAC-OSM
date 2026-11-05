"""CSL125 123.7M untrained FP32 engineering check; zero optimizer steps."""
import json, os, resource, sys, time, hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tac_osm.csl125 import CSL125, CSL125Config

start=time.perf_counter();torch.set_num_threads(1);torch.manual_seed(20261105)
model=CSL125(CSL125Config());print('constructed',model.parameter_count(), 'secs',round(time.perf_counter()-start,3),flush=True)
assert model.parameter_count()==123_699_534
x=torch.randint(0,model.config.vocab_size,(1,8),dtype=torch.long)
t0=time.perf_counter()
outs,state=model(x,all_rungs=True,stream_ids=('diagnostic',))
fwd=time.perf_counter()-t0
assert sorted(outs)==[2,8,20]
assert [outs[r]['token_logits'].shape for r in [2,8,20]]==[(1,8,32768)]*3
loss=sum(torch.nn.functional.cross_entropy(outs[r]['token_logits'][:,:-1,:].reshape(-1,32768),x[:,1:].reshape(-1)) / 3 for r in [2,8,20])
loss += 0.03*sum(torch.nn.functional.cross_entropy(outs[r]['tool_logits'],torch.tensor([3])) for r in [2,8,20])/3
loss += 0.03*sum(torch.nn.functional.cross_entropy(outs[r]['field_logits'][:,-1,:],torch.tensor([2])) for r in [2,8,20])/3
loss += 0.03*sum(torch.nn.functional.binary_cross_entropy_with_logits(outs[r]['accept_logit'],torch.tensor([[1.0]])) for r in [2,8,20])/3
assert torch.isfinite(loss)
t1=time.perf_counter(); loss.backward();bwd=time.perf_counter()-t1
items=list(model.named_parameters())
nonzero=sum(1 for _,p in items if p.grad is not None and bool(torch.count_nonzero(p.grad)))
missing=[name for name,p in items if p.grad is None]
nonfinite=[name for name,p in items if p.grad is not None and not torch.isfinite(p.grad).all().item()]
print('done forward/backward',round(fwd,3),round(bwd,3),'loss',round(loss.item(),4),'nonzero',nonzero,'missing',len(missing),'nonfinite',len(nonfinite),flush=True)
res={'kind':'CSL125_FULL_123699534_NO_FIT_FORWARD_BACKWARD','seed':20261105,'precision':'fp32','torch':torch.__version__,'parameters':model.parameter_count(),'depths':[2,8,20],'batch':1,'seq':8,'tokenizer_fitted':False,'weights':'random_untrained','optimizer_steps':0,'source_sha256':hashlib.sha256((ROOT/'src/tac_osm/csl125.py').read_bytes()).hexdigest(),'forward_seconds':fwd,'backward_seconds':bwd,'loss_random_targets':float(loss),'tensors_total':len(items),'nonzero_grad_tensors':nonzero,'missing_grad_names':missing,'nonfinite_grad_names':nonfinite,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'status':'PASS' if not missing and not nonfinite else 'FAIL','limitations':'Synthetic random input and labels, no optimizer, no trained capability, no GPU or quantized runtime.'}
out=Path('/mnt/data/CSL125_FULL_FORWARD_BACKWARD_2026-11-05.json')
out.write_text(json.dumps(res,indent=2)+'\n')
print(out,flush=True)