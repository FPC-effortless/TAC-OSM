"""Bounded, reproducible full 123.7M R1 integrity test; NEVER optimize weights."""
import hashlib, json, os, platform, resource, sys, time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tac_osm.csl125_r1 import CSL125R1, CSL125R1Config


def run():
    torch.set_num_threads(1)
    torch.manual_seed(20261105)
    start=time.perf_counter()
    m=CSL125R1(CSL125R1Config()).train()
    assert m.parameter_count()==123_699_534
    tokens=torch.randint(0,m.config.vocab_size,(1,8))
    ts=time.perf_counter()
    out,state=m(tokens,all_rungs=True,stream_ids=('full-no-fit',))
    fwd=time.perf_counter()-ts
    assert list(out)==[2,8,20]
    objective=sum(torch.nn.functional.cross_entropy(
       out[r]['token_logits'][:,:-1].reshape(-1,m.config.vocab_size),
       tokens[:,1:].reshape(-1)) / 3 for r in m.config.rungs)
    objective+=.03 * sum(out[r]['tool_logits'].square().mean() for r in m.config.rungs)
    objective+=.03 * sum(out[r]['field_logits'].square().mean() for r in m.config.rungs)
    objective+=.03 * sum(out[r]['accept_logit'].square().mean() for r in m.config.rungs)
    assert torch.isfinite(objective)
    ts=time.perf_counter()
    objective.backward()
    backward=time.perf_counter()-ts
    names=list(m.named_parameters())
    missing=[n for n,p in names if p.grad is None]
    nonfinite=[n for n,p in names if p.grad is not None and not torch.isfinite(p.grad).all().item()]
    zero=[n for n,p in names if p.grad is not None and not torch.count_nonzero(p.grad).item()]
    result={'stage':'SOURCE_BOUND_FULL_123699534_NO_FIT', 'source_sha256':hashlib.sha256((ROOT/'src/tac_osm/csl125_r1.py').read_bytes()).hexdigest(),
      'torch':str(torch.__version__),'device':'cpu','threads':1,'precision':'fp32','tokenizer_fitted':False,'total_parameters':m.parameter_count(),'parameter_breakdown':m.parameter_breakdown(),
      'activation_depths':list(out),'batch':1,'length':8,'optimizer_steps':0,'weights':'random_untrained',
      'forward_seconds':round(fwd,5),'backward_seconds':round(backward,5),'constructed_and_checked_seconds':round(time.perf_counter()-start,5),
      'random_label_loss':float(objective.detach()),'tensor_count':len(names),'nonzero_grad_tensors':len(names)-len(zero)-len(missing),
      'missing_grads':missing,'nonfinite_grads':nonfinite,'zero_grads':zero,'peak_RSS_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      'status':'PASS' if not missing and not nonfinite and not zero else 'FAIL',
      'limits':'Not a trained model, no optimizer updates, no real corpus, no CUDA, no full mathematical or anti-leakage proof.'}
    outpath=ROOT/'research/mathematical-engineering/csl125_r1/FULL_ENGINEERING.json'
    outpath.parent.mkdir(parents=True,exist_ok=True)
    outpath.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','total_parameters','tensor_count','nonzero_grad_tensors','forward_seconds','backward_seconds','peak_RSS_kib')},indent=2))
    assert result['status']=='PASS'
    return result
if __name__=='__main__':run()
