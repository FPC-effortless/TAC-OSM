"""No-fit real full-geometry random-label graph/gradient receipt for CSL125-R2."""
from __future__ import annotations
import json, time, resource, hashlib
from pathlib import Path
import torch
from tac_osm.csl125_r2 import CSL125R2
ROOT=Path(__file__).resolve().parents[1]

def run():
    torch.set_num_threads(1)
    torch.manual_seed(20261106)
    start=time.perf_counter()
    model=CSL125R2().train()
    assert model.parameter_count()==123822574
    tokens=torch.randint(0,32768,(1,8))
    t=time.perf_counter()
    outs,_=model(tokens,all_rungs=True,stream_ids=('R2_source_owned',))
    fwd=time.perf_counter()-t
    assert set(outs)=={2,8,20}
    objective=sum(torch.nn.functional.cross_entropy(
        outs[r]['token_logits'][:,:-1].reshape(-1,32768),tokens[:,1:].reshape(-1))/3
        +.03*outs[r]['tool_logits'].square().mean()
        +.03*outs[r]['field_logits'].square().mean()
        +.03*outs[r]['accept_logit'].square().mean() for r in (2,8,20))
    t=time.perf_counter();objective.backward();bwd=time.perf_counter()-t
    errors={
     'missing_grad':[n for n,p in model.named_parameters() if p.grad is None],
     'nonfinite_grad':[n for n,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all().item()],
     'zero_grad':[n for n,p in model.named_parameters() if p.grad is not None and torch.count_nonzero(p.grad).item()==0],
    }
    gates=[n for n,p in model.named_parameters() if 'write_gate' in n]
    receipt={'stage':'CSL125R2_NO_FIT_FULL_GRAPH',
      'source_sha256':hashlib.sha256((ROOT/'src/tac_osm/csl125_r2.py').read_bytes()).hexdigest(),
      'torch':torch.__version__,'device':'cpu','precision':'FP32','threads':1,
      'parameter_count':model.parameter_count(),'gradient_tensors':sum(1 for _ in model.parameters()),
      'write_gate_parameter_tensors':len(gates),'write_gate_parameters':sum(p.numel() for n,p in model.named_parameters() if 'write_gate' in n),
      'input_shape':[1,8],'rungs':[2,8,20],'optimizer_steps':0,
      'forward_s':round(fwd,4),'backward_s':round(bwd,4),
      'total_s':round(time.perf_counter()-start,4),'rss_max_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      'random_labels_only':True,'tokenizer_fitted':False,'errors':errors,
      'status':'PASS' if not any(errors.values()) else 'FAIL',
      'qualification':'NO_TRAINING_NO_LEARNED_CAPABILITY_OR_ADVANTAGE_NO_REAL_SOURCE'}
    path=ROOT/'research/mathematical-engineering/csl125_r2/FULL_GRAPH_NO_FIT.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('status','parameter_count','gradient_tensors','write_gate_parameter_tensors','forward_s','backward_s','rss_max_kib')},indent=2))
    if receipt['status']!='PASS':raise RuntimeError(errors)
    return receipt

if __name__=='__main__':run()
