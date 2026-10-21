"""Real-FIT-only parameter/operator audit before any optimizer update."""
import json
from pathlib import Path
import torch
from torch.profiler import profile,ProfilerActivity
from scripts.me034_stable_research import *


def run():
    torch.set_num_threads(1);verify_parent(ROOT/'results/me033')
    data=load_source(ROOT/'results/me022')['train'];rows=roles_manifest(data)
    inc=torch.load(ROOT/'results/me033/r0-s14001-inherited.pt',weights_only=True)
    inc.update(ewma_mean=torch.zeros(12),ewma_scale=torch.ones(12),global_logit=0.)
    s=subset(data,torch.tensor(rows[0]['readout']['indices'][:8]))
    ctx=auxiliary(s,inc['context_S'][:8]);offered=offered_items(s);y=targets(s)
    result={'heldout_labels_opened':False,'optimizer_updates':0,'arms':{}}
    for arm in ARMS:
        eager=construct(arm,inc,82001);n=torch.jit.script(eager)
        args=(s['items'],s['metadata'],offered,ctx)
        n(*args).sum().backward();n.zero_grad(set_to_none=True)
        with profile(activities=[ProfilerActivity.CPU],with_flops=True) as p:
            F.binary_cross_entropy_with_logits(n(*args),y).backward()
        result['arms'][arm]={'active_parameters':sum(v.numel() for v in n.parameters() if v.requires_grad),
            'stored_parameters':sum(v.numel() for v in n.parameters()),
            'buffer_bytes':sum(v.numel()*v.element_size() for v in n.buffers()),
            'B8_forward_backward_operator_FLOP_subtotal':sum(e.flops or 0 for e in p.key_averages()),
            'full_FLOPs':False}
    a=result['arms']['native_frozen']['B8_forward_backward_operator_FLOP_subtotal']
    b=result['arms']['gru_frozen']['B8_forward_backward_operator_FLOP_subtotal']
    result['native_gru_operator_subtotal_relative_gap']=abs(b/a-1)
    result['operator_subtotal_2pct_gate']=abs(b/a-1)<=.02
    result['full_compute_parity']=False
    write_json(ROOT/'research/mathematical-engineering/me034/preflight.json',result)
    print(json.dumps(result))


if __name__=='__main__':run()
