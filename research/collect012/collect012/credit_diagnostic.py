"""No-update diagnostic of the current-token value learning path.

This does not alter or select the frozen training runs. It establishes a local
gradient property, not that a different write order improves language quality.
"""
import json
import torch
from torch.nn import functional as F
from .common import ROOT,SPECS,build,verify_data

def main():
    torch.set_num_threads(1);data,_=verify_data();rows=[]
    for length in (1,8):
        slab=torch.tensor(list(data['train'][100:100+2*(length+1)])).reshape(2,length+1)
        for arm in SPECS:
            model=build(arm,12021).double();captures=[];handles=[]
            for block in model.core.blocks:
                if arm=='attention':
                    original=block._input_projections
                    def projected(x,original=original):
                        result=original(x);captures.append(result['local_value']);return result
                    block._input_projections=projected
                else:
                    def capture(module,args,output):captures.append(output)
                    handles.append(block.phase_memory.value.register_forward_hook(capture))
            logits,_=model(slab[:,:-1]);loss=F.cross_entropy(logits[:,-1],slab[:,-1])
            grad=torch.autograd.grad(loss,captures,allow_unused=True)
            rows.append(dict(arm=arm,length=length,loss=float(loss.detach()),
                value_current_token_gradient_l1=[0. if g is None else float(g[:,-1].abs().sum()) for g in grad],
                value_earlier_token_gradient_l1=[0. if g is None or length==1 else float(g[:,:-1].abs().sum()) for g in grad],
                no_optimizer_updates=True))
            for h in handles:h.remove()
    # Independent scalar-matrix test of a proposed same-token read blend.
    torch.manual_seed(12022);key=F.normalize(torch.randn(16,dtype=torch.float64),dim=0)
    p=torch.eye(16,dtype=torch.float64)*10;s=torch.randn(5,16,dtype=torch.float64)
    value=torch.randn(5,dtype=torch.float64,requires_grad=True);gain=.3
    z=p@key;u=z/(key@z);old=s@key
    updated=s+gain*(value-old)[:,None]*u[None,:]
    after=updated@key;blend=(1-gain)*old+gain*value
    torch.testing.assert_close(after,blend,atol=1e-12,rtol=1e-12)
    current_grad=torch.autograd.grad(after.sum(),value)[0]
    result=dict(status='MEASURED_LOCAL_CREDIT_PROPERTY_NOT_TRAINED_ABLATION',rows=rows,
        proposed_read_after_write=dict(unit_key_self_read_blend_max_error=float((after-blend).abs().max().detach()),
            current_value_gradient=current_grad.tolist(),optimizer_updates=0,
            caution='A causal observed-token self-read may supply a direct value-learning path. It may also encourage copying; trained benefit remains untested.'),
        interpretation='Prior-state phase reads have zero gradient to the current value emission. Earlier value emissions receive gradients over longer sequences. This is a structural distinction, not proof it caused any measured training deficit.')
    (ROOT/'results/CREDIT_DIAGNOSTIC.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
