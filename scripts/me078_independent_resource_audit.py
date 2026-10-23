"""No-fit independent torch.profiler verification of the dense FLOP subtotal.

The prospectively frozen dispatch ledger charges scalar nonlinear/reduction ops
separately. This audit checks its matrix multiply/add subtotal against PyTorch's
independent profiler; it does not convert elementary-operation estimates into
hardware instructions or claim uninstrumented operations are free.
"""
from pathlib import Path
import json
import torch
from scripts.me078_train_world import load_original,schedule,batch,model,ROOT,write,SEEDS
from scripts.me078_operation_accounting import profile
from tac_osm.variants.kinematic_world import dense_loss
import numpy as np


def main(cache):
    torch.set_num_threads(1)
    _,fit,_,identified,_=load_original(cache)
    orders,weights,_=schedule(fit,identified,SEEDS[0]);x,a,y,valid=batch(fit,orders[0]);normalizer=float(np.mean(identified['delta_scale']*identified['state_scale']))
    results={}
    for arm in ('native','gru','transformer'):
        torch.manual_seed(SEEDS[0]);m=model(arm,identified);bias=[0]
        def capture(layer,args,out):
            if layer.bias is not None:bias[0]+=out.numel()
        handles=[mod.register_forward_hook(capture) for mod in m.modules() if isinstance(mod,torch.nn.Linear)]
        ledger=profile(m,x,a,y,x,valid,torch.tensor(weights[0]),normalizer)
        for hook in handles:hook.remove()
        expected=sum(ledger['operation_breakdown'].get(k,0) for k in ('mm','addmm','bmm'))-bias[0]
        m.zero_grad(set_to_none=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as audit:
            p=m(x,a);loss=dense_loss(p,y,x,valid,torch.tensor(weights[0]),m.state_scale,normalizer)
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
        actual=sum(event.flops for event in audit.key_averages() if event.key in ('aten::mm','aten::addmm','aten::bmm'))
        if actual!=expected:raise RuntimeError(f'independent matrix accounting mismatch:{arm}/{actual}/{expected}')
        results[arm]={'dispatch_matrix_multiply_add_flops':expected,'independent_torch_profiler_matrix_flops':actual,'linear_bias_adds_charged_separately':bias[0],
           'matrix_fraction_of_charged_update':actual/ledger['total_update_operations'],'match':True}
    write(ROOT/'INDEPENDENT_MATRIX_RESOURCE_AUDIT.json',{'status':'PASS','no_optimizer_steps':True,'uses_authentic_FIT_batch_only':True,'arms':results})
    print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__':main(Path('/tmp/me077-original-source'))
