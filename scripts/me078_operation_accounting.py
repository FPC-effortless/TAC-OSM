"""No-fit actual ATen-dispatch arithmetic ledger. Every observed op is classified.

Scalar operation convention: one per add/multiply/divide/comparison or elementary
transcendental; sigmoid=4, SiLU=5, SiLU derivative=10. This is an algorithmic
operation/FLOP ESTIMATE, not hardware instructions, energy, or measured time.
Dense matrix products use 2*M*N*K plus bias adds. Reductions and backward ops
are charged explicitly. Unsupported float ops FAIL the profile, not omitted.
AdamW arithmetic is charged analytically (16/parameter plus scalar group/tensor
bookkeeping) because a no-fit preflight must not perform an optimizer update.
"""
from collections import Counter
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

class OperationLedger(TorchDispatchMode):
    def __init__(self):
        super().__init__();self.counts=Counter();self.calls=Counter();self.unsupported=Counter()
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {};result=func(*args,**kwargs)
        name=func.__name__.split('.')[0]
        flat,_=tree_flatten(result)
        floating=[v for v in flat if isinstance(v,torch.Tensor) and v.is_floating_point()]
        if not floating:return result
        n=sum(v.numel() for v in floating);cost=None
        if name in ('mm','addmm'):
            a,b=(args[1],args[2]) if name=='addmm' else args[:2]
            cost=2*a.shape[-2]*a.shape[-1]*b.shape[-1]+(n if name=='addmm' else 0)
        elif name=='bmm':cost=2*args[0].shape[0]*args[0].shape[1]*args[0].shape[2]*args[1].shape[2]
        elif name in ('sum','mean'):
            cost=max(0,args[0].numel()-n)+(n if name=='mean' else 0)
        elif name=='_softmax':
            groups=args[0].numel()//args[0].shape[args[1]];cost=5*n-2*groups
        elif name=='_softmax_backward_data':
            groups=args[0].numel()//args[0].shape[args[2]];cost=4*n-groups
        elif name=='binary_cross_entropy_with_logits':cost=12*args[0].numel()
        elif name=='linalg_vector_norm':cost=2*args[0].numel()
        elif name=='_foreach_norm':cost=sum(2*v.numel() for v in args[0])
        elif name=='softplus':cost=5*n
        elif name=='softplus_backward':cost=6*n
        elif name in ('sigmoid',):cost=4*n
        elif name=='silu':cost=5*n
        elif name=='silu_backward':cost=10*n
        elif name in ('sigmoid_backward','tanh_backward'):cost=3*n
        elif name=='lerp':cost=3*n
        elif name in ('rsub','add','add_','sub','sub_','mul','mul_','div','div_','pow','neg','abs','sqrt','rsqrt','exp','log','log1p','asinh','tanh','reciprocal','clamp_min','clamp_max','clamp','maximum','minimum','sign','threshold_backward'):
            cost=n
        elif name in ('view','_unsafe_view','reshape','transpose','t','permute','expand','select','select_backward','slice','slice_backward','clone','detach','lift_fresh','stack','cat','split','split_with_sizes','unbind','unsqueeze','squeeze','zeros','scalar_tensor','empty','empty_like','empty_strided','new_empty_strided','new_zeros','zeros_like','ones_like','full_like','fill_','zero_','copy_','_to_copy','to','masked_fill','where','alias','as_strided','constant_pad_nd','new_empty','slice_scatter','select_scatter'):
            cost=0
        if cost is None:self.unsupported[str(func)]+=1
        else:self.counts[name]+=int(cost)
        self.calls[str(func)]+=1
        return result


def profile(model,x,actions,targets,current,valid,importance,normalizer):
    from tac_osm.variants.kinematic_world import dense_loss
    model.zero_grad(set_to_none=True)
    with OperationLedger() as ledger:
        pred=model(x,actions)
        loss=dense_loss(pred,targets,current,valid,importance,model.state_scale,normalizer)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if ledger.unsupported:
        raise RuntimeError('Uncharged floating operations: '+str(dict(ledger.unsupported)))
    n=sum(p.numel() for p in model.parameters());tensors=len(list(model.parameters()))
    adamw=16*n+20*tensors+16
    return {'forward_backward_clip_scalar_operations':sum(ledger.counts.values()),
       'adamw_analytical_scalar_operations':adamw,'total_update_operations':sum(ledger.counts.values())+adamw,
       'operation_breakdown':dict(ledger.counts),'dispatch_calls':dict(ledger.calls),'unsupported_float_operators':dict(ledger.unsupported),
       'parameter_count':n,'learned_parameter_tensors':tensors}
