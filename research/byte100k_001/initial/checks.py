"""Fixed no-fit causal/AD correspondence checks, not quality evaluation."""
from dataclasses import fields,is_dataclass,replace
import hashlib
import torch
from torch.nn import functional as F
from tac_osm.byte100k_001 import predict_byte_chunk
from tac_osm.causal_byte_corpus import causal_byte_chunks
from research.byte100k_001.math_reference import mean_byte_ce_and_logit_gradient

ATOL=RTOL=3e-5


def identity(model):
    h=hashlib.sha256()
    for kind,items in [('parameter',model.named_parameters()),('buffer',model.named_buffers())]:
        for name,tensor in items:
            t=tensor.detach().cpu().contiguous()
            h.update(repr((kind,name,str(t.dtype),tuple(t.shape))).encode())
            h.update(t.numpy().tobytes())
    return h.hexdigest()


def leaves(value,path='state'):
    if is_dataclass(value):
        return {k:v for f in fields(value) for k,v in leaves(getattr(value,f.name),path+'.'+f.name).items()}
    if isinstance(value,(tuple,list)):
        return {k:v for i,x in enumerate(value) for k,v in leaves(x,path+'.'+str(i)).items()}
    return {path:value}


def compare_tensor(a,b):
    if a.shape!=b.shape or a.dtype!=b.dtype:
        return False,float('inf')
    error=float((a.detach().double()-b.detach().double()).abs().max()) if a.numel() else 0.
    good=bool(torch.isfinite(a).all() and torch.isfinite(b).all() and torch.allclose(a,b,atol=ATOL,rtol=RTOL))
    return good,error


def check_case(model,document):
    before=identity(model)
    iterator=causal_byte_chunks((document,),role='FIT',chunk_length=128)
    first,second=next(iterator),next(iterator)
    if len(first.inputs)!=128 or len(second.inputs)!=128:
        raise ValueError('registered FIT prefix requires256inputs')
    x=torch.tensor([(*first.inputs,*second.inputs)],dtype=torch.long)
    y=torch.tensor([(*first.targets,*second.targets)],dtype=torch.long)
    full,full_state=model(x,stream_ids=(document.stream_id,))
    loss=F.cross_entropy(full.reshape(-1,256),y.reshape(-1))
    reference,reference_grad=mean_byte_ce_and_logit_gradient(full.detach().numpy(),y.numpy())
    logit_grad=torch.autograd.grad(loss,full,retain_graph=True)[0]
    objective_good=abs(float(loss.detach())-reference)<=1e-6 and torch.allclose(logit_grad.double(),torch.from_numpy(reference_grad),atol=1e-6,rtol=1e-6)
    params=dict(model.named_parameters())
    grad_full=torch.autograd.grad(loss,tuple(params.values()),allow_unused=True)
    a,carry=predict_byte_chunk(model,first)
    b,streamed_carry=predict_byte_chunk(model,second,carry)
    incoming_grad=torch.autograd.grad(F.cross_entropy(b.reshape(-1,256),torch.tensor(second.targets)),
                                     carry.state.layers[0].slow,retain_graph=True)[0]
    joined=torch.cat([a,b],dim=1)
    stream_loss=F.cross_entropy(joined.reshape(-1,256),y.reshape(-1))
    grad_stream=torch.autograd.grad(stream_loss,tuple(params.values()),allow_unused=True)
    output_good,output_error=compare_tensor(full,joined)
    state_checks=[];state_error=0.
    left,right=leaves(full_state),leaves(streamed_carry.state)
    if left.keys()!=right.keys():raise ValueError('state layout mismatch')
    for key in left:
        if isinstance(left[key],torch.Tensor):
            good,error=compare_tensor(left[key],right[key]);state_checks.append(good);state_error=max(state_error,error)
        else:state_checks.append(left[key]==right[key])
    gradient_rows={}
    for (name,p),g,h in zip(params.items(),grad_full,grad_stream):
        connected=(g is not None and h is not None)
        if connected:
            good,error=compare_tensor(g,h)
            nonzero=bool(g.abs().max()>0 and h.abs().max()>0)
        else:good,nonzero,error=False,False,None
        gradient_rows[name]={'connected':connected,'finite_nonzero':nonzero and good,'max_abs_error':error,'within_tolerance':good}
    with torch.no_grad():
        poisoned=x.clone();poisoned[:,64:]=(poisoned[:,64:]+17)%256
        changed,_=model(poisoned,stream_ids=(document.stream_id,))
        future_good,future_error=compare_tensor(full[:,:64],changed[:,:64])
        changed_labels=replace(first,targets=tuple((v+19)%256 for v in first.targets))
        poisoned_output,poisoned_carry=predict_byte_chunk(model,changed_labels)
        label_good,label_error=compare_tensor(a,poisoned_output)
        label_state_good=all(torch.equal(v,leaves(poisoned_carry.state)[k]) if isinstance(v,torch.Tensor)
                             else v==leaves(poisoned_carry.state)[k] for k,v in leaves(carry.state).items())
    state_connected=bool(torch.isfinite(incoming_grad).all() and incoming_grad.abs().max()>0)
    checks={'count':model.validate_live_count()=={'trainable':99050,'deployed':99050},
        'finite_objective':bool(torch.isfinite(loss)),'independent_objective_and_logit_gradient':bool(objective_good),
        'stream_outputs':output_good,'stream_state':all(state_checks),
        'all_parameters_connected_nonzero_and_matching':all(r['connected'] and r['finite_nonzero'] and r['within_tolerance'] for r in gradient_rows.values()),
        'incoming_slow_state_connected':state_connected,'future_prefix_noninterference':future_good,
        'label_poison_noninterference':label_good and label_state_good,'parameters_buffers_unchanged':identity(model)==before}
    return {'document':document.path,'source_sha256':document.sha256,'targets_checked':256,
        'checks':checks,'status':'PASS' if all(checks.values()) else 'FAIL',
        'errors':{'stream_output':output_error,'stream_state':state_error,'future_prefix':future_error,'label_poison':label_error},
        'gradient_tensors':len(gradient_rows),'gradient_rows':gradient_rows},streamed_carry
