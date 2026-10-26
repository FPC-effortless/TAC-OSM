"""Real saved FIT tensors check role isolation; no optimizer update or fitting."""
import copy
import numpy as np
import pytest
import torch
from scripts import me109_residual_training as old
from scripts.me109_observatory_inputs import load,FIELDS
from scripts.me110_objective import balanced,losses
from tac_osm.typed_role_learning import uncertainty_masks,routed_gradients,TypedRoleAdamW


def inputs():
    arrays,_=load()
    return [torch.from_numpy(arrays['batch0_'+k].copy()) for k in FIELDS],torch.from_numpy(arrays['rawscale'].copy())


def test_actual_coordinate_partition_and_uncertainty_updates_preserve_median():
    torch.set_num_threads(1);torch.manual_seed(110)
    model=old.ResidualRoleSensorWorld();batch,scale=inputs()
    masks=uncertainty_masks(model)
    assert sum(int(m.sum()) for m in masks)==4410
    assert sum(int((~m).sum()) for m in masks)==100253
    with torch.no_grad():
        before,state=model(*batch[:6]);q0=model.raw_quantiles(before)[...,1]
        for p,m in zip(model.parameters(),masks):p.add_(torch.where(m,torch.full_like(p,.01),torch.zeros_like(p)))
        after,state2=model(*batch[:6]);q1=model.raw_quantiles(after)[...,1]
    assert torch.equal(q0,q1)
    for key,v in vars(state).items():
        if isinstance(v,torch.Tensor): assert torch.equal(v,getattr(state2,key))


@pytest.mark.parametrize('auxiliary_scale',[1.,1000.])
def test_auxiliary_magnitude_does_not_change_point_coordinate_gradients(auxiliary_scale):
    torch.set_num_threads(1);torch.manual_seed(110)
    model=old.ResidualRoleSensorWorld();batch,scale=inputs();masks=uncertainty_masks(model)
    def compute(multiplier):
        f,_,t=model(*batch[:6],return_trace=True);p,u,_=losses(model,f,t,batch,scale)
        point,receipt=routed_gradients(model,p,multiplier*u)
        return [x.grad.clone() for x in model.parameters()],point,receipt
    baseline,_,_=compute(1.)
    routed,point,receipt=compute(auxiliary_scale)
    assert receipt['point_on_uncertainty_nonzero']==0
    for a,b,mask in zip(baseline,routed,masks):assert torch.equal(a[~mask],b[~mask])
    # Read-only actual proposals, including FP32 parameter-rounding certificate.
    _,blocks=old.blocks(model)
    optimizer=TypedRoleAdamW(old.parameter_groups(model),blocks=list(blocks.values()))
    params0=[p.clone() for p in model.parameters()]
    candidates,moments,certificates=optimizer.propose(point)
    assert all(row['certified_upper_dot']<=0 for row in certificates)
    assert not optimizer.state
    for a,b in zip(params0,model.parameters()):assert torch.equal(a,b)


def test_balancing_uses_actual_availability_not_batch_position():
    errors=torch.arange(24,dtype=torch.float32).reshape(4,6)
    valid=torch.ones_like(errors,dtype=torch.bool);obs=valid.clone();obs[:1]=False
    expected=(errors[0].mean()+errors[1:].mean())/2
    assert balanced(errors,valid,obs)==expected
    order=torch.tensor([3,0,2,1])
    assert balanced(errors[order],valid[order],obs[order])==expected
    with pytest.raises(ValueError): balanced(errors,valid & False,obs)
