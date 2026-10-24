import torch
from tac_osm.gradient_surgery import projected_task_gradients


def test_two_conflicting_gradients_match_independent_projection_and_raw_descent():
    p=torch.nn.Parameter(torch.tensor([.3,.4],dtype=torch.float64))
    a=torch.tensor([1.,0.],dtype=p.dtype);b=torch.tensor([-1.,1.],dtype=p.dtype)
    report=projected_task_gradients([(p*a).sum(),(p*b).sum()],[p])
    expected=(a-a.dot(b)/b.dot(b)*b)+(b-b.dot(a)/a.dot(a)*a)
    torch.testing.assert_close(p.grad,expected,rtol=0,atol=0)
    assert p.grad.dot(a)>0 and p.grad.dot(b)>0 and report['projected_conflicting_pairs']==2


def test_nonconflicting_and_empty_mask_tasks_preserve_sum_gradient():
    p=torch.nn.Parameter(torch.tensor([.3,.4],dtype=torch.float64))
    projected_task_gradients([p.square().sum(),p.sum(),(p*0).sum()],[p],rotation=1)
    torch.testing.assert_close(p.grad,2*p+1,rtol=0,atol=1e-15)


def test_batched_task_vjp_matches_independent_reverse_gradients():
    from tac_osm.gradient_surgery import task_gradient_matrix
    p=torch.nn.Parameter(torch.tensor([.3,.4],dtype=torch.float64));w=torch.nn.Parameter(torch.eye(2,dtype=p.dtype))
    value=w@p;losses=(value.square().sum(),value.sin().sum(),value.exp().sum())
    expected=torch.stack([torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss,(p,w),retain_graph=True)]) for loss in losses])
    actual=task_gradient_matrix(losses,(p,w))
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
