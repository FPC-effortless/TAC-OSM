"""Mathematical counterexamples, not trained-model accuracy tests."""
import math
import torch


def test_hard_width_floor_has_zero_width_gradient():
    raw=torch.tensor(-1.,requires_grad=True);base=torch.tensor(1.,requires_grad=True)
    scale=torch.maximum(base*torch.exp(math.log(4.)*raw.tanh()),.8*base)
    graw,gbase=torch.autograd.grad(scale,(raw,base))
    assert float(graw)==0 and float(gbase)>.79


def test_unconditional_lag_difference_is_not_conditional_variance_floor():
    # Unit-variance stationary AR(1), conditional on last true observation.
    rho=.9;age=240
    declared_floor=2*(1-rho**24)*(age/24)
    exact_conditional_variance=1-rho**(2*age)
    assert declared_floor>18*exact_conditional_variance


def test_asymmetric_side_scales_do_not_preserve_base_variance():
    left,right=.8,4.
    variance=.5*(left**2+right**2)-(right-left)**2/(2*math.pi)
    assert variance>6.69


def test_two_psd_gram_terms_need_not_be_positive_definite():
    a=torch.tensor([[1.,0.],[0.,0.]])
    assert float(torch.linalg.eigvalsh(a@a.T+a@a.T).min())==0
