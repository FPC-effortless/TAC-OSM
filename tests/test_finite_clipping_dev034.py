import math
import pytest
import torch
from tac_osm.development.finite_clipping import clip_finite_cpu_gradients_


def test_fp64_norm_candidate_clips_finite_gradients_when_fp32_norm_overflows():
    p=torch.nn.Parameter(torch.ones(2));p.grad=torch.tensor([1e20,1e20])
    with pytest.raises(RuntimeError):torch.nn.utils.clip_grad_norm_([p],1.,error_if_nonfinite=True)
    norm=clip_finite_cpu_gradients_([p])
    assert math.isfinite(norm) and norm==pytest.approx(math.sqrt(2)*1e20)
    assert float(p.grad.double().norm())==pytest.approx(1.,abs=1e-6)


def test_invalid_elements_are_rejected_before_any_gradient_mutates():
    a=torch.nn.Parameter(torch.ones(2));a.grad=torch.tensor([3.,4.]);before=a.grad.clone()
    b=torch.nn.Parameter(torch.ones(1));b.grad=torch.tensor([float('nan')])
    with pytest.raises(RuntimeError):clip_finite_cpu_gradients_([a,b])
    assert torch.equal(a.grad,before) and torch.isnan(b.grad).all()
