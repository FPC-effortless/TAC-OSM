"""Numerical fixtures only: exact rational dot checks, not capability evidence."""
from fractions import Fraction
import pytest
import torch
from tac_osm.inward_role_guard import inward_candidates


@pytest.mark.parametrize('seed',range(4))
def test_actual_fp32_candidates_have_nonpositive_exact_rational_dot(seed):
    gen=torch.Generator().manual_seed(seed)
    p=torch.randn(47,generator=gen)
    g=torch.randn(47,generator=gen)
    d=torch.randn(47,generator=gen)*1e-5
    # Arrange the unconstrained proposal near the exact projection boundary.
    d=d-(g.double()*d.double()).sum()/g.double().square().sum()*g
    c=(p+d).float()
    old=[x.clone() for x in (p,g,c)]
    candidate,receipt=inward_candidates([p],[g],[c],[[0]])
    exact=sum(Fraction(float(a))*(Fraction(float(b))-Fraction(float(w))) for a,b,w in zip(g,candidate[0],p))
    assert exact<=0
    assert exact<=Fraction(receipt[0]['certified_upper_dot'])
    assert receipt[0]['certified_upper_dot']<=0
    assert all(torch.equal(a,b) for a,b in zip((p,g,c),old))


def test_zero_gradient_preserves_raw_and_bad_input_fails_closed():
    p=torch.tensor([1.,2.]);c=torch.tensor([2.,3.]);g=torch.zeros_like(p)
    out,receipt=inward_candidates([p],[g],[c],[[0]])
    assert torch.equal(out[0],c) and not receipt[0]['fallback']
    with pytest.raises(ValueError):inward_candidates([p.double()],[g],[c],[[0]])
    with pytest.raises(ValueError):inward_candidates([p],[g],[c],[])
    with pytest.raises(ValueError):inward_candidates([p],[g],[c*float('nan')],[[0]])
