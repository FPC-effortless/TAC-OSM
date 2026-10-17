import pytest
import torch
from tac_osm.development.local_mtsk import BankedMTSK
from tac_osm.development.mtsk_dynamics import components,jacobian_metrics,responsibility_tangent_basis


@pytest.mark.parametrize('multiple',[False,True])
def test_diagnostic_transition_matches_original_exactly(multiple):
    torch.manual_seed(3501);k=BankedMTSK(multiple=multiple);x=torch.randn(2,12)
    packed=k.initial(x)[:,0];packed=k.proposal(packed,x)
    assert torch.equal(components(k,packed,x)['next'],k.proposal(packed,x))


def test_conditional_row_identity_and_analytic_gain():
    torch.manual_seed(3502);key=torch.randn(3,dtype=torch.float64);alpha=.8;coefficient=1.7
    operator=alpha*torch.eye(3,dtype=torch.float64)-coefficient*torch.outer(key,key)
    old=torch.randn(3,dtype=torch.float64);change=torch.randn(3,dtype=torch.float64);value=.3
    step=lambda m:alpha*m+coefficient*(value-m@key)*key
    torch.testing.assert_close(step(old+change)-step(old),operator@change)
    expected=max(alpha,abs(alpha-coefficient*float(key@key)))
    assert float(torch.linalg.svdvals(operator)[0])==pytest.approx(expected)


def test_zero_sum_tangent_is_orthonormal_and_excludes_probability_mass_changes():
    q=responsibility_tangent_basis(8,3,'cpu')
    torch.testing.assert_close(q.T@q,torch.eye(21,dtype=torch.float64))
    torch.testing.assert_close(q.reshape(8,3,21).sum(0),torch.zeros(3,21,dtype=torch.float64),atol=1e-14,rtol=0)


def test_full_jacobian_matches_finite_difference_at_small_state():
    torch.set_num_threads(1);torch.manual_seed(3503);k=BankedMTSK();x=torch.randn(1,12)*.1
    packed=k.initial(x)[0,0];metrics=jacobian_metrics(k,packed,x[0]);j=metrics['jacobian']
    tangent=responsibility_tangent_basis(8,3,'cpu');direction=torch.cat((torch.randn(72,dtype=torch.float64),tangent@torch.randn(21,dtype=torch.float64))).float();direction/=direction.norm()
    eps=1e-3
    difference=(k.proposal((packed+eps*direction)[None],x)-k.proposal((packed-eps*direction)[None],x))[0]/(2*eps)
    torch.testing.assert_close(difference.double(),j@direction.double(),rtol=.02,atol=2e-5)
    assert metrics['ambient_gain']+1e-10>=metrics['simplex_tangent_gain']
    batched=jacobian_metrics(k,packed[None].repeat(2,1),x.repeat(2,1),index=(1,))
    torch.testing.assert_close(batched['jacobian'],j,rtol=1e-5,atol=1e-7)
