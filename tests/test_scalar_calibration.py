"""Constructed mathematical controls, not real learned-model scores."""
import math
import pytest
from tac_osm.scalar_calibration import LogitBatch, fit_temperature, loss_derivatives, score_temperature


def batch(rows=((2.,0.),(2.,0.)), labels=(0,1), role='calibration', doc='cal'):
    return LogitBatch(tuple(map(tuple,rows)),tuple(labels),(doc,)*len(rows),role)


def test_convex_derivatives_against_differences():
    b=batch();a=.8;eps=1e-4
    loss,g,h=loss_derivatives(b,a)
    lo=loss_derivatives(b,a-eps)[0];hi=loss_derivatives(b,a+eps)[0]
    assert g==pytest.approx((hi-lo)/(2*eps),abs=1e-7)
    assert h==pytest.approx((hi-2*loss+lo)/eps**2,abs=1e-6)
    assert h>=0


def test_calibration_guard_identity_and_boundaries():
    b=batch();fit=fit_temperature(b)
    assert fit.inverse_temperature==.25 and fit.calibration_nll<=fit.identity_nll
    assert b.logits==((2.,0.),(2.,0.))
    confident=fit_temperature(batch(labels=(0,0)))
    assert confident.inverse_temperature==4.
    flat=fit_temperature(batch(rows=((0.,0.),(0.,0.))))
    assert flat.inverse_temperature==1.


def test_bias_changes_mode_temperature_preserves_order():
    logits=(2.,1.,0.)
    for a in (.25,.6,1.,4.):
        assert sorted(range(3),key=lambda i:a*logits[i])==[2,1,0]
    assert max(range(3),key=lambda i:logits[i]+(0.,3.,0.)[i])==1


def test_score_can_worsen_even_when_calibration_improves():
    rows=((10.,)+(0.,)*127,)*2
    fitted=fit_temperature(batch(rows=rows))
    test=batch(rows,labels=(0,0),role='score',doc='other-dev')
    r=score_temperature(test,fitted,codec='mapped128')
    assert r['nll_nats']>r['identity_nll_nats']
    assert r['top1']==1 and r['top5']==1
    assert r['raw_bits_per_byte'] is None
    assert r['bits_per_symbol']==pytest.approx(r['nll_nats']/math.log(2))


def test_disjointness_and_score_selection_blocked():
    fitted=fit_temperature(batch())
    with pytest.raises(ValueError,match='scoring labels'):fit_temperature(batch(role='score'))
    with pytest.raises(ValueError,match='disjoint'):score_temperature(batch(role='score'),fitted,codec='mapped128')


@pytest.mark.parametrize('a',[0.,-1.,float('inf'),float('nan')])
def test_invalid_temperature(a):
    with pytest.raises(ValueError):loss_derivatives(batch(),a)


@pytest.mark.parametrize('change',[dict(role='test'),dict(labels=(True,0)),dict(logits=((float('nan'),0.),)*2),dict(documents=('',)*2)])
def test_bad_batch(change):
    values=dict(logits=((2.,0.),)*2,labels=(0,1),documents=('a','a'),role='calibration');values.update(change)
    with pytest.raises(ValueError):fit_temperature(LogitBatch(**values))


def test_byte256_metrics_and_stable_offset():
    b=batch(rows=((1002.,1000.),)*2)
    assert loss_derivatives(b,.8)==pytest.approx(loss_derivatives(batch(),.8))
    b=batch(rows=((2.,)+(0.,)*255,)*2)
    fitted=fit_temperature(b)
    s=batch(rows=b.logits,role='score',doc='other')
    r=score_temperature(s,fitted,codec='byte256')
    assert r['raw_bits_per_byte']==r['bits_per_symbol']


def test_actual_formula_matches_torch_second_derivative():
    torch=pytest.importorskip('torch')
    z=torch.tensor([[2.,-.1,.4],[.2,-.4,.9]],dtype=torch.float64)
    y=torch.tensor([2,0]);a=torch.tensor(.7,dtype=torch.float64,requires_grad=True)
    loss=torch.nn.functional.cross_entropy(a*z,y)
    g=torch.autograd.grad(loss,a,create_graph=True)[0]
    h=torch.autograd.grad(g,a)[0]
    b=batch(rows=z.tolist(),labels=y.tolist())
    assert loss_derivatives(b,.7)==pytest.approx((float(loss.detach()),float(g.detach()),float(h)),abs=1e-12)
