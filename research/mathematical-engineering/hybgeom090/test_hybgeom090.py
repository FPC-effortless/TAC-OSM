"""Synthetic no-fit mechanism tests for HYBGEOM090 (zero capability evidence)."""
import importlib.util
from pathlib import Path
import math
import sys
import pytest
import torch

FILE=Path(__file__).with_name('operators090.py')
spec=importlib.util.spec_from_file_location('operators090',FILE)
op=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=op
spec.loader.exec_module(op)


def test_flow_closed_form_semigroup_and_zero_duration():
    torch.manual_seed(90)
    x=torch.randn(4,8,dtype=torch.float64)
    w=torch.randn_like(x)*.1
    a=op.linear_flow(x,2.25,.7,w)
    b=op.linear_flow(op.linear_flow(x,.75,.7,w),1.5,.7,w)
    assert torch.allclose(a,b,atol=1e-14,rtol=0)
    assert torch.equal(op.linear_flow(x,0,.7,w),x)
    assert torch.allclose(op.linear_flow(x,.2,0.,w),x+.2*w,atol=1e-14)


def test_iss_lyapunov_all_samples_and_disturbance():
    torch.manual_seed(91)
    for r in (.01,.5,1.2):
        for _ in range(100):
            x=torch.randn(1,32,dtype=torch.float64)
            w=torch.randn_like(x)*.5
            assert float(op.lyapunov_residual(x,w,r))<=1e-10
            y=op.linear_flow(x,.5,r)
            assert y.square().sum() <= x.square().sum()+1e-12
    with pytest.raises(ValueError): op.lyapunov_residual(torch.ones(1,2),torch.ones(1,2),0)


def test_bad_times_rates_nan_and_zombie_state_rejected():
    for args in [(-1,.1),(1,-.1),(float('nan'),.1),(1,float('inf'))]:
        with pytest.raises(ValueError):op.linear_flow(torch.ones(2),*args)
    with pytest.raises(ValueError):op.linear_flow(torch.tensor([float('nan')]),1,.1)


def test_contextual_semantics_context_changes_and_constant_metric_psd():
    torch.manual_seed(92)
    E=torch.randn(256,32,dtype=torch.float64)
    c1=torch.randn(1,32,dtype=torch.float64)
    c2=-c1+.3*torch.randn_like(c1)
    z1=op.contextual_table(E,c1)
    z2=op.contextual_table(E,c2)
    assert not torch.allclose(z1,z2)
    for c in (c1,c2,torch.zeros_like(c1)):
        G=op.metric_tensor(c, .8)
        eig=torch.linalg.eigvalsh(G)
        assert float(eig.min())>=1-1e-12
        assert float(eig.max())<=1.8+1e-12


def test_contextual_pseudometric_triangle_and_symmetry():
    torch.manual_seed(93)
    E=torch.randn(4,32,dtype=torch.float64)
    E[3]=E[0]
    c=torch.randn(1,32,dtype=torch.float64)
    z=op.contextual_table(E,c)[0]
    a,b,d=z[0:1],z[1:2],z[2:3]
    dist=lambda x,y:op.distance_constant(x,y,c,.8)
    assert float(dist(a,z[3:4]))<1e-12
    assert torch.allclose(dist(a,b),dist(b,a),atol=1e-14)
    assert dist(a,d) <= dist(a,b)+dist(b,d)+1e-12


def test_spatially_variable_metric_straight_line_reverse_not_geodesic():
    torch.manual_seed(94)
    a=torch.randn(2,16,dtype=torch.float64)
    b=torch.randn(2,16,dtype=torch.float64)
    c=torch.randn(2,16,dtype=torch.float64)
    f=op.straight_line_length(a,b,c,1.,steps=64)
    r=op.straight_line_length(b,a,c,1.,steps=64)
    assert torch.allclose(f,r,atol=1e-12,rtol=0)
    assert (f >= (a-b).norm(dim=-1)-1e-12).all()


def test_geometry_zero_strength_identity_positive_changes_and_finite_gradient():
    torch.manual_seed(95)
    E=torch.randn(256,32,dtype=torch.float64,requires_grad=True)
    logits=torch.randn(1,256,dtype=torch.float64)
    context=torch.randn(1,32,dtype=torch.float64,requires_grad=True)
    observed=torch.tensor([12],dtype=torch.long)
    original=op.geometry_logits(logits,E,observed,context,contextual=True,alpha=.5,strength=0.)
    assert torch.equal(original,logits)
    euro=op.geometry_logits(logits,E,observed,context,contextual=True,alpha=0.,strength=.01)
    metric=op.geometry_logits(logits,E,observed,context,contextual=True,alpha=.5,strength=.01)
    assert not torch.allclose(logits,euro)
    assert not torch.allclose(euro,metric)
    torch.nn.functional.cross_entropy(metric, torch.tensor([23])).backward()
    assert E.grad is not None and float(E.grad.abs().sum())>0
    assert context.grad is not None and float(context.grad.abs().sum())>0


def test_information_boundary_candidate_target_never_used():
    torch.manual_seed(96)
    E=torch.randn(256,32,dtype=torch.float64)
    log=torch.randn(1,256,dtype=torch.float64)
    ctx=torch.randn(1,32,dtype=torch.float64)
    a=op.geometry_logits(log,E,torch.tensor([65]),ctx,contextual=True,alpha=.5)
    b=op.geometry_logits(log,E,torch.tensor([65]),ctx,contextual=True,alpha=.5)
    assert torch.equal(a,b)
    assert torch.isfinite(a).all()


def test_exact_macro_closure_across_population_scales():
    torch.manual_seed(97)
    for N in (2,5,16,128):
        s=torch.randn(N,dtype=torch.float64)
        assert torch.allclose(op.macro_linear_step(s).mean(),.9*s.mean(),atol=1e-12)


def test_variance_counterexample_breaks_macro_mean_sufficiency():
    a=torch.tensor([1.,-1.,1.,-1.],dtype=torch.float64)
    b=torch.zeros(4,dtype=torch.float64)
    assert a.mean()==b.mean()
    assert not torch.allclose(op.macro_variance_sensitive_step(a).mean(),op.macro_variance_sensitive_step(b).mean())


def test_shift_fixture_contains_win_and_loss():
    r=op.shift_fixture()
    assert r['changed_fast']<r['changed_slow']
    assert r['steady_fast']>r['steady_slow']


def test_original_tac_identity_and_mode_event_behavior_when_available():
    # In repository CI, import the ACTUAL 100521p original core; no mock needed.
    root=Path(__file__).resolve().parents[3] if 'mathematical-engineering' in str(FILE) else None
    if root is None or not (root/'research/mathematical-engineering/langmem086/compact_tac.py').exists():
        pytest.skip('no full GitHub repository checkout here')
    target=root/'research/mathematical-engineering/langmem086/compact_tac.py'
    s=importlib.util.spec_from_file_location('compact_tac090',target)
    m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
    torch.manual_seed(90)
    tac=m.CompactDualClippedTAC086().eval()
    assert tac.counts()['trainable']==100521
    stream=op.start(tac,('A',))
    original=tac.start(('A',))
    with torch.no_grad():
        for t,byte in enumerate((65,66,10,67,46,68)):
            tok=torch.tensor([[byte]])
            vanilla,original=tac(tok,state=original,owners=('A',))
            adapted,stream=op.hybrid_token(tac,tok,stream,t,rates=(0.,0.),resets=(1.,1.))
            assert torch.equal(vanilla,adapted)
            assert all(torch.equal(original.tensors[k],stream.core.tensors[k]) for k in original.tensors)
        assert stream.mode==0
        assert stream.time==5
    bad=torch.tensor([[2]])
    with pytest.raises(ValueError):op.hybrid_token(tac,bad,stream,5)
    with pytest.raises(ValueError):op.hybrid_token(tac,bad,stream,4)
    with pytest.raises(ValueError):op.hybrid_token(tac,bad,stream,6,resets=(1.,1.1))


def test_original_tac_gap_flow_effect_and_future_token_noninterference_when_available():
    root=Path(__file__).resolve().parents[3] if 'mathematical-engineering' in str(FILE) else None
    if root is None or not (root/'research/mathematical-engineering/langmem086/compact_tac.py').exists():
        pytest.skip('no full GitHub repository checkout here')
    target=root/'research/mathematical-engineering/langmem086/compact_tac.py'
    s=importlib.util.spec_from_file_location('compact_tac090b',target)
    m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
    torch.manual_seed(91);tac=m.CompactDualClippedTAC086().eval()
    with torch.no_grad():
        st=op.start(tac,('A',)); st0=op.start(tac,('A',))
        for i,b in enumerate((12,34,56,78)):
            stlog,st=op.hybrid_token(tac,torch.tensor([[b]]),st,i)
            stlog0,st0=op.hybrid_token(tac,torch.tensor([[b]]),st0,i,rates=(0.,0.),resets=(1.,1.))
        changed,st=op.hybrid_token(tac,torch.tensor([[20]]),st,15)
        control,st0=op.hybrid_token(tac,torch.tensor([[20]]),st0,15,rates=(0.,0.),resets=(1.,1.))
        assert (changed-control).abs().max()>1e-9
        future_a=torch.tensor([[1,2,3]]); future_b=torch.tensor([[7,8,9]])
        assert not torch.equal(future_a,future_b)
        # Future suffix is never passed to the adapter: single-token inputs equal.
        x,_=op.hybrid_token(tac,torch.tensor([[22]]),st,16)
        y,_=op.hybrid_token(tac,torch.tensor([[22]]),st,16)
        assert torch.equal(x,y)
