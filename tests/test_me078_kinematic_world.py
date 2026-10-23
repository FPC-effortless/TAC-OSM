"""Numerical/software fixtures; not physical learned-skill evidence."""
import json
from pathlib import Path
import pytest
import torch
from tac_osm.variants.kinematic_world import KinematicWorld,GRUCore,dense_loss
from scripts.me078_train_world import integrity
from scripts.me078_operation_accounting import OperationLedger


def test_actual_whole_model_counts_and_frozen_prefit_matching():
    m=integrity()
    assert len(m['arms'])==15
    counts=[]
    for arm in ('native','gru','transformer'):
        model=KinematicWorld(arm);n=sum(p.numel() for p in model.parameters());counts.append(n)
        assert 95000<=n<=105000
        assert all(r['trainable_parameters']==n for r in m['arms'] if r['architecture']==arm)
    assert max(counts)/min(counts)-1<=.01


@pytest.mark.parametrize('arm',['native','gru','transformer'])
def test_causal_prefixes_ordered_quantiles_and_finite_gradients(arm):
    torch.manual_seed(3);model=KinematicWorld(arm)
    x=torch.randn(2,8,14);a=torch.randn_like(x)
    pred=model(x,a)
    altered=x.clone();altered[:,4:]+=100
    changed=model(altered,a)
    for key in pred:
        torch.testing.assert_close(pred[key][:,:4],changed[key][:,:4],rtol=1e-5,atol=1e-6)
    assert (pred['lower']<=pred['median']).all() and (pred['median']<=pred['upper']).all()
    targets=torch.randn(2,8,3,14)
    loss=dense_loss(pred,targets,x,torch.ones(2,8,3,dtype=torch.bool),torch.ones(2),model.state_scale,.07)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_gru_equation_matches_independent_pytorch_cell():
    torch.manual_seed(9);core=GRUCore();cell=torch.nn.GRUCell(64,56)
    with torch.no_grad():
        cell.weight_ih.copy_(core.input.weight);cell.bias_ih.copy_(core.input.bias)
        cell.weight_hh.copy_(core.hidden.weight);cell.bias_hh.copy_(core.hidden.bias)
    x=torch.randn(3,8,64);h=torch.zeros(3,56);out=[]
    for t in range(8):h=cell(x[:,t],h);out.append(core.project(h))
    torch.testing.assert_close(core(x),torch.stack(out,1),rtol=1e-5,atol=1e-6)


def test_operation_ledger_counts_matrix_bias_and_elementwise_without_ignoring_unknowns():
    x=torch.randn(2,3);w=torch.randn(3,4);b=torch.randn(4)
    with OperationLedger() as ledger:
        y=torch.addmm(b,x,w);z=y*y
    assert ledger.counts['addmm']==2*2*3*4+8
    assert ledger.counts['mul']==8
    assert not ledger.unsupported
    with OperationLedger() as other:torch.erf(x)
    assert other.unsupported


def test_exact_importance_weight_recovers_natural_motion_prevalence():
    p=.08
    q_m=.5*p+.5;q_s=.5*(1-p)
    w_m=1/(.5+.5/p);w_s=2.
    assert abs(q_m*w_m-p)<1e-12
    assert abs(q_m*w_m+q_s*w_s-1)<1e-12


@pytest.mark.parametrize('scores,labels,auc,ap',[
    ([.9,.8,.2,.1],[1,1,0,0],1.,1.),
    ([.1,.2,.8,.9],[1,1,0,0],0.,5/12),
    ([.5,.5],[1,0],.5,.5),
])
def test_motion_ranking_has_correct_tie_and_direction_behavior(scores,labels,auc,ap):
    from scripts.me078_report_capability import ranking
    result=ranking([{'moving':bool(y),'motion_probability':p} for p,y in zip(scores,labels)])
    assert abs(result['auc']-auc)<1e-12
    assert abs(result['ap']-ap)<1e-12
