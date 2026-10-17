"""Numerical boundaries for post-hoc cost/ranking diagnostics."""
import importlib.util
from pathlib import Path
import numpy as np

path=Path(__file__).resolve().parents[1]/'scripts/diagnose_head_cost_dev040a.py'
spec=importlib.util.spec_from_file_location('cost_diagnostics',path)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_auc_gives_half_credit_for_ties():
    assert m.auc([0,1],[.5,.5])==.5
    assert m.auc([0,0,1,1],[0,.5,.5,1])==.875


def test_auc_is_undefined_for_one_class():
    assert m.auc([0,0],[.2,.9]) is None
    assert m.auc([1,1],[.2,.9]) is None


def test_cost_uses_service_counterfactual_and_strict_threshold():
    y=np.array([[[1,0,1],[0,0,0]]])
    z=m.physical(y,np.array([[.25,.8]]),.25)
    assert (z['failures'],z['services'],z['unnecessary_services'],z['total_cost'])==(1,1,1,5)


def test_exact_frontier_keeps_tied_decisions_together():
    y=np.array([[[1,0,1],[0,0,0],[1,0,1]]])
    score=np.array([[.2,.2,.8]])
    exact=m.exact_frontier(y,score)
    brute=[m.physical(y,score,t) for t in (1.,.5,-1.)]
    assert exact.tolist()==[[z['failures'],z['unnecessary_services']] for z in brute]
    assert m.weakly_covers(exact,exact)
    assert not m.weakly_covers(np.array([[2,0]]),np.array([[1,0]]))


def test_zero_weight_excludes_focus_and_other_weights_preserve_mass():
    y=np.zeros((2,8,3),dtype=int)
    cs=[dict(deadline='at',target_entity=0)]*2
    zero=m.loss_weights(y,cs,0.)
    assert zero[:,0,0].sum()==0 and np.isclose(zero.sum(),1.)
    w=m.loss_weights(y,cs,.1)
    assert np.isclose(w[:,0,0].sum(),.1) and np.isclose(w.sum(),1.)


def test_last_event_table_predictions_use_no_earlier_event():
    from types import SimpleNamespace
    obs=[SimpleNamespace(entity=e,kind='change',value=e%2,due=-1) for e in range(8)]
    first=SimpleNamespace(observations=obs,tick=512,horizon=32)
    extra=SimpleNamespace(entity=0,kind='scheduled',value=0,due=999)
    second=SimpleNamespace(observations=[extra]+obs,tick=512,horizon=32)
    assert m.last_keys(first)==m.last_keys(second)
