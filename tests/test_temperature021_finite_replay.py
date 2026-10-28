"""Original-runner negative guards; fake logits are not original evidence."""
import importlib.util
from pathlib import Path
import math
import pytest
import torch

path=Path(__file__).resolve().parents[1]/'research/mathematical-engineering/temperature021/replay_original_500k_calibration.py'
spec=importlib.util.spec_from_file_location('temperature021_finite',path)
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_nll_never_passes_tolerance(bad):
    with pytest.raises(ValueError,match='replay FAILED'):r.require_replay_nll(bad,2.5)
    with pytest.raises(ValueError,match='replay FAILED'):r.require_replay_nll(2.5,bad)


@pytest.mark.parametrize('a',[0.,-1.,float('nan'),float('inf')])
def test_nonpositive_nonfinite_temperature(a):
    with pytest.raises(ValueError):r.moments(torch.zeros(2,128,dtype=torch.float64),torch.tensor([0,1]),a)


def test_nonfinite_logits_rejected():
    z=torch.zeros(2,128,dtype=torch.float64);z[0,0]=float('nan')
    with pytest.raises(ValueError):r.fit(z,torch.tensor([0,1]))


@pytest.mark.parametrize('y',[torch.tensor([-1,1]),torch.tensor([128,1]),torch.tensor([0.,1.])])
def test_label_contract(y):
    with pytest.raises(ValueError):r.moments(torch.zeros(2,128,dtype=torch.float64),y,1.)


def test_finite_original_tolerance_and_reference_unchanged():
    r.require_replay_nll(2.5,2.5000001)
    with pytest.raises(ValueError):r.require_replay_nll(2.5,2.51)
    from tac_osm.scalar_calibration import LogitBatch,loss_derivatives
    row=(2.,)+(0.,)*127
    actual=r.moments(torch.tensor([row,row],dtype=torch.float64),torch.tensor([0,1]),.7)
    reference=loss_derivatives(LogitBatch((row,row),(0,1),('fixture','fixture'),'calibration'),.7)
    assert actual==pytest.approx(reference,abs=1e-12)


def test_recursive_replay_comparison_rejects_nonfinite_and_wrong_counts():
    import importlib.util
    helper=Path(__file__).resolve().parents[1]/'scripts/replay_temp021_from_git.py'
    spec=importlib.util.spec_from_file_location('git_replay_compare',helper)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.compare_receipts({'nll':2.5,'targets':3072},{'nll':2.5000001,'targets':3072})
    for actual in ({'nll':float('nan'),'targets':3072},{'nll':2.5,'targets':3071}):
        with pytest.raises(ValueError):m.compare_receipts(actual,{'nll':2.5,'targets':3072})
    with pytest.raises(ValueError):m.compare_receipts({'top1':.5+1e-12},{'top1':.5})
