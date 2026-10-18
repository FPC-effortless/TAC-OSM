"""Baseline invariants, usable gradients and causal sealed-data boundaries."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from tac_osm.development.residual_forecast import ResidualForecast,specifications
from data_real_maintenance_002 import split_windows,seconds,open_a,write


@pytest.mark.parametrize('arm',list(specifications()))
def test_zero_head_is_exact_persistence_then_learns_a_change(arm):
    torch.set_num_threads(1);m=ResidualForecast(**specifications()[arm],seed=7).double();x=torch.randn(6,24,8,dtype=torch.float64);x[:,:,0]=4.;assert torch.equal(m(x),x[:,-1]);opt=torch.optim.Adam(m.parameters(),lr=.01)
    for _ in range(3):
        opt.zero_grad();loss=(m(x)-(x[:,-1]+.2)).abs().mean();loss.backward();opt.step()
    assert torch.isfinite(m(x)).all() and not torch.equal(m(x),x[:,-1])
    assert sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)>0


def test_revin_predictor_is_translation_equivariant_after_learning():
    torch.set_num_threads(1);m=ResidualForecast('mlp',True,seed=13).double()
    with torch.no_grad():m.network[-1].weight.normal_(0,.01)
    x=torch.randn(4,24,8,dtype=torch.float64);shift=torch.arange(8,dtype=torch.float64)*10
    torch.testing.assert_close(m(x+shift),m(x)+shift,atol=1e-10,rtol=1e-10)


def test_window_split_causality_and_missing_bins():
    times=np.arange(seconds('2022-05-30'),seconds('2022-07-03'),600);values=np.arange(len(times),dtype=np.float32)[:,None].repeat(8,1);data=split_windows(times,values)
    for d in data.values():assert torch.equal(d['y'][:,0],d['x'][:,-1,0]+1)
    assert data['train']['time'].max()<seconds('2022-06-01')<=data['validation']['time'].min()
    assert data['sealed-a']['time'].max()<seconds('2022-07-01')<=data['sealed-b']['time'].min()
    bad=split_windows(np.delete(times,1000),np.delete(values,1000,axis=0));assert sum(len(v['x']) for v in bad.values())<sum(len(v['x']) for v in data.values())


def test_sealed_a_rejects_incomplete_fit_without_loading_any_payload(tmp_path,monkeypatch):
    write(tmp_path/'summary.json',{'status':'INCOMPLETE','fits':4});write(tmp_path/'selected-arm.json',{'checkpoint_hashes':{}})
    def forbidden(*args,**kwargs):raise RuntimeError('Loaded payload before seal guard')
    monkeypatch.setattr(torch,'load',forbidden)
    with pytest.raises(AssertionError):open_a(tmp_path,tmp_path)


def test_zero_persistence_scale_is_not_floored_into_a_favorable_test_score():
    from run_real_maintenance_002 import relative,evaluate
    assert relative([0.,.1],[0.,0.])==[1.,None]
    m=ResidualForecast().double();x=torch.zeros(4,24,8,dtype=torch.float64);d={'x':x,'y':x[:,-1].clone(),'time':torch.arange(4)};r=evaluate(m,d,torch.zeros(8,dtype=torch.float64))
    assert r['MASE']==[None]*8 and r['all_channels_no_worse']
    with torch.no_grad():m.network.bias.fill_(.1)
    r=evaluate(m,d,torch.zeros(8,dtype=torch.float64));assert r['worst_relative_MAE'] is None and not r['all_channels_no_worse']


def test_analysis_preserves_zero_baseline_ties_without_hiding_positive_error():
    from analyze_real_maintenance_002 import ratio_array
    r=ratio_array(np.array([0.,.01,1.]),np.array([0.,0.,2.]))
    assert r[0]==1. and np.isnan(r[1]) and r[2]==.5
