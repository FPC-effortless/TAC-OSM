"""Causal time windows, unchanged evidence and trainable raw-sensor controls."""
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_real_maintenance_001 import windows,seconds,loss
from tac_osm.development.maintenance_forecast import MaintenanceForecast,configurations


def test_trailing_context_does_not_contain_target_or_impute_outages():
    times=np.arange(seconds('2020-02-21 12:00:00'),seconds('2020-02-23'),600);values=np.arange(len(times),dtype=np.float32)[:,None].repeat(15,1);d=windows(times,values)
    for split in d.values():
        if len(split['x']):
            assert torch.equal(split['x'][:,-1,0]+1,split['y'][:,0])
            assert torch.equal(split['x'][:,0,0]+24,split['y'][:,0])
    assert d['train']['time'].max()<seconds('2020-02-22')<=d['validation']['time'].min()
    bad=windows(np.delete(times,30),np.delete(values,30,axis=0))
    assert sum(len(v['x']) for v in bad.values())<sum(len(v['x']) for v in d.values())


@pytest.mark.parametrize('arm',['tac','no_casm','gru','mlp'])
def test_all_retrained_forecasters_are_parameter_matched_and_learn(arm):
    torch.set_num_threads(1);specs=configurations();s=specs[arm];m=MaintenanceForecast(arm,s['width'],s['aux'],41)
    value=loss(m,torch.randn(8,24,15),torch.randn(8,15));value.backward()
    assert torch.isfinite(value) and m.future[-1].weight.grad.abs().sum()>0
    assert abs(specs[arm]['parameters']/specs['tac']['parameters']-1)<=.05
    assert set(m.state_dict())>={'mean','scale'}


def test_raw_sensor_model_never_imports_teacher_or_fault_table():
    import ast
    root=Path(__file__).resolve().parents[1];tree=ast.parse((root/'src/tac_osm/development/maintenance_forecast.py').read_text())
    imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]+[a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
    assert all(name and name.split('.')[0]=='torch' for name in imports)
