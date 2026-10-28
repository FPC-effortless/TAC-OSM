"""Falsification controls for the correspondence checker, not model evidence."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch

spec=importlib.util.spec_from_file_location('graph_audit',Path(__file__).resolve().parents[1]/'scripts/audit_context027_graph.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

class Fixture(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.weight=torch.nn.Parameter(torch.tensor([.3,.7],dtype=torch.float64))
    def forward(self,x,**kwargs):
        y=x[:,:,None]*self.weight
        return y,{'fixture':y.sum(-1)}

def test_checker_rejects_wrong_forward_and_forward_equal_wrong_backward():
    model=Fixture();x=torch.tensor([[1,2]]);mask=torch.ones_like(x,dtype=torch.bool)
    for kind in ('forward','backward'):
        def serial(w,x,mask):
            y=x[:,:,None]*w['weight']
            if kind=='forward': y=y+1
            else: y=2*y.detach()-y  # identical output, intentionally opposite VJP
            return y,{'fixture':y.sum(-1)}
        r=audit.one_case(model,SimpleNamespace(serial_graph=serial),x,mask,1e-10,1e-8)
        assert r['status']=='FAIL'
        assert ('grad:weight' in r['failed']) if kind=='backward' else ('logits' in r['failed'])

def test_checker_rejects_nonfinite_and_boolean_mismatch():
    import pytest
    with pytest.raises(ValueError,match='nonfinite'):
        audit.comparison(torch.tensor([float('nan')]),torch.zeros(1),1e-10,1e-8)
    assert audit.comparison(torch.tensor([True]),torch.tensor([False]),1e-10,1e-8)['scaled_error']>1
