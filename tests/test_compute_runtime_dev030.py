import importlib.util
from pathlib import Path
import pytest
import torch
from tac_osm.development.compute_runtime import DeviceBoundary
from tac_osm.development.optimization_program import ARMS
from tac_osm.development.schema_maintenance import make_pair

ROOT=Path(__file__).resolve().parents[1]


def load_preflight():
    spec=importlib.util.spec_from_file_location('compute_preflight_fixture',ROOT/'scripts/run_compute_preflight_dev030.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('arm',ARMS)
def test_cpu_adapter_preserves_predictions_gradients_and_parameter_names(arm):
    script=load_preflight();original=script.build(arm,30303)
    adapted=DeviceBoundary(script.build(arm,30303),'cpu')
    inputs,query,receipts,target,_=script.fixture(2,32)
    # Independent calls through the original and adapted public boundaries.
    states=[model.encode(*inputs) for model in (original,adapted)]
    before=[model.decode(state,query) for model,state in zip((original,adapted),states)]
    after=[model.decode(model.update(state,*(x[:,0] for x in receipts)),query)
           for model,state in zip((original,adapted),states)]
    assert torch.equal(before[0],before[1]) and torch.equal(after[0],after[1])
    for b,a in zip(before,after):
        (torch.nn.functional.cross_entropy(b.reshape(-1,2),target.reshape(-1))+
         torch.nn.functional.cross_entropy(a.reshape(-1,2),target.reshape(-1))).backward()
    assert set(original.state_dict())==set(adapted.state_dict())
    for (n,a),(m,b) in zip(original.named_parameters(),adapted.named_parameters()):
        assert n==m
        assert (a.grad is None)==(b.grad is None)
        if a.grad is not None:assert torch.equal(a.grad,b.grad)
    assert adapted.metadata()['state_device']=='cpu'


def test_missing_cuda_fails_instead_of_changing_backend(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    script=load_preflight()
    with pytest.raises(RuntimeError,match='refusing silent CPU fallback'):
        DeviceBoundary(script.build('native',30303),'cuda:0')


def test_cpu_only_evidence_cannot_recommend_gpu():
    script=load_preflight()
    assert script.recommendation({})['prefer_gpu'] is False
    rows={a:dict(cpu={'median_seconds_per_update':1}) for a in ARMS}
    assert script.recommendation(rows)['gpu_validated'] is False
    for row in rows.values():
        row.update(parity={'passed':True},cuda={'median_seconds_per_update':.9})
    assert script.recommendation(rows)['prefer_gpu'] is False
    for row in rows.values():row['cuda']['median_seconds_per_update']=.5
    assert script.recommendation(rows)['prefer_gpu'] is True


def test_engineering_fixtures_do_not_reuse_scientific_splits():
    script=load_preflight();fingerprints=set(script.fixture(2,32)[-1])
    for split in ('train','validation','test'):
        pair=make_pair(303002,32,32,'pending_cancel','at',domain='dev029-v1-'+split)
        assert not fingerprints&{c.fingerprint for c in pair}
