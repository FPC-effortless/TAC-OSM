"""Constructed accounting/failure regressions; no research fits or DEV access."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

STUDY = Path(__file__).resolve().parents[1]/'research/mathematical-engineering/langmath101'

def load(name):
    spec = importlib.util.spec_from_file_location('accounting_test_'+name, STUDY/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def row(name, inputs=None, outputs=None):
    return {'stage': 'test', 'signature': json.dumps([name, inputs or [], outputs or []]), 'calls': 1}

def test_actual_adamw_first_and_warm_steps_preserve_weights_and_dense_charge():
    torch.manual_seed(7)
    m = torch.nn.Linear(3, 2)
    control = torch.nn.Linear(3, 2); control.load_state_dict(m.state_dict())
    opt = torch.optim.AdamW(m.parameters(), foreach=False)
    other = torch.optim.AdamW(control.parameters(), foreach=False)
    meter, gate = load('meter'), load('owner_two_lane')
    for _ in range(2):
        opt.zero_grad(set_to_none=True); other.zero_grad(set_to_none=True)
        with meter.ActualTrainingLedger() as ledger:
            m(torch.ones(4, 3)).square().mean().backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1., error_if_nonfinite=True)
            opt.step()
        control(torch.ones(4, 3)).square().mean().backward()
        torch.nn.utils.clip_grad_norm_(control.parameters(), 1., error_if_nonfinite=True)
        other.step()
        assert all(torch.equal(a, b) for a, b in zip(m.parameters(), control.parameters()))
        for key, slot in opt.state_dict()['state'].items():
            assert all(torch.equal(v, other.state_dict()['state'][key][k]) for k, v in slot.items())
        aten = [r for r in ledger.operators if json.loads(r['signature'])[0].startswith('aten::')]
        markers = [r for r in ledger.operators if r not in aten]
        assert {json.loads(r['signature'])[0] for r in markers} == {
            'profiler::_record_function_enter_new', 'profiler::_record_function_exit'}
        assert ledger.receipt()['counted_dense_units'] == gate.counted_dense_units(aten)['counted_dense_units']
        assert ledger.receipt()['operator_sha256'] == gate.digest(ledger.operators)
        assert ledger.receipt()['operator_sha256'] != gate.digest(aten)

@pytest.mark.parametrize('bad', [row('custom::hidden_matmul'),
    row('profiler::_record_function_exit', [['torch.float32', [3, 3]]])])
def test_unknown_or_tensor_bearing_markers_fail_closed(bad):
    gate = load('owner_two_lane')
    dense = row('aten::mm', [['torch.float32', [2, 3]], ['torch.float32', [3, 2]]])
    with pytest.raises(ValueError): gate.counted_dense_units([dense, bad])

def test_failure_after_optimizer_before_ledger_preserves_actual_update(tmp_path, monkeypatch):
    meter, failure = load('meter'), load('failure')
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.AdamW(model.parameters(), foreach=False)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    completed = 0
    def reject(_): raise ValueError('injected accounting failure')
    monkeypatch.setattr(meter._owner(), 'counted_dense_units', reject)
    try:
        with meter.ActualTrainingLedger():
            model(torch.ones(4, 3)).sum().backward(); opt.step(); completed += 1
    except ValueError as error:
        def checkpoint(path, m, o, metadata):
            torch.save({'model': m.state_dict(), 'optimizer': o.state_dict(), 'metadata': metadata}, path)
            return hashlib.sha256(path.read_bytes()).hexdigest()
        C = SimpleNamespace(checkpoint=checkpoint, state_digest=lambda m: 'test_digest',
            save=lambda p, v: p.write_text(json.dumps(v)))
        failure.preserve(C, tmp_path, model, opt, {}, [], completed, 1, 0.1, error)
    receipt = json.loads((tmp_path/'FAILURE.json').read_text())
    saved = torch.load(tmp_path/'FAILED_MODEL.pt', weights_only=True)
    assert receipt['actual_steps'] == receipt['optimizer_steps'] == 1
    assert receipt['completed_ledger_entries'] == 0
    assert set(receipt['optimizer_slot_steps'].values()) == {1}
    assert saved['metadata']['stage'] == 'FAILED_RETAINED_NOT_EVALUABLE'
    assert all(torch.equal(v, saved['model'][k]) for k, v in model.state_dict().items())
    assert any(not torch.equal(before[k], v) for k, v in saved['model'].items())
