"""DEV-021 synthetic integrity checks; not trained-model research outcomes."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from tac_osm.development.noisy_memory import DeltaMemory, MemoryConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
module = importlib.util.spec_from_file_location("dev021", ROOT / "scripts/run_dev021_suffix_transport.py")
dev = importlib.util.module_from_spec(module)
module.loader.exec_module(dev)


def test_exact_trained_linear_difference_identity_and_no_additive_write():
    model = DeltaMemory(MemoryConfig()).double().eval()
    a = torch.randn(3, 4, 8, 24, dtype=torch.float64)
    b = torch.randn(3, 4, 8, 24, dtype=torch.float64)
    x = torch.randn(3, 12, dtype=torch.float64)
    value = torch.tensor([0, 1, 0])
    expected = model.update(a, x, value) - model.update(b, x, value)
    observed, decay = dev.suffix_difference_step(model, a - b, x, value)
    torch.testing.assert_close(observed, expected, rtol=1e-12, atol=1e-12)
    assert ((decay > 0) & (decay < 1)).all()


def test_two_by_two_counterfactual_factorial_interventions():
    memory = DeltaMemory(MemoryConfig()).double().eval()
    d = torch.randn(2, 4, 8, 24, dtype=torch.float64)
    x = torch.randn(2, 12, dtype=torch.float64)
    v = torch.tensor([0, 1])
    normal, actual_decay = dev.suffix_difference_step(memory, d, x, v, True, True)
    no_erase, _ = dev.suffix_difference_step(memory, d, x, v, True, False)
    no_decay, _ = dev.suffix_difference_step(memory, d, x, v, False, True)
    neither, _ = dev.suffix_difference_step(memory, d, x, v, False, False)
    torch.testing.assert_close(no_erase, d * actual_decay[:, :, None, None])
    torch.testing.assert_close(neither, d)
    torch.testing.assert_close(normal - no_erase, no_decay - neither, rtol=1e-12, atol=1e-12)


def test_public_suffixes_identical_after_true_last_difference():
    groups = list(dev.fresh_groups(43, 1))
    assert len(groups) == 12
    for due, deadline, role, pairs, pivot in groups:
        assert 0 <= pivot < 31
        a, b = pairs[0]
        assert a.observations[pivot] != b.observations[pivot]
        assert a.observations[pivot+1:] == b.observations[pivot+1:]
        assert a.wait_targets[a.target_entity] != b.wait_targets[b.target_entity]


def test_equality_of_model_visible_suffix_not_only_raw_ticks():
    from run_learned_schema_dev import public_stream_tensors
    _, _, _, pairs, pivot = next(dev.fresh_groups(59, 1))
    a, b = pairs[0]
    features, entities, values = public_stream_tensors([a.observations, b.observations])
    torch.testing.assert_close(features[0, pivot+1:], features[1, pivot+1:])
    assert torch.equal(entities[0, pivot+1:], entities[1, pivot+1:])
    assert torch.equal(values[0, pivot+1:], values[1, pivot+1:])


def test_audit_rejects_incomplete_or_mislabeled_grid():
    spec = importlib.util.spec_from_file_location("dev021audit", ROOT / "scripts/audit_dev021_suffix_transport.py")
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    with pytest.raises(AssertionError):
        auditor.aggregate([])
    with pytest.raises(AssertionError):
        auditor.audit_one({"study": "DEV-021", "smoke": False, "status": "RELIABLE_WORLD_MODEL"})
