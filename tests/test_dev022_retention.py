"""DEV-022 boundaries and interventions, independent of favorable outcomes."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from tac_osm.development.memory_baselines import MemoryBaseline  # noqa: E402
from tac_osm.development.retention_controls import ARMS, RetentionControl  # noqa: E402
from tac_osm.development.noisy_memory import parameter_hash  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dev022", ROOT / "scripts/run_dev022_retention.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
import audit_dev022_retention as auditor  # noqa: E402
from run_learned_schema_dev import public_queries  # noqa: E402


@pytest.mark.parametrize("arm", ARMS)
def test_registered_models_share_initialization_and_state_budget(arm):
    model = RetentionControl(arm, 9)
    reference = MemoryBaseline(model.arm, 9)
    assert parameter_hash(model) == parameter_hash(reference)
    assert model.state_scalars == (41 if arm == "schema41" else 0 if arm == "memoryless" else 768)
    if model.arm == "tac_joint":
        assert all(p.requires_grad != model.disable_decay for p in model.memory.decay.parameters())
        assert all(p.requires_grad != model.disable_erase for p in model.memory.erase.parameters())
        assert all(p.requires_grad != model.aligned for p in model.read_key.parameters())


def test_native_exact_update_and_decode_parity():
    torch.set_num_threads(2)
    native = RetentionControl("native", 3)
    original = MemoryBaseline("tac_joint", 3)
    state, x = torch.randn(2, 4, 8, 24), torch.randn(2, 12)
    entities, values = torch.tensor([1, 7]), torch.tensor([0, 1])
    assert torch.equal(native._update_encoded(state, x, entities, values), original._update_encoded(state, x, entities, values))
    queries = public_queries([32, 32], [32, 32])
    assert torch.equal(native.decode(state, queries), original.decode(state, queries))


def test_neither_has_no_contraction_for_identical_event():
    model = RetentionControl("neither", 2)
    state = torch.randn(2, 4, 8, 24)
    x = torch.randn(1, 12).expand(2, -1)
    result = model._update_encoded(state, x, torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.long))
    torch.testing.assert_close(result[0] - result[1], state[0] - state[1])


def test_entity_update_and_read_isolate_public_entity_slices():
    model = RetentionControl("entity", 5)
    state = torch.randn(1, 4, 8, 24)
    x = torch.randn(1, 12)
    result = model._update_encoded(state, x, torch.tensor([2]), torch.tensor([1]))
    assert torch.equal(result[..., :6], state[..., :6])
    assert torch.equal(result[..., 9:], state[..., 9:])
    queries = public_queries([32], [32])
    altered = state.clone()
    altered[..., 21:] += 10
    assert torch.equal(model.decode(state, queries)[:, 2], model.decode(altered, queries)[:, 2])


def test_aligned_reader_backpropagates_through_writer_key_only():
    model = RetentionControl("aligned", 7)
    model.decode(torch.randn(2, 4, 8, 24), public_queries([32, 32], [32, 32])).sum().backward()
    assert model.memory.key.weight.grad.abs().sum() > 0
    assert model.read_key.weight.grad is None


def test_independent_audit_rejects_corrupted_physical_outcome_and_logits():
    torch.set_num_threads(2)
    spec = runner.specification()
    spec.update(h_levels=[512], eval_steps=1)
    rows, _ = runner.evaluate(RetentionControl("memoryless", 22043), 22043, spec, False)
    auditor.validate_rows(rows, 43, "memoryless", spec)
    wrong = copy.deepcopy(rows)
    wrong[0]["controls"]["selected"]["total_cost"] += 1
    with pytest.raises(AssertionError):
        auditor.validate_rows(wrong, 43, "memoryless", spec)
    wrong = copy.deepcopy(rows)
    wrong[0]["logits"][0][0] = [float("nan"), 0.]
    with pytest.raises(AssertionError):
        auditor.validate_rows(wrong, 43, "memoryless", spec)


def test_supplementary_success_cannot_replace_failed_primary_retention():
    spec = auditor.spec_and_sources()
    records = []
    for seed in spec["seeds"]:
        for regime in spec["regimes"]:
            for arm in spec["arms"]:
                accuracy = 1.0 if arm == "entity_no_decay" else .5
                records.append({"seed": seed, "regime": regime, "arm": arm, "source_sha256": {},
                    "training": {"before_components": {"event_encoder.": "same"},
                                 "initial_common_components": {}, "initial_hash": "same", "fingerprints": []},
                    "model": {"parameters": 1, "trainable_parameters": 1}, "rows": [],
                    "summary": {"primary": accuracy, "swap_primary": 0., "selected_failures": 0.,
                                "selected_unnecessary_services": 0., "selected_total_cost": .1,
                                "none_total_cost": .2}})
    result = auditor.summarize(records, spec)
    for regime in spec["regimes"]:
        out = result["by_regime"][regime]
        assert out["arms"]["entity_no_decay"]["combined_development_gate"]
        assert not out["primary_retention_success"]
    assert not result["official_TAC_overall_success"]
