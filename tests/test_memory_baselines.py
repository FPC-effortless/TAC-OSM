"""Causal boundaries and mathematical controls for DEV-018."""
import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from tac_osm.development.memory_baselines import ARMS, MemoryBaseline, GatedDelta
from tac_osm.development.schema_maintenance import make_pair

ROOT = Path(__file__).resolve().parents[1]


def inputs(history=4):
    x = torch.randn(2, history, 16)
    entities = torch.arange(history)[None].expand(2, -1) % 8
    values = torch.zeros(2, history, dtype=torch.long)
    queries = torch.randn(2, 8, 3, 14)
    return x, entities, values, queries


def test_all_arms_public_boundary_and_state_budget():
    x, e, v, q = inputs()
    for arm in ARMS:
        m = MemoryBaseline(arm, 4).eval()
        state = m.encode(x, e, v)
        assert m.decode(state, q).shape == (2, 8, 3, 2)
        if arm not in ("memoryless", "transformer"):
            assert state[0].numel() == m.state_scalars
        assert torch.isfinite(state).all()


def test_native_controls_identical_initial_tensors():
    models = [MemoryBaseline(a, 9) for a in ARMS if a.startswith("tac")]
    first = models[0].state_dict()
    for m in models[1:]:
        assert all(torch.equal(v, first[k]) for k, v in m.state_dict().items())


def test_all_arms_common_encoder_query_head_initialization():
    models = [MemoryBaseline(a, 9) for a in ARMS]
    first = models[0].state_dict()
    for m in models[1:]:
        for k, v in first.items():
            if k.startswith(("event_encoder.", "query_encoder.", "head.")):
                assert torch.equal(v, m.state_dict()[k])


def test_memoryless_cannot_read_earlier_observations():
    x, e, v, q = inputs()
    other = x.clone()
    other[:, :-1] += 100
    m = MemoryBaseline("memoryless").eval()
    assert torch.equal(m.decode(m.encode(x, e, v), q), m.decode(m.encode(other, e, v), q))


def test_schema_only_public_entity_slot_is_updated():
    m = MemoryBaseline("schema41")
    state = torch.zeros(2, 41)
    new = m.update(state, torch.randn(2, 16), torch.tensor([2, 5]), torch.zeros(2, dtype=torch.long))
    for b, entity in enumerate((2, 5)):
        slots = new[b, :40].reshape(8, 5)
        assert slots[entity].abs().sum() > 0
        assert torch.count_nonzero(slots[[e for e in range(8) if e != entity]]) == 0


def test_parameter_intervention_masks():
    for arm in ARMS:
        m = MemoryBaseline(arm)
        assert all(p.requires_grad == (arm not in ("tac_memory", "tac_reader")) for p in m.event_encoder.parameters())
        assert all(p.requires_grad == (arm not in ("schema41_frozen", "tac_encoder", "tac_reader")) for p in m.memory.parameters())


def test_frozen_writer_still_passes_encoder_gradients():
    m = MemoryBaseline("tac_encoder")
    x, e, v, q = inputs()
    m.decode(m.encode(x, e, v), q).square().sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.event_encoder.parameters())
    assert all(p.grad is None for p in m.memory.parameters())


def test_standard_gated_delta_matches_explicit_outer_product():
    m = GatedDelta()
    state, x = torch.randn(2, 4, 8, 24), torch.randn(2, 12)
    key = torch.nn.functional.normalize(m.key(x).reshape(2, 4, 24), dim=-1)
    value = m.value(x).reshape(2, 4, 8)
    alpha, beta = m.gates(x).sigmoid().chunk(2, -1)
    expected = torch.empty_like(state)
    for b in range(2):
        for h in range(4):
            decayed = alpha[b, h] * state[b, h]
            expected[b, h] = decayed + beta[b, h] * torch.outer(value[b, h] - decayed @ key[b, h], key[b, h])
    torch.testing.assert_close(m(state, x), expected)


def test_transformer_stores_and_extends_full_history():
    m = MemoryBaseline("transformer").eval()
    x, e, v, q = inputs()
    state = m.encode(x, e, v)
    assert state.shape == (2, 4, 12) and m.state_scalars is None
    new = m.update(state, x[:, -1], e[:, -1], v[:, -1])
    assert new.shape == (2, 5, 12)
    assert torch.equal(new[:, :4], state)
    assert torch.isfinite(m.decode(new, q)).all()


def test_contract_training_exposure_and_primary_are_explicit():
    spec = json.loads((ROOT / "contracts/DEV-018.json").read_text())
    assert [e["name"] for e in spec["endpoints"] if e["primary"]] == ["H512_mature_pending_future_accuracy"]
    assert {c["history"] * c["batch_size"] for c in spec["regimes"].values()} == {1024}
    assert spec["train_due_delays"] == spec["eval_due_delays"]


def test_pending_pairs_have_same_recent_observation_and_opposite_mature_outcome():
    for h in (32, 128, 512):
        for role in ("pending_cancel", "reissued_cancel"):
            a, b = make_pair(18018, h, 512, role, "at", domain="dev018-eval")
            assert a.observations[-1] == b.observations[-1]
            assert a.wait_targets[a.target_entity] != b.wait_targets[b.target_entity]


def test_auditor_imports_without_torch_dependency():
    path = ROOT / "scripts/audit_memory_controls_dev.py"
    tree = __import__("ast").parse(path.read_text())
    imports = [n.module for n in __import__("ast").walk(tree) if isinstance(n, __import__("ast").ImportFrom)]
    assert not any(n and (n.startswith("torch") or "run_memory" in n) for n in imports)
    spec = importlib.util.spec_from_file_location("dev018_audit_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.interval([.1, .2, .3])
    assert abs(result["mean"] - .2) < 1e-12
    assert result["exploratory_95_t_df2"][0] < 0
