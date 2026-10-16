"""DEV-023 factor isolation, frozen-corner parity and audited outcome integrity."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from tac_osm.development.entity_aging import ARMS, EntityAgingControl  # noqa: E402
from tac_osm.development.retention_controls import RetentionControl  # noqa: E402
from tac_osm.development.noisy_memory import parameter_hash  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dev023", ROOT / "scripts/run_dev023_entity_aging.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
import audit_dev023_entity_aging as auditor  # noqa: E402
from run_learned_schema_dev import public_queries  # noqa: E402


@pytest.mark.parametrize("arm", ARMS)
def test_initial_values_and_trainable_budget(arm):
    model = EntityAgingControl(arm, 3)
    prior = RetentionControl(arm if arm in ("schema41", "memoryless") else "native", 3)
    assert parameter_hash(model) == parameter_hash(prior)
    if arm not in ("schema41", "memoryless"):
        assert model.state_scalars == 768
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 8674


@pytest.mark.parametrize("arm,prior_arm", [("native_global", "native"), ("masked_local", "entity")])
def test_prior_corners_match_exactly(arm, prior_arm):
    torch.set_num_threads(2)
    current, prior = EntityAgingControl(arm, 8), RetentionControl(prior_arm, 8)
    state, x = torch.randn(2, 4, 8, 24), torch.randn(2, 12)
    entities, values = torch.tensor([1, 7]), torch.tensor([0, 1])
    assert torch.equal(current._update_encoded(state, x, entities, values), prior._update_encoded(state, x, entities, values))
    q = public_queries([512, 512], [128, 128])
    assert torch.equal(current.decode(state, q), prior.decode(state, q))


@pytest.mark.parametrize("arm", ARMS[:4])
def test_decay_scope_is_independent_of_key_mask(arm):
    model = EntityAgingControl(arm, 9)
    with torch.no_grad():
        model.memory.value.weight.zero_()
        model.memory.erase.weight.zero_()
        model.memory.erase.bias.fill_(-1000)
        model.memory.decay.weight.zero_()
        model.memory.decay.bias.fill_(float(torch.logit(torch.tensor(.8))))
    state, x = torch.ones(1, 4, 8, 24), torch.zeros(1, 12)
    result = model._update_encoded(state, x, torch.tensor([2]), torch.tensor([1]))
    expected = state * .8
    if arm.endswith("local"):
        expected = state.clone()
        expected[..., 6:9] = .8
    torch.testing.assert_close(result, expected)


def test_local_aging_does_not_mask_unaddressed_additive_writes():
    native, masked = EntityAgingControl("native_local", 4), EntityAgingControl("masked_local", 4)
    state, x = torch.zeros(1, 4, 8, 24), torch.randn(1, 12)
    a = native._update_encoded(state, x, torch.tensor([2]), torch.tensor([1]))
    b = masked._update_encoded(state, x, torch.tensor([2]), torch.tensor([1]))
    assert a[..., :6].abs().sum() > 0 and a[..., 9:].abs().sum() > 0
    assert b[..., :6].abs().sum() == b[..., 9:].abs().sum() == 0


def test_local_aging_does_not_mask_unaddressed_erase():
    native, masked = EntityAgingControl("native_local", 4), EntityAgingControl("masked_local", 4)
    for model in (native, masked):
        with torch.no_grad():
            model.memory.value.weight.zero_()
            model.memory.decay.weight.zero_()
            model.memory.decay.bias.fill_(1000)
    state, x = torch.randn(1, 4, 8, 24), torch.randn(1, 12)
    a = native._update_encoded(state, x, torch.tensor([2]), torch.tensor([1]))
    b = masked._update_encoded(state, x, torch.tensor([2]), torch.tensor([1]))
    assert not torch.equal(a[..., :6], state[..., :6])
    assert torch.equal(b[..., :6], state[..., :6])


def test_cached_independent_audit_still_rejects_corrupt_outcome_and_target():
    torch.set_num_threads(2)
    spec = runner.specification()
    spec.update(h_levels=[512], eval_steps=1)
    rows, _ = runner.evaluate(EntityAgingControl("memoryless", 23043), 23043, spec, False)
    auditor.validate_rows(rows, 43, "memoryless", spec)
    count = len(auditor.OUTCOMES)
    auditor.validate_rows(rows, 43, "memoryless", spec)
    assert len(auditor.OUTCOMES) == count
    wrong = copy.deepcopy(rows)
    wrong[0]["controls"]["selected"]["total_cost"] += 1
    with pytest.raises(AssertionError):
        auditor.validate_rows(wrong, 43, "memoryless", spec)
    wrong = copy.deepcopy(rows)
    wrong[0]["wait_targets"][0] = 1 - wrong[0]["wait_targets"][0]
    with pytest.raises(AssertionError):
        auditor.validate_rows(wrong, 43, "memoryless", spec)


def test_interval_uses_five_seed_df4_critical_value():
    result = auditor.interval([0., .1, .2, .3, .4])
    assert result["mean"] == pytest.approx(.2)
    assert result["exploratory_95_t_df4"] == pytest.approx([.003675683852, .396324316148])
    with pytest.raises(AssertionError):
        auditor.interval([0., .1, .2])


def test_secondary_local_bundle_cannot_replace_failed_primary_masking():
    spec = auditor.spec_and_sources()
    records = []
    for seed in spec["seeds"]:
        for regime in spec["regimes"]:
            for arm in spec["arms"]:
                accuracy = 1.0 if arm == "masked_local" else .5
                summary = {"primary": accuracy, "swap_primary": 0.}
                for mode in ("selected", "none"):
                    summary.update({mode + "_failures": 0., mode + "_unnecessary_services": 0.,
                                    mode + "_total_cost": .1})
                count = 8674 if arm not in ("schema41", "memoryless") else 1
                records.append({"seed": seed, "regime": regime, "arm": arm, "source_sha256": {},
                    "training": {"initial_common_components": {}, "initial_hash": "same", "fingerprints": []},
                    "model": {"parameters": count, "trainable_parameters": count}, "rows": [], "summary": summary})
    result = auditor.summarize(records, spec)
    for regime in spec["regimes"]:
        out = result["by_regime"][regime]
        assert out["arms"]["masked_local"]["combined_selected_development_gate"]
        assert not out["primary_coordinate_prediction_success"]
        assert not out["primary_coordinate_prediction_and_selected_cost_success"]
    assert not result["official_TAC_overall_success"]
