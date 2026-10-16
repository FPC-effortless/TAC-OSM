"""DEV-024 matched initialization, freeze gradients and attribution integrity."""
import copy
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from tac_osm.development.writer_attribution import ARMS, WriterAttributionControl, component_active  # noqa: E402
from tac_osm.development.entity_aging import EntityAgingControl  # noqa: E402
from tac_osm.development.noisy_memory import parameter_hash  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dev024", ROOT / "scripts/run_dev024_writer_attribution.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
import audit_dev024_writer_attribution as auditor  # noqa: E402
from run_learned_schema_dev import public_queries  # noqa: E402


@pytest.mark.parametrize("arm", ARMS)
def test_identical_random_initialization_and_registered_budget(arm):
    model = WriterAttributionControl(arm, 9)
    reference = EntityAgingControl("schema41" if arm.startswith("schema41") else "masked_local", 9)
    assert parameter_hash(model) == parameter_hash(reference)
    contract = auditor.spec_and_sources()
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == contract["active_parameters"][arm]
    for prefix in ("event_encoder.", "memory."):
        assert all(p.requires_grad == component_active(arm, prefix) for n, p in model.named_parameters() if n.startswith(prefix))


@pytest.mark.parametrize("arm", ARMS)
def test_freeze_does_not_change_forward_and_preserves_input_gradients(arm):
    torch.set_num_threads(2)
    model = WriterAttributionControl(arm, 8)
    reference = EntityAgingControl("schema41" if arm.startswith("schema41") else "masked_local", 8)
    # A short public sequence exercises gradients THROUGH a frozen writer.
    events = torch.randn(2, 5, 16)
    entities, values = torch.zeros(2, 5, dtype=torch.long), torch.ones(2, 5, dtype=torch.long)
    state = model.encode(events, entities, values)
    q = public_queries([32, 32], [32, 32])
    logits = model.decode(state, q)
    assert torch.equal(logits, reference.decode(reference.encode(events, entities, values), q))
    logits.square().sum().backward()
    for prefix in ("event_encoder.", "memory."):
        grads = [p.grad for n, p in model.named_parameters() if n.startswith(prefix)]
        assert any(g is not None and g.abs().sum() > 0 for g in grads) == component_active(arm, prefix)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.head.parameters())


def test_audit_rejects_cached_cost_and_target_corruption():
    torch.set_num_threads(2)
    spec = runner.specification()
    spec.update(h_levels=[512], eval_steps=1)
    rows, _ = runner.evaluate(WriterAttributionControl("both_frozen", 24043), 24043, spec, False)
    auditor.validate_rows(rows, 43, "both_frozen", spec)
    for field in ("cost", "target"):
        wrong = copy.deepcopy(rows)
        if field == "cost":
            wrong[0]["controls"]["selected"]["total_cost"] += 1
        else:
            wrong[0]["wait_targets"][0] ^= 1
        with pytest.raises(AssertionError):
            auditor.validate_rows(wrong, 43, "both_frozen", spec)


def test_second_writer_contrast_cannot_be_replaced_by_joint_success():
    spec = auditor.spec_and_sources()
    records = []
    for seed in spec["seeds"]:
        for regime in spec["regimes"]:
            for arm in spec["arms"]:
                accuracy = .5 if arm == "frozen_writer" else 1.
                summary = {"primary": accuracy, "swap_primary": 0.}
                for mode in ("selected", "none"):
                    summary.update({mode + "_failures": 0., mode + "_unnecessary_services": 0., mode + "_total_cost": .1})
                records.append({"seed": seed, "regime": regime, "arm": arm, "source_sha256": {},
                    "training": {"initial_common_components": {}, "initial_hash": "same", "fingerprints": []},
                    "model": {"parameters": 8674, "trainable_parameters": spec["active_parameters"][arm]}, "rows": [], "summary": summary})
    result = auditor.summarize(records, spec)
    for out in result["by_regime"].values():
        assert out["arms"]["joint"]["combined_selected_development_gate"]
        assert not out["writer_attribution_prediction_success"]
        assert not out["writer_attribution_prediction_and_selected_cost_success"]
    assert not result["official_TAC_overall_success"]
