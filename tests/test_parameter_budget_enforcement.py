"""Software fixtures only: no research data, model capability or result claim."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import torch
from torch import nn
import pytest
from tac_osm.parameter_budget import (
    ParameterBudgetError,
    budgeted_optimizer,
    checked_optimizer_step,
    save_training_checkpoint,
    file_sha,
    live_inventory,
)
from scripts.validate_research_parity import check_manifest
from scripts.audit_training_budget import audit, inspect_file


class Toy(nn.Module):
    def __init__(self, n=100000, frozen=0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))
        if frozen:
            self.frozen = nn.Parameter(torch.ones(frozen), requires_grad=False)


def single(n=100000, deployed=None):
    return {
        "schema_version": "tac-osm.parity.v1",
        "study_id": "ME001_REUSED_ID_SOFTWARE_FIXTURE",
        "study_type": "single_model_training",
        "registered_before_first_fit": True,
        "real_world_source": {
            "name": "SOFTWARE FIXTURE ONLY",
            "observed_real_outcomes": True,
            "sha256": "a" * 64,
            "license": "Software-only declared fixture",
        },
        "models": [
            {
                "seed": 1,
                "trainable_parameters": n,
                "deployed_parameters": n if deployed is None else deployed,
            }
        ],
    }


def manifest_for(kind, n):
    m = single(n)
    m["study_id"] = "ME074_ENFORCEMENT_SOFTWARE_FIXTURE"
    if kind == "single":
        return m
    if kind == "comparison":
        from tests.test_research_parity_gate import fixture

        m = fixture()
        for row in m["arms"]:
            row.update(trainable_parameters=n, deployed_parameters=n)
        return m
    if kind == "capability":
        m.update(
            study_type="capability_development",
            no_architecture_superiority_claim=True,
            scientific_purpose="Software-only capability validation fixture",
            architecture_count=1,
            seeds=[1, 2],
            trainable_parameters=n,
            deployed_parameters=n,
            training_budget={
                "epochs": 1,
                "batch_size": 1,
                "backprop_events": 1,
                "expected_targets_per_seed": 1,
            },
            repository_storage_plan="Software-only fixture storage plan, no actual study",
            split_sha256={k: "b" * 64 for k in ("FIT", "TEMP", "CQR", "REPORT")},
        )
        m.pop("models")
        return m
    if kind == "deferred":
        return {
            "study_id": "ME066_REUSED_SOURCE_DEFERRED",
            "study_type": "architecture_comparison",
            "source_based_deferred_registration": True,
            "registered_before_first_fit": True,
            "dataset_fit_receipt_sha256": "a" * 64,
            "frozen_episode_split_sha256": {
                k: "b" * 64 for k in ("FIT", "DEV", "TEST")
            },
            "seeds": [1, 2],
            "parameter_counts": {"native_tac": n, "gru_world": n},
            "deployed_parameter_counts": {"native_tac": n, "gru_world": n},
            "training_budget": {
                "updates": 1,
                "batch_size": 1,
                "history_frames": 1,
                "backprop_frames": 1,
                "tuning_trials": 0,
            },
            "same_authentic_training_data": True,
            "same_real_training_sample_order_within_each_seed": True,
            "original_test_unopened_until_all_ten_checkpoints_frozen": True,
            "cpu_time_matching_required": False,
        }
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", ["single", "comparison", "capability", "deferred"])
@pytest.mark.parametrize("n", [40000, 94999, 105001, 150000])
@pytest.mark.parametrize("phase", ["preflight", "final"])
def test_every_fitting_path_rejects_out_of_band_counts(kind, n, phase):
    with pytest.raises(ParameterBudgetError, match="100k target"):
        check_manifest(manifest_for(kind, n), phase)


@pytest.mark.parametrize("kind", ["single", "comparison", "capability", "deferred"])
def test_supported_paths_accept_declared_in_band_but_declarations_do_not_prove_fit(
    kind,
):
    m = manifest_for(kind, 100000)
    assert (
        check_manifest(m, "preflight")["eligible_for_architecture_superiority"] is False
    )
    with pytest.raises((ParameterBudgetError, KeyError)):
        check_manifest(m, "final")


def configure(tmp_path, model=None, declared=100000):
    model = Toy() if model is None else model
    path = tmp_path / "prefit.json"
    path.write_text(json.dumps(single(declared)))
    receipt = tmp_path / "runtime.json"
    opt = budgeted_optimizer(
        model, torch.optim.SGD, lr=0.1, budget_manifest=path, budget_receipt=receipt
    )
    return model, opt, path, receipt


def completed(tmp_path):
    model, opt, path, receipt = configure(tmp_path)
    model.weight.sum().backward()
    checked_optimizer_step(opt)
    checkpoint = tmp_path / "original.pt"
    save_training_checkpoint(
        {"model": model.state_dict(), "optimizer": opt.state_dict()}, checkpoint
    )
    m = single()
    m["prefit_registration_sha256"] = file_sha(path)
    m["models"][0]["measured"] = {
        "trainable_parameters": 100000,
        "deployed_parameters": 100000,
        "checkpoint_sha256": file_sha(checkpoint),
        "runtime_evidence": {
            "receipt_path": str(receipt),
            "receipt_sha256": file_sha(receipt),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": file_sha(checkpoint),
        },
    }
    return model, opt, m, receipt, checkpoint


def test_live_model_overrules_a_lying_compliant_manifest_before_any_update(tmp_path):
    model, opt, _, receipt = configure(tmp_path, Toy(40000))
    before = model.weight.detach().clone()
    model.weight.sum().backward()
    with pytest.raises(ParameterBudgetError, match="100k target"):
        checked_optimizer_step(opt)
    torch.testing.assert_close(model.weight, before, atol=0, rtol=0)
    assert not receipt.exists() and not opt.state


def test_hidden_frozen_backbone_is_included_in_deployed_count(tmp_path):
    model, opt, _, _ = configure(tmp_path, Toy(100000, 20000))
    with pytest.raises(ParameterBudgetError, match="deployed"):
        checked_optimizer_step(opt)


def test_receipt_is_durable_before_first_optimizer_update_and_final_tensors_verify(
    tmp_path,
):
    model, opt, path, receipt = configure(tmp_path)
    before = model.weight.detach().clone()
    seen = []

    def observe(optimizer, args, kwargs):
        value = json.loads(receipt.read_text())
        seen.append(value["preflight"]["optimizer_steps_before_check"])
        torch.testing.assert_close(model.weight, before, atol=0, rtol=0)

    opt.register_step_pre_hook(observe)
    model.weight.sum().backward()
    checked_optimizer_step(opt)
    assert seen == [0]
    checkpoint = tmp_path / "model.pt"
    save_training_checkpoint({"model": model.state_dict()}, checkpoint)
    m = single()
    m["prefit_registration_sha256"] = file_sha(path)
    m["models"][0]["measured"] = {
        "trainable_parameters": 100000,
        "deployed_parameters": 100000,
        "checkpoint_sha256": file_sha(checkpoint),
        "runtime_evidence": {
            "receipt_path": str(receipt),
            "receipt_sha256": file_sha(receipt),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": file_sha(checkpoint),
        },
    }
    assert check_manifest(m, "final")["status"] == "PASS_SINGLE_MODEL"


def test_mutating_topology_after_guard_binding_blocks_the_next_update(tmp_path):
    model, opt, _, receipt = configure(tmp_path)
    model.weight.sum().backward()
    checked_optimizer_step(opt)
    model.extra = nn.Parameter(torch.ones(1))
    before = model.weight.detach().clone()
    with pytest.raises(ParameterBudgetError, match="whole model|inventory"):
        checked_optimizer_step(opt)
    torch.testing.assert_close(model.weight, before, atol=0, rtol=0)
    assert json.loads(receipt.read_text())["successful_optimizer_steps"] == 1


def test_optimizer_cannot_bind_a_different_model_or_omit_a_trainable_head(tmp_path):
    model = Toy()
    foreign = Toy()
    model, opt, _, _ = configure(tmp_path, model)
    opt.param_groups[0]["params"] = [foreign.weight]
    with pytest.raises(ParameterBudgetError, match="whole model"):
        checked_optimizer_step(opt)


def test_old_optimizer_restore_is_allowed_but_new_update_is_never_exempt(tmp_path):
    model = Toy(40000)
    opt = budgeted_optimizer(model, torch.optim.SGD, lr=0.1)
    restored = deepcopy(opt.state_dict())
    opt.load_state_dict(restored)
    with pytest.raises(ParameterBudgetError, match="100k target"):
        opt.step()


def test_shared_parameters_are_counted_once(tmp_path):
    model = Toy()
    model.alias = model.weight
    counts, _ = live_inventory(model)
    assert counts == {"trainable_parameters": 100000, "deployed_parameters": 100000}
    model, opt, _, _ = configure(tmp_path, model)
    model.weight.sum().backward()
    checked_optimizer_step(opt)


def test_postfit_declared_numbers_and_a_sha_without_runtime_proof_are_insufficient():
    m = single()
    m["models"][0]["measured"] = {
        "trainable_parameters": 100000,
        "deployed_parameters": 100000,
        "checkpoint_sha256": "b" * 64,
    }
    with pytest.raises(ParameterBudgetError, match="runtime evidence"):
        check_manifest(m, "final")


def test_actual_checkpoint_corruption_cannot_be_hidden_by_declared_counts(tmp_path):
    _, _, m, _, checkpoint = completed(tmp_path)
    checkpoint.write_bytes(b"wrong original bytes")
    with pytest.raises(ParameterBudgetError, match="checkpoint SHA"):
        check_manifest(m, "final")


def test_runtime_proof_is_bound_to_original_prefit_manifest(tmp_path):
    _, _, m, _, _ = completed(tmp_path)
    m["prefit_registration_sha256"] = "f" * 64
    with pytest.raises(ParameterBudgetError, match="preregistration"):
        check_manifest(m, "final")


def test_supported_training_entrypoints_have_no_raw_optimizer_or_save_bypass():
    _, failures = audit()
    assert failures == []


def test_ci_coverage_detects_optimizer_import_aliases_and_dynamic_construction(
    tmp_path,
):
    for source in (
        "from torch.optim import SGD as Hidden\no=Hidden(m.parameters())\n",
        'import torch as t\no=getattr(t.optim,"AdamW")(m.parameters())\n',
    ):
        p = tmp_path / "future.py"
        p.write_text(source)
        assert inspect_file(p)


def test_valid_frozen_provenance_allows_integrity_replay_but_not_a_new_fit(
    tmp_path, monkeypatch
):
    import scripts.verify_frozen_replay as frozen

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    old = single(40000)
    (repo / "prefit.json").write_text(json.dumps(old))
    (repo / "original.pt").write_bytes(b"original-byte-integrity-fixture")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Software Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "frozen fixture",
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "entries": {
                    "original": {
                        "commit": commit,
                        "registration_path": "prefit.json",
                        "registration_sha256": file_sha(repo / "prefit.json"),
                        "checkpoints": [
                            {
                                "path": "original.pt",
                                "bytes": len((repo / "original.pt").read_bytes()),
                                "sha256": file_sha(repo / "original.pt"),
                            }
                        ],
                    }
                }
            }
        )
    )
    monkeypatch.setattr(frozen, "ROOT", repo)
    monkeypatch.setattr(frozen, "REGISTRY", registry)
    result = check_manifest(old, operation="replay", provenance="original")
    assert (
        result["training_authorized"] is False
        and result["verified_original_checkpoints"] == 1
    )
    with pytest.raises(ParameterBudgetError, match="100k target"):
        check_manifest(old)
    altered = deepcopy(old)
    altered["models"][0]["trainable_parameters"] = 41000
    with pytest.raises(ParameterBudgetError, match="frozen"):
        check_manifest(altered, operation="replay", provenance="original")


def test_in_band_live_count_must_equal_its_declared_budget(tmp_path):
    model, opt, _, _ = configure(tmp_path, Toy(99781), declared=100000)
    with pytest.raises(ParameterBudgetError, match="differs from"):
        checked_optimizer_step(opt)


def test_checkpoint_tensor_shape_is_checked_even_if_file_hashes_are_redeclared(
    tmp_path,
):
    _, _, m, receipt, checkpoint = completed(tmp_path)
    torch.save({"model": {"weight": torch.zeros(40000)}}, checkpoint)
    r = json.loads(receipt.read_text())
    r["checkpoints"][0]["sha256"] = file_sha(checkpoint)
    receipt.write_text(json.dumps(r))
    measured = m["models"][0]["measured"]
    measured["checkpoint_sha256"] = file_sha(checkpoint)
    measured["runtime_evidence"].update(
        receipt_sha256=file_sha(receipt), checkpoint_sha256=file_sha(checkpoint)
    )
    with pytest.raises(ParameterBudgetError, match="inventory"):
        check_manifest(m, "final")


def test_global_backstop_rejects_unregistered_or_copied_guard_optimizers(tmp_path):
    from tac_osm.parameter_budget import install_unbound_optimizer_guard

    model, opt, _, _ = configure(tmp_path)
    raw = torch.optim.SGD(model.parameters(), lr=0.1)
    handle = install_unbound_optimizer_guard()
    try:
        with pytest.raises(ParameterBudgetError, match="unregistered"):
            raw.step()
        raw._tac_parameter_budget_guard = opt._tac_parameter_budget_guard
        with pytest.raises(ParameterBudgetError, match="unregistered"):
            raw.step()
        checked_optimizer_step(opt)
    finally:
        handle.remove()


def test_frozen_registration_edit_after_first_update_blocks_further_updates(tmp_path):
    model, opt, path, _ = configure(tmp_path)
    checked_optimizer_step(opt)
    value = json.loads(path.read_text())
    value["extra_tuning"] = True
    path.write_text(json.dumps(value))
    with pytest.raises(ParameterBudgetError, match="changed during"):
        checked_optimizer_step(opt)


def test_checkpoint_hashing_has_no_numpy_dependency(monkeypatch):
    from tac_osm.parameter_budget import parameter_values_sha

    def unavailable(*args, **kwargs):
        raise AssertionError("NumPy is not required for the runtime guard")

    monkeypatch.setattr(torch.Tensor, "numpy", unavailable)
    model = Toy()
    _, inventory = live_inventory(model)
    assert len(parameter_values_sha(model.state_dict(), inventory)) == 64


def test_final_declared_update_count_must_equal_actual_guarded_updates(tmp_path):
    _, _, manifest, _, _ = completed(tmp_path)
    manifest["models"][0]["measured"]["actual_steps"] = 10
    with pytest.raises(ParameterBudgetError, match="step count"):
        check_manifest(manifest, "final")


def test_final_deployed_count_cannot_contradict_the_guarded_whole_model(tmp_path):
    _, _, manifest, _, _ = completed(tmp_path)
    manifest["models"][0]["measured"]["deployed_parameters"] = 100001
    with pytest.raises(ParameterBudgetError, match="actual model size"):
        check_manifest(manifest, "final")
