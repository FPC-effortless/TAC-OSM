"""Prospective source/ledger enforcement fixtures; no fitting or capability claims."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from tac_osm.research_objective_budget import (
    OPTIMIZER_MARKER_CONTRACT, PROFILER_ENTER, PROFILER_EXIT, OperatorLedger,
    canonical, digest, ledger_counter, required_study_paths, source_bindings,
    validate_comparison, verify_same_ledger,
)

STUDY = "research/mathematical-engineering/langcontrol098"
SHARED = ("src/tac_osm/research_objective_budget.py", "scripts/validate_research_parity.py")
STUDY_FILES = ("models.py", "objective.py", "train.py", "resources.py", "data.py", "profile098.py", "run.py")


def entry(name, calls=1, *, stage="optimizer", inputs=None, outputs=None):
    return {"stage": stage, "signature": canonical([name, inputs or [], outputs or []]).decode(), "calls": calls}


def marked_ledger():
    return [entry("aten::mul", 3, inputs=[["torch.float32", [2]]], outputs=[["torch.float32", [2]]]),
            entry(PROFILER_ENTER), entry(PROFILER_EXIT)]


def create_sources(root):
    for name in (*SHARED, *(STUDY + "/" + name for name in STUDY_FILES)):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Constructed executable binding fixture; no training.\n")
    inherited = root / "inherited_model.py"
    inherited.write_text("# Constructed inherited graph fixture.\n")
    return inherited.name


def test_markers_are_opt_in_and_counts_are_retained():
    rows = marked_ledger()
    with pytest.raises(ValueError, match="ATen"):
        ledger_counter(rows)
    counter = ledger_counter(rows, optimizer_marker_contract=OPTIMIZER_MARKER_CONTRACT)
    assert len(counter) == 3 and sum(counter.values()) == 5
    assert counter[("optimizer", rows[1]["signature"])] == 1
    with pytest.raises(ValueError, match="unknown optimizer marker contract"):
        ledger_counter(rows, optimizer_marker_contract="allow-all-profiler")


@pytest.mark.parametrize("mutation", [
    "unknown", "input", "output", "stage", "bool", "float", "zero", "negative",
    "duplicate", "unbalanced", "noncanonical", "extra_field", "arithmetic_shape", "markers_only",
])
def test_opt_in_keeps_narrow_marker_and_arithmetic_contract(mutation):
    rows = marked_ledger()
    if mutation == "unknown":
        rows[1] = entry("custom::hidden_arithmetic")
    elif mutation in ("input", "output"):
        rows[1] = entry(PROFILER_ENTER, **{"inputs" if mutation == "input" else "outputs": [["torch.float32", [2]]]})
    elif mutation == "stage":
        rows[1] = entry(PROFILER_ENTER, stage="forward_backward")
    elif mutation in ("bool", "float", "zero", "negative"):
        rows[1]["calls"] = {"bool": True, "float": 1.0, "zero": 0, "negative": -1}[mutation]
    elif mutation == "duplicate":
        rows.append(deepcopy(rows[1]))
    elif mutation == "unbalanced":
        rows[1]["calls"] = 2
    elif mutation == "noncanonical":
        rows[1]["signature"] = json.dumps(json.loads(rows[1]["signature"]), indent=2)
    elif mutation == "extra_field":
        rows[1]["ignored_numeric_work"] = 0
    elif mutation == "arithmetic_shape":
        rows[0] = entry("aten::mul", inputs=[["torch.float32", [-1]]])
    else:
        rows = rows[1:]
    with pytest.raises(ValueError):
        ledger_counter(rows, optimizer_marker_contract=OPTIMIZER_MARKER_CONTRACT)


@pytest.mark.parametrize("mutation", ["arithmetic", "markers"])
def test_balanced_markers_do_not_hide_any_cross_arm_count_mismatch(mutation):
    left, right = marked_ledger(), marked_ledger()
    if mutation == "arithmetic":
        right[0]["calls"] += 1
    else:
        right[1]["calls"] = right[2]["calls"] = 2
    with pytest.raises(ValueError, match="counts differ"):
        verify_same_ledger(left, right, "prospective pair", require_matrix=False,
                           optimizer_marker_contract=OPTIMIZER_MARKER_CONTRACT)


def test_actual_record_function_rows_are_preserved_without_optimizer_step():
    torch = pytest.importorskip("torch")
    with OperatorLedger("optimizer") as ledger:
        with torch.autograd.profiler.record_function("prospective-serialization-only"):
            values = torch.ones(2) + torch.ones(2)
    rows = ledger.rows()
    non_aten = [json.loads(row["signature"])[0] for row in rows
                if not json.loads(row["signature"])[0].startswith("aten::")]
    assert set(non_aten) == {PROFILER_ENTER, PROFILER_EXIT}
    counter = ledger_counter(rows, optimizer_marker_contract=OPTIMIZER_MARKER_CONTRACT)
    assert sum(counter.values()) == sum(row["calls"] for row in rows)
    assert values.tolist() == [2., 2.]


def test_explicit_bundle_binds_new_profiler_and_every_direct_python_source(tmp_path):
    inherited = create_sources(tmp_path)
    sources = source_bindings(tmp_path, [inherited], study_directory=STUDY, profile_entrypoint="profile098.py")
    assert set(sources) == {inherited, *SHARED, *(STUDY + "/" + name for name in STUDY_FILES)}
    assert not any("langacquire097" in name for name in sources)
    changed = tmp_path / STUDY / "objective.py"
    changed.write_text("# Actual changed formula must invalidate source profile.\n")
    after = source_bindings(tmp_path, [inherited], study_directory=STUDY, profile_entrypoint="profile098.py")
    assert sources[STUDY + "/objective.py"] != after[STUDY + "/objective.py"]


@pytest.mark.parametrize("directory,entrypoint", [
    ("/tmp/study", "profile098.py"),
    ("research/mathematical-engineering/../langcontrol098", "profile098.py"),
    (STUDY, "../profile098.py"), (STUDY, "/tmp/profile098.py"),
    (STUDY, "missing.py"),
])
def test_source_bundle_rejects_escape_and_missing_entrypoint(tmp_path, directory, entrypoint):
    create_sources(tmp_path)
    with pytest.raises(ValueError):
        required_study_paths(tmp_path, directory, entrypoint)


def test_source_bundle_rejects_symlinked_profiler(tmp_path):
    inherited = create_sources(tmp_path)
    path = tmp_path / STUDY / "profile098.py"
    path.unlink()
    path.symlink_to(tmp_path / inherited)
    with pytest.raises(ValueError, match="symlink"):
        source_bindings(tmp_path, [inherited], study_directory=STUDY, profile_entrypoint="profile098.py")


def comparison_fixture(root):
    # Reuse the established constructed receipt generator, not a trained model.
    old_path = Path(__file__).with_name("test_langacquire097_budget.py")
    spec = importlib.util.spec_from_file_location("langcontrol098_legacy_receipt_fixture", old_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    manifest, profile = helper.fixture(root)
    inherited = create_sources(root)
    sources = source_bindings(root, [inherited], study_directory=STUDY, profile_entrypoint="profile098.py")
    profile["source_hashes"] = sources
    profile["implementation_sha256"] = digest(sources)
    manifest.update(study_directory=STUDY, profile_entrypoint="profile098.py",
                    optimizer_marker_contract=OPTIMIZER_MARKER_CONTRACT,
                    initial_state_schema="langcontrol098.initial-state.v1",
                    implementation_sha256=digest(sources))
    torch = pytest.importorskip("torch")
    for seed in manifest["seeds"]:
        path = root / f"INIT_tac_{seed}.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        saved["schema"] = manifest["initial_state_schema"]
        torch.save(saved, path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in profile["profiles"]:
            if row["seed"] == seed:
                row["initial_state_sha256"] = sha
        for row in manifest["models"]:
            if row["seed"] == seed:
                row["measured"]["actual_profile_initial_state_sha256"] = sha
    for row in manifest["models"]:
        measured = row["measured"]
        first, warm = marked_ledger(), marked_ledger()
        warm[0]["calls"] += 1
        measured["optimizer_operators_first"] = first
        measured["optimizer_operators_warm"] = warm
        measured["optimizer_operator_step_sha256"] = [digest(first), digest(warm), digest(warm)]
    helper.rewrite_profile(root, manifest, profile)
    return manifest, profile, helper


def test_complete_prospective_bundle_and_explicit_marker_contract_pass_both_gates(tmp_path):
    manifest, _, _ = comparison_fixture(tmp_path)
    for phase in ("preflight", "final"):
        result = validate_comparison(manifest, phase, root=tmp_path)
        assert result["status"] == "PASS_WITHIN_FAMILY_OPERATOR_PARITY"
        assert result["eligible_for_architecture_superiority"] is False
        assert result["absolute_training_flops"] == "INCONCLUSIVE_NOT_CLAIMED"


@pytest.mark.parametrize("mutation", ["omitted", "added", "changed", "wrong_schema", "missing_entrypoint"])
def test_recomputed_hashes_cannot_hide_missing_source_membership_or_archive_identity(tmp_path, mutation):
    manifest, profile, helper = comparison_fixture(tmp_path)
    if mutation == "omitted":
        del profile["source_hashes"][STUDY + "/objective.py"]
        profile["implementation_sha256"] = manifest["implementation_sha256"] = digest(profile["source_hashes"])
        helper.rewrite_profile(tmp_path, manifest, profile)
    elif mutation == "added":
        (tmp_path / STUDY / "new_decoder.py").write_text("# New executable cannot escape profile binding.\n")
    elif mutation == "changed":
        (tmp_path / STUDY / "objective.py").write_text("# Formula changed after source profile.\n")
    elif mutation == "wrong_schema":
        manifest["initial_state_schema"] = "langacquire097.initial-state.v1"
    else:
        del manifest["profile_entrypoint"]
    with pytest.raises(ValueError):
        validate_comparison(manifest, root=tmp_path)


def test_opt_in_marker_contract_remains_required_at_final_gate(tmp_path):
    manifest, _, _ = comparison_fixture(tmp_path)
    del manifest["optimizer_marker_contract"]
    assert validate_comparison(manifest, "preflight", root=tmp_path)["status"] == "PASS_WITHIN_FAMILY_OPERATOR_PARITY"
    with pytest.raises(ValueError, match="ATen"):
        validate_comparison(manifest, "final", root=tmp_path)
