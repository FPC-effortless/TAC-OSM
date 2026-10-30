"""Serialization-only fixtures. No model fitting, predictions or optimizer steps."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


review = load("langacquire097_postfit_review_test", ROOT / "scripts/langacquire097_postfit_review.py")
budget = load("langacquire097_postfit_original_budget_test", ROOT / "src/tac_osm/research_objective_budget.py")


def entry(name, calls=1, *, stage="optimizer", inputs=None, outputs=None):
    return {"stage": stage, "signature": json.dumps(
        [name, inputs or [], outputs or []], separators=(",", ":")), "calls": calls}


def fixture():
    return [entry("aten::mul", 3, inputs=[["torch.float32", [2]]], outputs=[["torch.float32", [2]]]),
            entry(review.PROFILER_ENTER), entry(review.PROFILER_EXIT)]


def test_narrow_parser_retains_every_arithmetic_and_marker_count():
    rows = fixture()
    original = budget.ledger_counter
    with pytest.raises(ValueError, match="ATen"):
        original(rows)
    with review.compatible_ledger_parser(budget):
        actual = budget.ledger_counter(rows)
        assert len(actual) == 3
        assert sum(actual.values()) == 5
        assert actual[("optimizer", rows[0]["signature"])] == 3
        budget.verify_same_ledger(rows, deepcopy(rows), "same", require_matrix=False)
    assert budget.ledger_counter is original
    with pytest.raises(ValueError, match="ATen"):
        budget.ledger_counter(rows)


@pytest.mark.parametrize("kind", ["unknown", "input", "output", "stage", "boolean", "zero", "duplicate", "unbalanced", "serialization"])
def test_only_exact_empty_optimizer_profiler_markers_are_accepted(kind):
    rows = fixture()
    if kind == "unknown":
        rows[1] = entry("custom::unbounded_work")
    elif kind in ("input", "output"):
        rows[1] = entry(review.PROFILER_ENTER, **{
            "inputs" if kind == "input" else "outputs": [["torch.float32", [2]]]})
    elif kind == "stage":
        rows[1] = entry(review.PROFILER_ENTER, stage="forward_backward")
    elif kind == "boolean":
        rows[1]["calls"] = True
    elif kind == "zero":
        rows[1]["calls"] = 0
    elif kind == "duplicate":
        rows.append(deepcopy(rows[1]))
    elif kind == "unbalanced":
        rows[1]["calls"] = 2
    elif kind == "serialization":
        rows[1]["signature"] = json.dumps(json.loads(rows[1]["signature"]), indent=1)
    with review.compatible_ledger_parser(budget), pytest.raises(ValueError):
        budget.ledger_counter(rows)


def test_real_arithmetic_and_profiler_count_mismatches_still_fail():
    left, right = fixture(), fixture()
    with review.compatible_ledger_parser(budget):
        right[0]["calls"] += 1
        with pytest.raises(ValueError, match="counts differ"):
            budget.verify_same_ledger(left, right, "arithmetic", require_matrix=False)
        right = fixture()
        right[1]["calls"] = right[2]["calls"] = 2
        with pytest.raises(ValueError, match="counts differ"):
            budget.verify_same_ledger(left, right, "markers", require_matrix=False)


def test_original_arithmetic_parser_and_original_source_bytes_remain_authoritative():
    path = ROOT / "src/tac_osm/research_objective_budget.py"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = fixture()
    rows[0]["signature"] = json.dumps(["aten::mul", [["torch.float32", [-1]]], []], separators=(",", ":"))
    with review.compatible_ledger_parser(budget), pytest.raises(ValueError, match="shape"):
        budget.ledger_counter(rows)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert before == review.ORIGINAL_BUDGET_SHA256


def test_actual_record_function_trace_reproduces_and_repairs_only_the_format_error():
    torch = pytest.importorskip("torch")
    with budget.OperatorLedger("optimizer") as ledger:
        with torch.autograd.profiler.record_function("serialization-fixture"):
            values = torch.ones(2)
            values = values + values
    rows = ledger.rows()
    assert {json.loads(row["signature"])[0] for row in rows
            if not json.loads(row["signature"])[0].startswith("aten::")} == review.PROFILER_MARKERS
    with pytest.raises(ValueError, match="ATen"):
        budget.ledger_counter(rows)
    with review.compatible_ledger_parser(budget):
        assert sum(budget.ledger_counter(rows).values()) == sum(row["calls"] for row in rows)
    assert values.tolist() == [2., 2.]


def git_repo(folder):
    subprocess.run(["git", "init", "-q", str(folder)], check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.invalid"], cwd=folder, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=folder, check=True)
    (folder / "source.py").write_text("value = 1\n")
    subprocess.run(["git", "add", "source.py"], cwd=folder, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture source"], cwd=folder, check=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=folder, text=True).strip()


def test_original_checkout_must_be_exact_clean_and_without_added_executable_source(tmp_path):
    root = tmp_path / "original"
    commit = git_repo(root)
    assert review.verify_source_checkout(root, commit) == root.resolve()
    with pytest.raises(ValueError, match="exact original"):
        review.verify_source_checkout(root, "0" * 40)
    (root / "source.py").write_text("value = 2\n")
    with pytest.raises(ValueError, match="tracked source"):
        review.verify_source_checkout(root, commit)
    (root / "source.py").write_text("value = 1\n")
    (root / "shadow.py").write_text("value = 9\n")
    with pytest.raises(ValueError, match="unexpected executable"):
        review.verify_source_checkout(root, commit)


def test_supplied_review_commit_must_bind_actual_driver_bytes(tmp_path):
    root = tmp_path / "review"
    commit = git_repo(root)
    driver = root / "source.py"
    assert review.verify_review_driver(driver, None) is False
    assert review.verify_review_driver(driver, commit) is True
    with pytest.raises(ValueError, match="authenticated"):
        review.verify_review_driver(driver, "0" * 40)
    driver.write_text("value = 2\n")
    with pytest.raises(ValueError, match="driver bytes differ"):
        review.verify_review_driver(driver, commit)


def test_inprocess_final_gate_keeps_original_manifests_and_all_measured_rows(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    manifest = {"models": [{"name": "full", "seed": 1701}, {"name": "query", "seed": 1701}],
                "untouched_contract": "original frozen criteria"}
    (root / "manifest.json").write_text(json.dumps(manifest))
    resources = SimpleNamespace(
        ROOT=root, FAMILIES=("tac",), read_json=lambda path: json.loads(Path(path).read_text()),
        write_json=review.write_json, sha256=review.sha256)
    called = []

    def validate(value, phase):
        called.append((deepcopy(value), phase))
        assert value["untouched_contract"] == manifest["untouched_contract"]
        return {"status": "PASS", "eligible_for_architecture_superiority": False}

    runner, validator = SimpleNamespace(R=resources), SimpleNamespace(check_manifest=validate)
    output = tmp_path / "evaluation"
    output.mkdir()
    records = {("tac", objective, 1701): {"receipt": {"identity": objective, "steps": 256}}
               for objective in ("full", "query")}
    deadline = SimpleNamespace(check=lambda: None)
    log = []
    result = review.reviewed_final_gates(runner, validator, {"review": "explicit"})(
        records, {"manifest_paths": {"tac": "manifest.json"}}, output, deadline, log)
    assert [row["measured"]["identity"] for row in called[0][0]["models"]] == ["full", "query"]
    assert called[0][1] == "final" and result["tac"]["status"] == "PASS"
    assert json.loads((root / "manifest.json").read_text()) == manifest
    assert log[0]["exit_code"] == 0
    assert (output / "FINAL_GATE_REVIEW_TAC.json").is_file()


def test_replay_packaging_keeps_exact_initial_and_profile_bytes(tmp_path):
    root = tmp_path / "source"
    profile_dir = root / "profiles"
    profile_dir.mkdir(parents=True)
    payload = b"constructed initialization file, never a trained model"
    initial = profile_dir / "INIT_tac_1701.pt"
    initial.write_bytes(payload)
    profile = {"profiles": [{"seed": 1701, "objective": objective,
                            "initial_state_file": initial.name,
                            "initial_state_sha256": review.sha256(initial)}
                           for objective in ("full", "query")]}
    path = profile_dir / "PROFILE_tac.json"
    path.write_text(json.dumps(profile))
    index = profile_dir / "PROFILE_INDEX.json"
    index.write_text(json.dumps({"optimizer_steps": 0, "profiles": [{"sha256": review.sha256(path)}]}))
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"operator_profile_path": "profiles/PROFILE_tac.json",
                                    "operator_profile_sha256": review.sha256(path)}))
    resource = SimpleNamespace(ROOT=root, HERE=root, FAMILIES=("tac",), SEEDS=(1701,),
                               read_json=lambda path: json.loads(Path(path).read_text()))
    result = review.copy_initial_snapshots(SimpleNamespace(R=resource),
        {"manifest_paths": {"tac": "manifest.json"}}, tmp_path / "retained")
    assert result[initial.name]["sha256"] == review.sha256(initial)
    assert (tmp_path / "retained" / initial.name).read_bytes() == payload
    assert (tmp_path / "retained" / index.name).read_bytes() == index.read_bytes()
    assert result[index.name]["sha256"] == review.sha256(index)
    assert (tmp_path / "retained/ORIGINAL_INITIALIZATION_FILES.json").is_file()
