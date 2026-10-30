#!/usr/bin/env python3
"""Evaluation-only compatibility review of the original LANGACQUIRE097 fits.

The original source, model/optimizer archives, profiles, actor, evaluator and
scientific thresholds stay unchanged. The original ledger parser rejects two
empty-tensor profiler markers emitted around AdamW.step. This driver admits
only those serialization markers in memory and retains their counts in every
comparison. It never executes a fit or optimizer step.

All substantive assembly/evaluation functions come from the clean, exact
training checkout. Separate receipts identify this driver as a revised
postfit verifier; the original on-disk verifier is not represented as repaired.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace


TRAINING_COMMIT = "d21c7e7b385ca164eac9046fafca43fc9a3762b1"
ORIGINAL_BUDGET_SHA256 = "4855d96a02dec22669d033440e7391da6d469244b5b2f182c4f6fdb94b5ab7a1"
STUDY = Path("research/mathematical-engineering/langacquire097")
PROFILER_ENTER = "profiler::_record_function_enter_new"
PROFILER_EXIT = "profiler::_record_function_exit"
PROFILER_MARKERS = frozenset((PROFILER_ENTER, PROFILER_EXIT))
PATCH_RULE = {
    "allowed_names": sorted(PROFILER_MARKERS),
    "required_stage": "optimizer",
    "required_tensor_inputs": [],
    "required_tensor_outputs": [],
    "counts": "positive integers, unique rows, balanced enter/exit; retained in Counter comparisons",
    "other_operations": "every ATen row goes through the unchanged original parser; any other namespace fails",
    "scientific_scope": "profiler serialization compatibility only; no arithmetic, data, parameter or outcome waiver",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def compatibility_counter(original_counter, row):
    """Retain all work; narrowly recognize empty optimizer profiler markers."""
    entries = row.get("operators") if isinstance(row, dict) else row
    if not isinstance(entries, list) or not entries:
        raise ValueError("complete nonempty operator ledger required")
    arithmetic = []
    markers = Counter()
    seen = set()
    balanced = Counter()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("operator row must be an object")
        stage, signature, calls = (entry.get(key) for key in ("stage", "signature", "calls"))
        if not isinstance(signature, str):
            raise ValueError("operator signature missing")
        try:
            decoded = json.loads(signature)
        except (TypeError, ValueError) as error:
            raise ValueError("invalid operator signature") from error
        if not (isinstance(decoded, list) and len(decoded) == 3 and isinstance(decoded[0], str)):
            raise ValueError("actual operator signature required")
        name = decoded[0]
        if name.startswith("aten::"):
            arithmetic.append(entry)
            continue
        if name not in PROFILER_MARKERS:
            raise ValueError("unrecognized non-ATen operator: " + name)
        if stage != "optimizer" or decoded[1] != [] or decoded[2] != []:
            raise ValueError("profiler compatibility requires empty tensor inputs/outputs in optimizer stage")
        if set(entry) != {"stage", "signature", "calls"} or signature != json.dumps(decoded, separators=(",", ":")):
            raise ValueError("profiler marker rows must use the original canonical serialization")
        if type(calls) is not int or calls <= 0:
            raise ValueError("profiler call counts require positive integers")
        key = stage, signature
        if key in seen:
            raise ValueError("duplicate profiler ledger row")
        seen.add(key)
        markers[key] = calls
        balanced[name] += calls
    if balanced[PROFILER_ENTER] != balanced[PROFILER_EXIT]:
        raise ValueError("unbalanced optimizer profiler marker counts")
    # No arithmetic row is filtered, rewritten or independently reinterpreted.
    # A profiler-only ledger is rejected by the original nonempty-work parser.
    result = original_counter(arithmetic)
    if set(result) & set(markers):
        raise ValueError("duplicate operator identity")
    result.update(markers)
    return result


@contextmanager
def compatible_ledger_parser(budget):
    """Change one Python binding temporarily; never write original source."""
    original = budget.ledger_counter

    def reviewed_counter(row):
        return compatibility_counter(original, row)

    budget.ledger_counter = reviewed_counter
    try:
        yield
    finally:
        budget.ledger_counter = original


def verify_source_checkout(root, commit):
    """Check Git identity/clean tracked bytes, without trusting current branch."""
    root = Path(root).resolve()
    if re.fullmatch("[0-9a-f]{40}", commit or "") is None:
        raise ValueError("complete original source commit required")
    actual_root = Path(subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=root, text=True).strip()).resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if actual_root != root or head != commit:
        raise ValueError("a separate checkout at the exact original training commit is required")
    for command in (["git", "diff", "--quiet", commit, "--"],
                    ["git", "diff", "--cached", "--quiet", commit, "--"]):
        if subprocess.run(command, cwd=root, check=False).returncode:
            raise ValueError("original tracked source bytes or index changed")
    extras = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=root, text=True).splitlines()
    if any(path.endswith((".py", ".pyc")) for path in extras):
        raise ValueError("unexpected executable source in the original checkout")
    return root


def verify_review_driver(path, commit):
    """Authenticate optional review-commit metadata against this driver's bytes."""
    if commit is None:
        return False
    if re.fullmatch("[0-9a-f]{40}", commit) is None:
        raise ValueError("complete review-driver commit required")
    path = Path(path).resolve()
    try:
        root = Path(subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=path.parent,
            text=True, stderr=subprocess.DEVNULL).strip()).resolve()
        relative = path.relative_to(root).as_posix()
        original = subprocess.check_output(
            ["git", "show", commit + ":" + relative], cwd=root, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, ValueError) as error:
        raise ValueError("review-driver commit cannot be authenticated in its checkout") from error
    if original != path.read_bytes():
        raise ValueError("review-driver bytes differ from supplied review commit")
    return True


def load_exact(name, path):
    path = Path(path).resolve()
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path:
            raise ValueError("source module collision: " + name)
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def load_original(root):
    root = verify_source_checkout(root, TRAINING_COMMIT)
    budget_path = root / "src/tac_osm/research_objective_budget.py"
    if sha256(budget_path) != ORIGINAL_BUDGET_SHA256:
        raise ValueError("original verifier differs from the exact reviewed source")
    source_path = root / "src"
    sys.path.insert(0, str(source_path))
    # A new process is required so canonical imports cannot reuse another tree.
    for name, module in tuple(sys.modules.items()):
        if name == "tac_osm" or name.startswith("tac_osm."):
            file = getattr(module, "__file__", None)
            if file and not Path(file).resolve().is_relative_to(source_path):
                raise ValueError("foreign TAC module already imported: " + name)
    budget = importlib.import_module("tac_osm.research_objective_budget")
    if Path(budget.__file__).resolve() != budget_path.resolve():
        raise ValueError("verifier import did not resolve to the original checkout")
    validator = load_exact("langacquire097_review_original_validator", root / "scripts/validate_research_parity.py")
    runner = load_exact("langacquire097_review_original_runner", root / STUDY / "run.py")
    return runner, validator, budget


def reviewed_final_gates(runner, validator, review):
    """Same original manifests and strict validator, executed in this process."""
    resources = runner.R

    def final_gates(matrix, register, output, deadline, log):
        gates = {}
        for family in resources.FAMILIES:
            deadline.check()
            started = time.perf_counter()
            manifest = resources.read_json(resources.ROOT / register["manifest_paths"][family])
            for row in manifest["models"]:
                row["measured"] = matrix[(family, row["name"], row["seed"])]["receipt"]
            path = output / ("FINAL_PARITY_" + family.upper() + ".json")
            resources.write_json(path, manifest)
            result = validator.check_manifest(manifest, "final")
            if result.get("eligible_for_architecture_superiority") is not False:
                raise ValueError("postfit compatibility review cannot qualify architecture superiority")
            report = output / ("FINAL_GATE_REVIEW_" + family.upper() + ".json")
            resources.write_json(report, {"original_resource_gate": result,
                "postfit_verifier_review": review,
                "original_source_files_unchanged": True})
            entry = {"label": "final-parity-" + family,
                     "elapsed_seconds": time.perf_counter() - started,
                     "exit_code": 0,
                     "execution": "original validator in-process with explicit profiler-marker compatibility parser"}
            log.append(entry)
            resources.write_json(output / (entry["label"] + ".execution.json"), entry)
            gates[family] = {"status": "PASS", "manifest_sha256": resources.sha256(path),
                "scope": "within-family complete operator parity; explicit postfit profiler-marker review",
                "review_report_sha256": resources.sha256(report)}
            deadline.check()
        return gates

    return final_gates


def copy_initial_snapshots(runner, register, destination):
    """Retain the exact original zero-update archives beyond CI expiration."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    resources = runner.R
    copied = {}
    for family in resources.FAMILIES:
        manifest = resources.read_json(resources.ROOT / register["manifest_paths"][family])
        profile_path = resources.ROOT / manifest["operator_profile_path"]
        if sha256(profile_path) != manifest["operator_profile_sha256"]:
            raise ValueError("original profile bytes differ before replay packaging")
        profile = resources.read_json(profile_path)
        originals = {profile_path.name: (profile_path, manifest["operator_profile_sha256"])}
        for seed in resources.SEEDS:
            pair = [row for row in profile["profiles"] if row["seed"] == seed]
            if (len(pair) != 2 or {row["objective"] for row in pair} != {"full", "query"}
                    or pair[0]["initial_state_file"] != pair[1]["initial_state_file"]
                    or pair[0]["initial_state_sha256"] != pair[1]["initial_state_sha256"]):
                raise ValueError("complete paired original initialization identity required")
            name = pair[0]["initial_state_file"]
            if name != f"INIT_{family}_{seed}.pt":
                raise ValueError("original initialization filename differs")
            originals[name] = (profile_path.parent / name, pair[0]["initial_state_sha256"])
        for name, (source, expected) in originals.items():
            if source.is_symlink() or sha256(source) != expected:
                raise ValueError("original initialization/profile archive changed")
            target = destination / name
            with source.open("rb") as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            if sha256(target) != expected:
                raise ValueError("copied original replay archive differs")
            copied[name] = {"sha256": expected, "bytes": target.stat().st_size}
    index = resources.HERE / "profiles/PROFILE_INDEX.json"
    if index.exists():
        if index.is_symlink():
            raise ValueError("original profile index must be a regular artifact file")
        name = index.name
        expected = sha256(index)
        target = destination / name
        with index.open("rb") as incoming, target.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
        if sha256(target) != expected or sha256(index) != expected:
            raise ValueError("copied original profile index differs")
        copied[name] = {"sha256": expected, "bytes": target.stat().st_size}
    write_json(destination / "ORIGINAL_INITIALIZATION_FILES.json", {
        "schema": "langacquire097.original-initialization-replay.v1",
        "training_source_commit": TRAINING_COMMIT,
        "optimizer_steps_in_initial_states": 0,
        "files": copied})
    return copied


def run_review(args):
    if args.source_commit != TRAINING_COMMIT:
        raise ValueError("this narrow review supports only the original LANGACQUIRE097 training commit")
    root = Path(args.source_root).resolve()
    output, data, fits = (Path(value).resolve() for value in (args.output, args.data, args.fits))
    assembled = Path(args.assembled).resolve() if args.assembled else output / "assembled"
    if any(path.is_relative_to(root) for path in (output, data, assembled)):
        raise ValueError("review outputs, data and assembly must be outside the original source checkout")
    if output.exists() or assembled.exists() or data.exists():
        raise FileExistsError("use new review, assembly and data paths; earlier evidence is immutable")
    review_commit_verified = verify_review_driver(__file__, args.review_commit)
    runner, validator, budget = load_original(root)
    config = root / STUDY / "REGISTER.json"
    before = runner.R.verify_source(TRAINING_COMMIT, config)
    review = {"schema": "langacquire097.postfit-verifier-review.v1",
        "training_source_commit": TRAINING_COMMIT,
        "review_driver_commit": args.review_commit,
        "review_driver_commit_verified": review_commit_verified,
        "review_driver_sha256": sha256(__file__),
        "original_verifier_sha256": ORIGINAL_BUDGET_SHA256,
        "repair_rule": PATCH_RULE,
        "model_training_code_changed": False,
        "model_weights_retrained_or_selected": False,
        "actor_scorer_cohort_thresholds_changed": False,
        "original_verifier_disk_bytes_changed": False,
        "new_postfit_verifier_implementation": True,
        "qualification": "internal research; no architecture, AGI or independent qualification",
        "source_hashes_before": before}
    output.mkdir(parents=True)
    write_json(output / "POSTFIT_VERIFIER_REVIEW.json", review)
    original_final_gates = runner.final_gates
    old_cwd, old_pythonpath = Path.cwd(), os.environ.get("PYTHONPATH")
    started = time.perf_counter()
    try:
        os.chdir(root)
        os.environ["PYTHONPATH"] = str(root / "src")
        initial_files = copy_initial_snapshots(runner, runner.R.load_register(config),
                                               output / "initial-states")
        runner.assemble(SimpleNamespace(config=config, source_commit=TRAINING_COMMIT,
                                        fits=fits, output=assembled))
        with compatible_ledger_parser(budget):
            runner.final_gates = reviewed_final_gates(runner, validator, review)
            runner.evaluate(SimpleNamespace(config=config, source_commit=TRAINING_COMMIT,
                fits=assembled, data=data, output=output / "evaluation"))
        after = runner.R.verify_source(TRAINING_COMMIT, config)
        verify_source_checkout(root, TRAINING_COMMIT)
        if before != after or sha256(root / "src/tac_osm/research_objective_budget.py") != ORIGINAL_BUDGET_SHA256:
            raise ValueError("original source changed during postfit review")
        write_json(output / "POSTFIT_REVIEW_COMPLETED.json", {
            "status": "COMPLETED_EVALUATION_ONLY_REVIEW",
            "training_source_commit": TRAINING_COMMIT,
            "review_driver_sha256": review["review_driver_sha256"],
            "review_driver_commit": args.review_commit,
            "original_source_hashes_unchanged": True,
            "original_initialization_files_preserved": initial_files,
            "new_optimizer_steps": 0,
            "elapsed_seconds": time.perf_counter() - started,
            "result_path": "evaluation/RESULTS.json"})
    except BaseException as error:
        write_json(output / "FAILED_POSTFIT_REVIEW.json", {
            "status": "FAILED_OR_INCONCLUSIVE_RETAINED",
            "training_source_commit": TRAINING_COMMIT,
            "review_driver_sha256": review["review_driver_sha256"],
            "reason": type(error).__name__ + ": " + str(error),
            "new_optimizer_steps": 0})
        raise
    finally:
        runner.final_gates = original_final_gates
        os.chdir(old_cwd)
        if old_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = old_pythonpath


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", default=TRAINING_COMMIT)
    parser.add_argument("--review-commit")
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assembled", type=Path)
    run_review(parser.parse_args())


if __name__ == "__main__":
    main()
