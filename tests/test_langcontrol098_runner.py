"""No-fit source/receipt boundary tests for the controlled-loss follow-up.

Transport fixtures are explicitly untrained. No test executes an optimizer,
downloads publisher data, or measures model capability.
"""
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
from unittest import mock
import zipfile

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "research/mathematical-engineering/langcontrol098"
OLD = ROOT / "research/mathematical-engineering/langacquire097"


def load(name, directory=HERE, prefix="langcontrol098_runner_independent_"):
    spec = importlib.util.spec_from_file_location(prefix + name, directory / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


T, E, RUN, D = (load(name) for name in ("train", "evaluate", "run", "data"))
OLD_MODELS = load("models", OLD, "langcontrol098_unchanged_parent_")
OLD_DATA = load("data", OLD, "langcontrol098_unchanged_parent_")


@pytest.mark.parametrize("damage", ["weights", "lambda", "normalization", "reduction", "missing", "study"])
def test_registration_rejects_any_coefficient_or_normalization_change(tmp_path, damage):
    registration = json.loads((HERE / "REGISTER.json").read_text())
    if damage == "weights":
        registration["objective_control"]["domain_weights"] = {"lexical": 0.6, "arc": 0.4}
    elif damage == "lambda":
        registration["objective_control"]["prefix_lambda_by_objective"]["full"] = 0.5
    elif damage == "normalization":
        registration["objective_control"]["normalize_by_total_coefficient_mass"] = True
    elif damage == "reduction":
        registration["objective_control"]["reduction"] = "global_token_average"
    elif damage == "missing":
        registration.pop("objective_control")
    else:
        registration["study_id"] = "LANGACQUIRE097"
    candidate = tmp_path / "changed.json"
    candidate.write_text(json.dumps(registration))
    with pytest.raises(ValueError, match="fixed-coefficient"):
        T.R.load_register(candidate)


def test_local_fitting_guard_runs_before_data_or_optimizer_access():
    with mock.patch.dict(os.environ, {"GITHUB_ACTIONS": "false"}), \
         mock.patch.object(torch.optim, "AdamW", side_effect=AssertionError("optimizer forbidden")) as optimizer, \
         mock.patch.object(T.R, "load_register", side_effect=AssertionError("no data/config access")) as registration:
        with pytest.raises(ValueError, match="routed to GitHub Actions"):
            T.train(SimpleNamespace())
        optimizer.assert_not_called()
        registration.assert_not_called()


@pytest.mark.parametrize("damage", ["missing", "lambda", "weights", "normalization"])
def test_evaluator_rejects_wrong_objective_mapping_before_model_creation(tmp_path, damage):
    config = HERE / "REGISTER.json"
    registration = T.R.load_register(config)
    source_commit, sources = "a" * 40, {"constructed_fixture.py": "b" * 64}
    mapping = T.O.objective_mapping("query")
    if damage == "lambda":
        mapping["prefix_lambda"] = 1.0
    elif damage == "weights":
        mapping["domain_weights"]["arc"] = 0.6
    elif damage == "normalization":
        mapping["normalize_by_total_coefficient_mass"] = True
    metadata = {
        "constructed_untrained_transport_fixture": True,
        "study_id": "LANGCONTROL098", "family": "gru", "objective": "query", "seed": 1701,
        "status": "SEALED_BEFORE_DEV", "optimizer_steps": registration["training"]["steps"],
        "source_commit": source_commit, "source_hashes": sources,
        "register_sha256": T.R.sha256(config),
        "data_manifest_sha256": registration["data"]["manifest_sha256"],
    }
    if damage != "missing":
        metadata["objective_control"] = mapping
    payload = {"schema": "langcontrol098.checkpoint.v1", "metadata": metadata,
               "model": {}, "optimizer": {}, "rng": {}}
    path = tmp_path / "UNTRAINED_TRANSPORT_FIXTURE.pt"
    path.write_bytes(b"Constructed metadata fixture; not a trained checkpoint")
    with mock.patch.object(E.R, "verify_source", return_value=sources), \
         mock.patch.object(torch, "load", return_value=payload), \
         mock.patch.object(E.M, "make_model", side_effect=AssertionError("mapping must fail first")) as make_model:
        with pytest.raises(ValueError, match="coefficient-control mapping"):
            E.load_checkpoint(path, E.R.sha256(path), config, source_commit)
        make_model.assert_not_called()


@pytest.mark.parametrize("damage", ["missing", "wrong"])
def test_artifact_matrix_rejects_missing_or_wrong_receipt_mapping(tmp_path, damage):
    config = tmp_path / "fixture_register.json"
    config.write_text('{"constructed_transport_fixture": true}')
    registration = {"training": {"steps": 2, "batch_size": 4, "context": 272}}
    source_commit, sources = "a" * 40, {"constructed_fixture.py": "b" * 64}
    # Supply every matrix row so a missing-file check cannot mask the mapping
    # assertion; no tensor loader or model forward is reached.
    for family in RUN.R.FAMILIES:
        for objective in RUN.R.OBJECTIVES:
            for seed in RUN.R.SEEDS:
                folder = tmp_path / "incoming" / f"{family}-{objective}-{seed}"
                folder.mkdir(parents=True)
                checkpoint = folder / "MODEL.pt"
                checkpoint.write_bytes(b"Constructed untrained transport fixture")
                receipt = {
                    "constructed_transport_fixture": True,
                    "study_id": "LANGCONTROL098", "family": family, "objective": objective, "seed": seed,
                    "status": "SEALED_BEFORE_DEV", "source_commit": source_commit, "source_hashes": sources,
                    "register_sha256": RUN.R.sha256(config), "optimizer_steps": 2, "actual_steps": 2,
                    "actual_examples": 8, "actual_dense_compute_tokens": 2176,
                    "checkpoint_sha256": RUN.R.sha256(checkpoint),
                    "replay": {"cold_reload_bitwise_equal": True},
                    "objective_control": T.O.objective_mapping(objective),
                }
                if (family, objective, seed) == ("tac", "full", 1701):
                    if damage == "missing":
                        receipt.pop("objective_control")
                    else:
                        receipt["objective_control"] = T.O.objective_mapping("query")
                (folder / "FIT_RECEIPT.json").write_text(json.dumps(receipt))
    with mock.patch.object(RUN, "verify_model_checkpoint", side_effect=AssertionError("no tensor load")) as verify:
        with pytest.raises(ValueError, match="source/config-mismatched"):
            RUN.model_artifacts(tmp_path / "incoming", registration, config, source_commit, sources)
        verify.assert_not_called()


def _seed_seal_fixture(directory):
    """Complete transport matrix with marker bytes, explicitly never trained."""
    config = directory / "fixture_register.json"
    config.write_text('{"constructed_transport_fixture": true}')
    registration = {"training": {"steps": 2}}
    commit = "a" * 40
    incoming = directory / "incoming"
    target = None
    for seed in RUN.R.SEEDS:
        seed_folder = incoming / f"seed-{seed}"
        models = []
        for family in RUN.R.FAMILIES:
            for objective in RUN.R.OBJECTIVES:
                name = f"{family}-{objective}"
                folder = seed_folder / name
                folder.mkdir(parents=True)
                checkpoint = folder / "MODEL.pt"
                checkpoint.write_bytes(f"UNTRAINED TRANSPORT FIXTURE {seed}-{name}".encode())
                receipt = {
                    "constructed_untrained_transport_fixture": True,
                    "study_id": "LANGCONTROL098", "family": family, "objective": objective, "seed": seed,
                    "objective_control": T.O.objective_mapping(objective),
                    "status": "SEALED_BEFORE_DEV", "actual_steps": 2,
                    "source_commit": commit, "register_sha256": RUN.R.sha256(config),
                    "checkpoint_sha256": RUN.R.sha256(checkpoint),
                    "schedule_sha256": f"{seed:064x}", "actual_tokens": 1000,
                    "actual_dense_compute_tokens": 2176, "actual_examples": 8,
                }
                receipt_path = folder / "FIT_RECEIPT.json"
                receipt_path.write_text(json.dumps(receipt))
                item = {"family": family, "objective": objective, "seed": seed,
                        "checkpoint": name + "/MODEL.pt", "checkpoint_sha256": RUN.R.sha256(checkpoint),
                        "receipt": name + "/FIT_RECEIPT.json", "receipt_sha256": RUN.R.sha256(receipt_path)}
                models.append(item)
                if target is None:
                    target = receipt_path
        seal = {"schema": "langcontrol098.seed-fit-seal.v1", "seed": seed,
                "source_commit": commit, "register_sha256": RUN.R.sha256(config),
                "DEV_predictions_started": False, "models": models}
        (seed_folder / "FITS_SEAL.json").write_text(json.dumps(seal))
    return incoming, config, registration, commit, target


def test_complete_transport_seals_verify_without_dataset_or_model_execution(tmp_path):
    incoming, config, registration, commit, _ = _seed_seal_fixture(tmp_path)
    with mock.patch.object(torch.nn.Module, "__call__", side_effect=AssertionError("no model forward")), \
         mock.patch.object(torch.optim, "AdamW", side_effect=AssertionError("no optimizer")):
        matrix = RUN.all_fits(incoming, registration, config, commit)
    assert len(matrix) == 40
    assert set(matrix) == {(family, objective, seed)
                           for family in RUN.R.FAMILIES
                           for objective in RUN.R.OBJECTIVES
                           for seed in RUN.R.SEEDS}


@pytest.mark.parametrize("damage", ["study", "family", "objective", "seed", "mapping_missing",
                                     "mapping_wrong", "mapping_boolean_type", "register"])
def test_resealed_receipt_identity_or_mapping_mutations_fail_at_seed_seal_boundary(tmp_path, damage):
    incoming, config, registration, commit, target = _seed_seal_fixture(tmp_path)
    receipt = json.loads(target.read_text())
    if damage == "study":
        receipt["study_id"] = "LANGACQUIRE097"
    elif damage == "family":
        receipt["family"] = "gru"
    elif damage == "objective":
        receipt["objective"] = "query"
    elif damage == "seed":
        receipt["seed"] = 1702
    elif damage == "mapping_missing":
        receipt.pop("objective_control")
    elif damage == "mapping_wrong":
        receipt["objective_control"] = T.O.objective_mapping("query")
    elif damage == "mapping_boolean_type":
        receipt["objective_control"]["normalize_by_total_coefficient_mass"] = 0
    else:
        receipt["register_sha256"] = "f" * 64
    target.write_text(json.dumps(receipt))
    # Recompute the transport hash deliberately: identity/mapping validation
    # must detect false metadata even when every file matches its supplied pin.
    seal_path = target.parent.parent / "FITS_SEAL.json"
    seal = json.loads(seal_path.read_text())
    for item in seal["models"]:
        if item["receipt"] == target.relative_to(seal_path.parent).as_posix():
            item["receipt_sha256"] = RUN.R.sha256(target)
    seal_path.write_text(json.dumps(seal))
    with mock.patch.object(torch.nn.Module, "__call__", side_effect=AssertionError("no model forward")), \
         mock.patch.object(torch.optim, "AdamW", side_effect=AssertionError("no optimizer")):
        with pytest.raises(ValueError, match="eligible for DEV scoring"):
            RUN.all_fits(incoming, registration, config, commit)


@pytest.mark.parametrize("family", ["tac", "gru", "ssm", "transformer"])
def test_new_study_models_exactly_preserve_parent_graph_logits_and_state(family):
    torch.set_num_threads(1)
    original = OLD_MODELS.make_model(family, 1701).eval()
    current = E.M.make_model(family, 1701).eval()
    assert original.counts() == current.counts()
    assert T.R.model_digest(original) == T.R.model_digest(current)
    tokens = torch.arange(40).reshape(2, 20)
    owners = ("fixture-owner-a", "fixture-owner-b")
    with torch.inference_mode():
        for observed in (tokens[:, :9], tokens[:, 9:]):
            if observed.shape[1] == 9:
                old_logits, old_state = original(observed, owners=owners)
                new_logits, new_state = current(observed, owners=owners)
            else:
                old_logits, old_state = original(observed, state=old_state, owners=owners)
                new_logits, new_state = current(observed, state=new_state, owners=owners)
            torch.testing.assert_close(old_logits, new_logits, atol=0, rtol=0)
            assert (old_state.arm, old_state.owners, old_state.clock) == (new_state.arm, new_state.owners, new_state.clock)
            assert set(old_state.tensors) == set(new_state.tensors)
            for name in old_state.tensors:
                torch.testing.assert_close(old_state.tensors[name], new_state.tensors[name], atol=0, rtol=0)


def test_data_and_source_provenance_are_byte_identical_to_parent():
    assert (HERE / "data.py").read_bytes() == (OLD / "data.py").read_bytes()
    assert (HERE / "SOURCE_PROVENANCE.json").read_bytes() == (OLD / "SOURCE_PROVENANCE.json").read_bytes()
    assert D.SALT == OLD_DATA.SALT
    for source_id in ("fixture-code-owner:123", "fixture-arc-task:0"):
        assert D.opaque_episode_id(source_id) == OLD_DATA.opaque_episode_id(source_id)


def _document(name):
    return b"".join((b"filler_value = 0\n" * 8 + name + b" = 1\n" + b"\n" * 45
                     + name + b" = 2\n" + b"\n" * 100) for _ in range(12))


def _archive():
    task = {"train": [{"input": [[0, 1]], "output": [[1, 0]]},
                       {"input": [[2, 3]], "output": [[3, 2]]}],
            "test": [{"input": [[4, 5]], "output": [[5, 4]]}]}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for identifier in ("fixture-a", "fixture-b"):
            archive.writestr(f"fixture/data/training/{identifier}.json", json.dumps(task))
        archive.writestr("fixture/data/evaluation/protected.json", "must never be read")
    return buffer.getvalue()


def test_offline_fixture_materialization_preserves_actor_ids_and_all_artifact_hashes(tmp_path):
    raw = {"cpython/fit.py": _document(b"fit_memory_token"),
           "cpython/dev.py": _document(b"dev_memory_token"),
           "cpython.LICENSE": b"Constructed fixture license", "arc.zip": _archive()}
    rows = [{"path": name, "partition": partition, "repo": "constructed-fixture", "commit": "fixture",
             "sha256": D.digest(raw["cpython/" + name])}
            for name, partition in (("fit.py", "FIT"), ("dev.py", "DEV"))]
    arc_policy = copy.deepcopy(D.contract()["arc_policy"])
    arc_policy.update(fit_tasks=1, dev_tasks=1)
    contract = {"cpython_files": rows, "cpython_license": {},
                "arc_assets": {"arc.zip": {"sha256": D.digest(raw["arc.zip"])}},
                "lexical_policy": {"fit_counts": {group: 1 for group in D.GROUPS},
                                   "dev_counts": {group: 1 for group in D.GROUPS}},
                "arc_policy": arc_policy}
    outputs = []
    for label, module in (("old", OLD_DATA), ("new", D)):
        cache = tmp_path / label
        with mock.patch.object(module, "contract", return_value=contract), \
             mock.patch.object(module, "_read_source", side_effect=lambda cache, relative, specification, bootstrap=None: raw[relative]), \
             mock.patch.object(module.urllib.request, "urlopen", side_effect=AssertionError("no network")):
            manifest = module.prepare(cache)
        assert manifest["status"] == "COHORT_READY"
        actors = json.loads((cache / "DEV_ACTOR.json").read_text())
        assert all(set(actor) == {"id", "domain", "acquisition_bytes", "query_bytes", "generation_limit"}
                   for actor in actors)
        outputs.append({name: (cache / name).read_bytes() for name in
                        ("FIT.json", "DEV_ACTOR.json", "DEV_JUDGE.json", "DEV_CONTROL_ACTOR.json",
                         "DEV_DONORS.json", "SELECTION.json", "MANIFEST.json")})
    assert outputs[0] == outputs[1]
