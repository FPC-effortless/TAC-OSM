"""Actual FIT-only, zero-optimizer operator profile for LANGACQUIRE097.

The source protocol and cohort must be committed before this CPU job. It
profiles the registered first batch for each seed, comparing the exact same
initialized source model under full/query masks. No DEV or checkpoint selection
is performed, and no optimizer update is executed or simulated on model weights.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from tac_osm.research_objective_budget import OperatorLedger, canonical, digest, source_bindings, verify_same_ledger

import importlib.util
_spec = importlib.util.spec_from_file_location("langacquire097_profile_resources", HERE / "resources.py")
R = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = R
_spec.loader.exec_module(R)
M = R.load_local("models")
O = R.load_local("objective")
TRAIN = R.load_local("train")


def optimizer_contract(training):
    return {"name": "AdamW", "learning_rate": training["learning_rate"],
            "weight_decay": training["weight_decay"], "grad_clip": training["grad_clip"],
            "betas": training.get("betas", [0.9, 0.999]), "eps": training.get("eps", 1e-8), "amsgrad": False,
            "foreach": False, "clip_foreach": False, "fused": False}


def create_profiles(data, config, output, source_commit, families=None):
    if output.exists():
        raise FileExistsError("never overwrite an operator-profile attempt")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    register = R.load_register(config)
    R.verify_source(source_commit, config)
    manifest, rows = TRAIN.validate_data(data, register)
    training = register["training"]
    sources = source_bindings(ROOT, M.source_digests())
    families = tuple(families or R.FAMILIES)
    if not families or any(f not in R.FAMILIES for f in families):
        raise ValueError("registered model families required")
    output.mkdir(parents=True)
    summary = []
    for family in families:
        profiles = []
        for seed in register["seeds"]:
            planned = TRAIN.schedule(rows, seed, training["steps"], training["batch_size"])
            planned_ids = [[rows[i]["id"] for i in batch] for batch in planned]
            schedule_sha = R.json_digest(planned_ids)
            scheduled_tokens = sum(sum(len(rows[i][key]) for key in
                                   ("acquisition_bytes", "query_bytes", "target_bytes")) - 1
                                   for batch in planned for i in batch)
            inputs, targets, query_mask, valid_mask, exposure = TRAIN.collate(rows, planned[0], training["context"])
            paired = []
            initial_file = output / f"INIT_{family}_{seed}.pt"
            for objective in register["objectives"]:
                model = M.make_model(family, seed).train()
                counts = R.assert_counts(model, M.PARAMETERS[family])
                before = R.model_digest(model)
                if not initial_file.exists():
                    payload = {"schema": "langacquire097.initial-state.v1", "family": family, "seed": seed,
                               "model": R.cpu_tree(model.state_dict()), "model_sha256": before, "optimizer_steps": 0}
                    with initial_file.open("xb") as handle:
                        torch.save(payload, handle)
                initial_sha = R.sha256(initial_file)
                model.zero_grad(set_to_none=True)
                started = time.perf_counter()
                with OperatorLedger("forward_backward") as ledger:
                    loss = O.sequence_loss(model, inputs, targets, query_mask,
                                           objective=objective, valid_mask=valid_mask,
                                           owners=tuple(f"train-{i}" for i in range(training["batch_size"])))
                    loss.loss.backward()
                elapsed = time.perf_counter() - started
                gradients = TRAIN.gradient_receipt(model)
                after = R.model_digest(model)
                if before != after:
                    raise ValueError("a no-fit profiler changed model weights")
                row = {"family": family, "seed": seed, "objective": objective,
                       "trainable_parameters": counts["trainable"], "deployed_parameters": counts["deployed"],
                       "weights_before": before, "weights_after": after,
                       "initial_state_file": initial_file.name, "initial_state_sha256": initial_sha,
                       "fit_sha256": R.sha256(data / "FIT.json"), "schedule_sha256": schedule_sha,
                       "planned_steps": training["steps"], "planned_examples_seen": len(planned) * training["batch_size"],
                       "planned_tokens_seen": scheduled_tokens,
                       "planned_dense_compute_tokens": len(planned) * training["batch_size"] * training["context"],
                       "input_shape": list(inputs.shape), "operators": ledger.rows(),
                       "operator_sha256": ledger.sha256(), "elapsed_seconds": elapsed,
                       "all_gradients_connected_and_finite": gradients["all_connected_and_finite"],
                       "all_parameter_gradients_nonzero": gradients["all_nonzero"],
                       "parameter_tensor_count": len(gradients["parameters"]),
                       "actual_first_batch_exposure": exposure,
                       "supervised_targets": loss.supervised_targets,
                       "optimizer_steps": 0}
                paired.append(row)
                profiles.append(row)
                print(f"profiled {family} seed={seed} objective={objective} {elapsed:.3f}s", flush=True)
            verify_same_ledger(paired[0], paired[1], family + " full/query prefit")
        profile = {"schema": "tac-osm.objective-operator-profile.v1", "family": family,
                   "source_commit": source_commit,
                   "source_sha256": register["data"]["source_provenance_sha256"],
                   "recipe_sha256": digest(training), "implementation_sha256": digest(sources),
                   "training_recipe": training,
                   "source_hashes": sources, "optimizer_contract": optimizer_contract(training),
                   "optimizer_steps": 0, "profiles": profiles,
                   "hardware": R.runtime_hardware(),
                   "scope": "complete ATen signatures/shapes/counts for actual FIT forward/backward; absolute hardware FLOPs unknown",
                   "optimizer_prefit_basis": "same fixed AdamW configuration, source model, parameter shapes and connected gradients; actual first/warm ledgers required postfit"}
        path = output / ("PROFILE_" + family + ".json")
        R.write_json(path, profile)
        summary.append({"family": family, "path": str(path), "sha256": R.sha256(path),
                        "implementation_sha256": profile["implementation_sha256"],
                        "recipe_sha256": profile["recipe_sha256"],
                        "source_sha256": profile["source_sha256"], "optimizer_contract": profile["optimizer_contract"]})
    R.write_json(output / "PROFILE_INDEX.json", {"profiles": summary, "optimizer_steps": 0})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", "--register", dest="config", type=Path, default=HERE / "REGISTER.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--family", choices=R.FAMILIES, action="append")
    args = parser.parse_args()
    create_profiles(args.data, args.config, args.output, args.source_commit, args.family)


if __name__ == "__main__":
    main()
