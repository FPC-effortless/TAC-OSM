"""Frozen final-vs-initial DEV and actually-seen FIT scoring. Zero optimization."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

from common import (ROOT, HERE, REGISTRATION_COMMIT, EVAL_PROTOCOL_SHA256, PROTOCOL_SHA,
                    Totals, batch_bytes, cohorts, digest, identity, load_plan, require,
                    unigram_control, verify_corpus)


def forward(model, tokens):
    result = model(tokens)
    return result[0] if isinstance(result, tuple) else result


def score_logits(logits, targets, torch):
    require(logits.shape == (*targets.shape, 256) and bool(torch.isfinite(logits).all()), "Invalid actual model output")
    values = logits.double()
    losses = torch.logsumexp(values, dim=-1) - values.gather(-1, targets[..., None]).squeeze(-1)
    require(bool(torch.isfinite(losses).all()), "Nonfinite actual next-byte NLL")
    correct = logits.argmax(-1).eq(targets)
    return losses, correct


def deadline(started, plan):
    require(time.monotonic()-started < plan["execution"]["worker_soft_deadline_minutes"]*60, "Registered inference wall budget exceeded")


def audit_boundary(model, data, ids, plan, torch, cohort_label="DEV"):
    x, y = batch_bytes(data, ids)
    x = torch.tensor(x, dtype=torch.long); y = torch.tensor(y, dtype=torch.long)
    tolerance = plan["evaluation"]["future_poison_tolerance"]
    with torch.inference_mode():
        clean = forward(model, x)
        require(torch.equal(clean, forward(model, x)), "Nondeterministic frozen predictions")
        changed = x.clone(); changed[:, 64:] = (changed[:, 64:] + 97) % 256
        future = forward(model, changed)
        require(torch.allclose(clean[:, :64], future[:, :64], **tolerance), "Future input changed earlier predictions")
        require(torch.allclose(clean[:2], forward(model, x[:2]), **tolerance), "Entity/batch interference")
        # A producer change to the final withheld target must not reach the input.
        name, offset = ids[0]; raw = bytearray(data[name][offset:offset+129]); raw[-1] = (raw[-1]+71) % 256
        require(list(raw[:-1]) == x[0].tolist(), "Withheld final target reached model inputs")
        losses, _ = score_logits(clean, y, torch)
        other = torch.nn.functional.cross_entropy(clean.double().reshape(-1, 256), y.reshape(-1), reduction="none").reshape_as(y)
        err = float((losses-other).abs().max())
        require(err <= plan["evaluation"]["scorer_agreement_atol"], "Independent metric arithmetic disagrees")
    return {"deterministic_prediction":"PASS", "future_suffix_invariance":"PASS",
            "entity_batch_isolation":"PASS", "withheld_target_producer_boundary":"PASS",
            "independent_cross_entropy_agreement":"PASS", "maximum_scorer_difference":err,
            "scope":"first original " + cohort_label + " batch, 128-byte contexts; bounded diagnostics, not global noninterference"}


def score(model, data, ids, records, label, plan, started, torch):
    total = Totals(records); begin = time.monotonic(); batch = plan["evaluation"]["batch_size"]
    with torch.inference_mode():
        for index in range(0, len(ids), batch):
            deadline(started, plan)
            chosen = ids[index:index+batch]
            x, y = batch_bytes(data, chosen)
            x = torch.tensor(x, dtype=torch.long); y = torch.tensor(y, dtype=torch.long)
            losses, correct = score_logits(forward(model, x), y, torch)
            sums = losses.sum(1).tolist(); hits = correct.sum(1).tolist()
            for (name, _), nll, hit in zip(chosen, sums, hits):
                total.add(name, nll, hit, 128)
            if index % (64*batch) == 0:
                print(json.dumps({"mode":label, "windows_completed":min(index+batch, len(ids)), "windows_total":len(ids)}), flush=True)
    result = total.finish()
    require(result["targets"] == len(ids)*128, "Scored target denominator differs")
    result.update({"window_ids_sha256":identity(ids), "scoring_cpu_wall_seconds":time.monotonic()-begin})
    result["teacher_forced_target_bytes_per_second"] = result["targets"]/result["scoring_cpu_wall_seconds"]
    return result


def run(args):
    plan = load_plan()
    key = (args.scale, args.arm, args.seed)
    lock = next((r for r in plan["checkpoint_locks"] if (r["scale"], r["arm"], r["seed"]) == key), None)
    require(lock is not None, "Unregistered checkpoint")
    require(not args.output.exists(), "Refuse overwrite of previous measurement")
    receipt, data = verify_corpus(args.source_receipt, args.corpus, plan)
    window_sets, seen = cohorts(receipt, plan, args.seed)
    require(digest(args.checkpoint) == lock["checkpoint_sha256"], "Frozen checkpoint bytes changed")
    original_receipt = args.checkpoint.parent / "FIT_RECEIPT.json"
    require(digest(original_receipt) == lock["receipt_sha256"], "Original training receipt changed")
    sys.path.insert(0, str(ROOT / "research/langcap103"))
    import preflight
    import torch
    require(torch.__version__ == plan["evaluation"]["torch_version"], "Original Torch CPU version required")
    torch.set_num_threads(plan["evaluation"]["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    require(all(payload[k] == value for k, value in (("scale", args.scale), ("arm", args.arm),
            ("seed", args.seed), ("steps", 2560), ("protocol_sha256", PROTOCOL_SHA),
            ("source_sha256", plan["source"]["source_sha256"]))), "Trained payload identity changed")
    original = json.loads((ROOT / "research/langcap105/OWNER_PROTOCOL.json").read_text())
    spec = next(r for r in original["models"][args.scale]["configurations"] if r["arm"] == args.arm)
    config = {k:v for k, v in spec.items() if k != "trainable_parameters"}
    torch.manual_seed(args.seed)
    model = preflight.build(config, original["models"][args.scale]["target_parameters"],
                            preflight.load_native(), preflight.comparator_classes(torch)).float().eval()
    require(sum(p.numel() for p in model.parameters()) == lock["trainable_and_deployed_parameters"], "Deployed size changed")
    initial = {k:v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(payload["model"], strict=True)
    require(all(p.dtype == torch.float32 and bool(torch.isfinite(p).all()) for p in model.parameters()), "Nonfinite/incorrect precision checkpoint")
    model.requires_grad_(False)
    trained_digest = preflight.parameter_digest(model)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    try:
        model.load_state_dict(initial, strict=True)
        initial_digest = preflight.parameter_digest(model)
        initial_dev = score(model, data, window_sets["DEV"], receipt["identity"]["files"], "INITIAL_DEV", plan, started, torch)
        (args.output / "INITIAL_DEV.json").write_text(json.dumps(initial_dev, indent=2, allow_nan=False)+"\n")
        require(preflight.parameter_digest(model) == initial_digest, "Initialization changed while scoring")
        model.load_state_dict(payload["model"], strict=True)
        del initial
        boundary = audit_boundary(model, data, window_sets["DEV"][:plan["evaluation"]["batch_size"]], plan, torch)
        dev = score(model, data, window_sets["DEV"], receipt["identity"]["files"], "TRAINED_DEV", plan, started, torch)
        (args.output / "TRAINED_DEV.json").write_text(json.dumps(dev, indent=2, allow_nan=False)+"\n")
        fit = score(model, data, window_sets["FIT_SEEN_REFERENCE"], receipt["identity"]["files"], "TRAINED_FIT_REFERENCE", plan, started, torch)
        (args.output / "TRAINED_FIT_REFERENCE.json").write_text(json.dumps(fit, indent=2, allow_nan=False)+"\n")
        require(preflight.parameter_digest(model) == trained_digest and digest(args.checkpoint) == lock["checkpoint_sha256"], "Frozen trained weights changed")
        require(all(p.grad is None and not p.requires_grad for p in model.parameters()), "Gradient/optimization path unexpectedly active")
        baseline = unigram_control(data, seen, window_sets["DEV"], receipt["identity"]["files"])
        result = {"schema":"langcap105.eval01.frozen-dev-result.v1", "status":"FROZEN_EXPLORATORY_DEV_SCORED",
            "scale":args.scale, "arm":args.arm, "seed":args.seed, "checkpoint_sha256":lock["checkpoint_sha256"],
            "protocol_sha256":EVAL_PROTOCOL_SHA256, "registration_commit":REGISTRATION_COMMIT,
            "original_training_run":plan["original_training_run"], "source_sha256":receipt["source_sha256"],
            "evaluator_commit":os.environ.get("EVALUATOR_COMMIT", "LOCAL_EXPLORATORY_EXECUTION"),
            "metrics":{"INITIAL_DEV":initial_dev, "TRAINED_DEV":dev, "TRAINED_FIT_REFERENCE":fit},
            "initial_to_trained_dev_nll_reduction":initial_dev["nll"]-dev["nll"],
            "fixed_final_dev_minus_fit_reference_nll":dev["nll"]-fit["nll"],
            "learning_transfers_to_dev_under_registered_criterion":dev["nll"] < initial_dev["nll"],
            "unmatched_diagnostic":baseline, "boundary_checks":boundary, "frozen_weights_unchanged":True,
            "optimizer_updates":0, "checkpoint_selection":False, "protected_test_access":False,
            "independent_scientific_qualification":False, "architecture_superiority_eligible":False,
            "runtime":{"torch_version":torch.__version__, "python":platform.python_version(),
                "cpu_threads":2, "platform":platform.platform(), "elapsed_scoring_and_checks_seconds":time.monotonic()-started,
                "scope":"teacher-forced CPU batch scoring, not autoregressive decode or measured FLOPs"},
            "limits":plan["limits"]}
        (args.output / "RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        print(json.dumps({"status":result["status"], "key":key, "dev_nll":dev["nll"], "dev_accuracy":dev["accuracy"],
                          "initial_dev_nll":initial_dev["nll"], "fit_reference_nll":fit["nll"]}), flush=True)
    except Exception as exc:
        (args.output / "FAILURE.json").write_text(json.dumps({"status":"FAILED_PRESERVED", "key":key,
            "type":type(exc).__name__, "message":str(exc), "checkpoint_sha256":lock["checkpoint_sha256"],
            "optimizer_updates":0, "protected_test_access":False}, indent=2)+"\n")
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-receipt", type=Path, required=True); p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scale", choices=("small", "medium"), required=True)
    p.add_argument("--arm", choices=("tac_associative_stack", "causal_transformer", "stacked_gru"), required=True)
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()
    try:
        run(args)
    except Exception as exc:
        if not args.output.exists():
            args.output.mkdir(parents=True)
            (args.output / "FAILURE.json").write_text(json.dumps({"status":"PREFLIGHT_FAILED_BEFORE_SCORING",
                "type":type(exc).__name__, "message":str(exc), "optimizer_updates":0}, indent=2)+"\n")
        raise
