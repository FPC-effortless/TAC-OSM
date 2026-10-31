"""Evaluate sealed LANGMATH101 checkpoint without fitting or judge inputs.

The final integrated package supplies common.py/data.py/metrics.py/arc_controls.py.
Full language distributions and free-running episodic distributions/generations
are written and hashed BEFORE DEV_JUDGE is parsed for target-side scoring.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import fields, is_dataclass, replace
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CORE = ("carry", "reset", "donor", "direct_history", "query_only")
SELECTIVE = ("reset_preserve_clock", "context_reset", "lexical_reset", "hidden_reset",
             "fast_reset", "slow_reset", "banks_reset", "top_level_reset", "slot0_reset",
             "upward_off", "downward_off", "color_relabel",
             "code0", "code1", "force_byte", "force_discrete", "force_latent")


class UnsupportedIntervention(ValueError):
    pass


def load_local(name):
    key = "langmath101_evaluation_" + name
    if key in sys.modules: return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, HERE / (name + ".py"))
    mod = importlib.util.module_from_spec(spec); sys.modules[key] = mod; spec.loader.exec_module(mod)
    return mod


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f: json.dump(value, f, indent=2, allow_nan=False); f.write("\n")


def clone_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().clone()
    if isinstance(value, dict): return {k: clone_tree(v) for k, v in value.items()}
    if isinstance(value, tuple): return tuple(clone_tree(v) for v in value)
    if isinstance(value, list): return [clone_tree(v) for v in value]
    return value


def clone_state(state, *, owners=None):
    if not is_dataclass(state): raise UnsupportedIntervention("model state is not an explicit dataclass")
    updates = {f.name: clone_tree(getattr(state, f.name)) for f in fields(state)}
    if owners is not None: updates["owners"] = tuple(owners)
    return replace(state, **updates)


def state_bytes(state):
    tensors = getattr(state, "tensors", {})
    tensor_bytes = sum(t.numel() * t.element_size() for t in tensors.values() if isinstance(t, torch.Tensor))
    pending = getattr(state, "pending_bytes", ())
    raw_bytes = sum(len(r) for r in pending if isinstance(r, bytes))
    token_count = sum(len(r) for r in getattr(state, "cache_tokens", ()))
    return {"tensor_bytes": tensor_bytes, "non_tensor_raw_prefix_bytes": raw_bytes,
            "non_tensor_cached_token_ID_count": token_count,
            "metadata_and_Python_container_overhead": "NOT_MEASURED"}


def _float_state_schema(state):
    if not is_dataclass(state): return False
    names = {f.name for f in fields(state)}
    return names <= {"owners", "clock", "tensors", "arm"} and all(
        isinstance(v, torch.Tensor) and v.dtype in (torch.float32, torch.float64)
        for v in getattr(state, "tensors", {}).values())


def _bank_fields(state):
    keys = set(getattr(state, "tensors", {})); pairs = []
    if {"memory", "mass"} <= keys: pairs.append(("memory", "mass"))
    for key in sorted(keys):
        if key.endswith(".matrix") and key[:-len("matrix")] + "mass" in keys:
            pairs.append((key, key[:-len("matrix")] + "mass"))
    for matrix, mass in pairs:
        a, b = state.tensors[matrix], state.tensors[mass]
        if a.ndim != 4 or b.ndim != 3 or a.shape[1] != 2 or b.shape[1] != 2 or a.shape[-1] != b.shape[-1]:
            raise UnsupportedIntervention("registered associative matrix/mass geometry differs")
    return pairs


def selective_reset(model, state, condition):
    """Explicit physical fields only; no inferred architecture semantics."""
    out = clone_state(state); t = out.tensors
    if condition == "reset_preserve_clock":
        if not _float_state_schema(state):
            raise UnsupportedIntervention("clock-preserving reset undefined for raw-history/pending metadata state")
        return replace(out, tensors={k: torch.zeros_like(v) for k, v in t.items()}), "all floating tensors zero; logical clock preserved"
    if condition in ("fast_reset", "slow_reset", "banks_reset", "slot0_reset"):
        pairs = _bank_fields(state)
        if not pairs: raise UnsupportedIntervention("no registered associative bank fields")
        for matrix, mass in pairs:
            if condition == "slot0_reset":
                t[matrix][..., 0] = 0; t[mass][..., 0] = 0
            else:
                for j in ((0,) if condition == "fast_reset" else (1,) if condition == "slow_reset" else (0, 1)):
                    t[matrix][:, j] = 0; t[mass][:, j] = 0
        return out, ("first associative address coordinate removed from matrix+mass; not discovered executable-structure truth"
                     if condition == "slot0_reset" else "selected coupled matrix+mass reset at query boundary; query may repopulate it")
    if condition == "top_level_reset":
        if not getattr(model, "hierarchy", False): raise UnsupportedIntervention("model has no registered hierarchy")
        prefix = str(model.depth - 1) + "."
        selected = [k for k in t if k.startswith(prefix)]
    elif condition == "context_reset": selected = [k for k in t if k == "context" or k.endswith(".context")]
    elif condition == "lexical_reset": selected = [k for k in t if k == "lexical"]
    elif condition == "hidden_reset": selected = [k for k in t if k == "hidden"]
    else: raise UnsupportedIntervention("unknown selective intervention")
    if not selected: raise UnsupportedIntervention("requested state dimension not present")
    for key in selected:
        if t[key].dtype not in (torch.float32, torch.float64): raise UnsupportedIntervention("selected field is not floating state")
        t[key] = torch.zeros_like(t[key])
    return out, "query-boundary reset of explicit fields: " + ",".join(selected)


@contextmanager
def communication_lesion(model, condition):
    hooks = []
    if condition in ("upward_off", "downward_off"):
        name = "upward" if condition == "upward_off" else "downward"
        modules = getattr(model, name, None)
        if not getattr(model, "hierarchy", False) or modules is None or not len(modules):
            raise UnsupportedIntervention("registered hierarchical " + name + " projections absent")
        for module in modules:
            hooks.append(module.register_forward_hook(lambda _m, _args, output: torch.zeros_like(output)))
    try: yield
    finally:
        for hook in hooks: hook.remove()


def adaptive_kwargs(model, condition):
    if condition.startswith("force_"):
        if getattr(model, "mode", None) != "a5_adaptive":
            raise UnsupportedIntervention("forced routes require the registered adaptive controller")
        return {"force": condition[6:]}
    if condition in ("code0", "code1"):
        if getattr(model, "mode", None) not in ("a5_discrete", "a5_adaptive"):
            raise UnsupportedIntervention("discrete codebook intervention absent")
        return {"code_override": int(condition[-1])}
    return {}


def call(model, tokens, state, owners, extra, costs):
    kwargs = dict(state=state, owners=owners, **extra)
    if "execution" in inspect.signature(model.forward).parameters: kwargs["execution"] = "inference"
    started = time.perf_counter(); logits, state = model(tokens, **kwargs)
    costs["model_seconds"] += time.perf_counter() - started
    costs["forward_calls"] = costs.get("forward_calls", 0) + 1
    costs["processed_input_byte_positions"] = costs.get("processed_input_byte_positions", 0) + int(tokens.numel())
    if logits.shape != (*tokens.shape, 256) or not bool(torch.isfinite(logits).all()):
        raise ValueError("finite aligned next-byte predictions required")
    receipt = getattr(model, "last_receipt", None)
    if receipt:
        for key in ("source_bytes", "recurrent_updates", "candidate_recurrent_calls", "byte_readouts"):
            costs[key] = costs.get(key, 0) + int(receipt.get(key, 0))
        for event in receipt.get("events", ()):
            if event.get("code") is not None:
                costs.setdefault("committed_code_ID_counts", Counter())[str(event["code"])] += 1
    return logits, state


def advance(model, raw, state, window, owners, extra, costs):
    """The stateless-window arm gets only its last four legal observed bytes."""
    if not raw: raise ValueError("cannot predict without an observed byte")
    if getattr(model, "arm", None) == "stateless_window4":
        outputs = []
        for byte in raw:
            window = (window + bytes([byte]))[-4:]
            tokens = torch.tensor([list(window)], dtype=torch.long)
            logits, state = call(model, tokens, None, owners, extra, costs)
            outputs.append(logits[:, -1:])
        return torch.cat(outputs, 1), state, window
    tokens = torch.tensor([list(raw)], dtype=torch.long)
    logits, state = call(model, tokens, state, owners, extra, costs)
    return logits, state, b""


def one_prediction(model, actor, condition, donor_actor=None):
    owners = ("eval-stream",); costs = {"model_seconds": 0.0}; extra = adaptive_kwargs(model, condition)
    original_state = None; window = b""
    with communication_lesion(model, condition):
        if condition not in ("direct_history", "query_only", "donor"):
            _, original_state, window = advance(model, bytes(actor["acquisition_bytes"]), None, window, owners, extra, costs)
        if condition == "donor":
            if donor_actor is None: raise UnsupportedIntervention("legal same-domain/same-clock donor unavailable")
            _, original_state, window = advance(model, bytes(donor_actor["acquisition_bytes"]), None, b"", owners, extra, costs)
            original_state = clone_state(original_state, owners=owners)
        elif condition == "reset":
            # Raw Transformer/BPE histories and pending metadata require a true
            # fresh start, not invalid nonzero-clock empty token histories.
            original_state = model.start(owners); window = b""
        elif condition in SELECTIVE and condition not in ("upward_off", "downward_off", "color_relabel", "code0", "code1", "force_byte", "force_discrete", "force_latent"):
            original_state, note = selective_reset(model, original_state, condition)
            costs["intervention_definition"] = note
        q = bytes(actor["query_bytes"])
        if condition == "direct_history": q = bytes(actor["acquisition_bytes"]) + q
        logits, current, window = advance(model, q, original_state, window, owners, extra, costs)
        costs["prefill_model_seconds"] = costs["model_seconds"]
        costs["prefill_forward_calls"] = costs["forward_calls"]
        lp = logits[:, -1].log_softmax(-1)[0].detach().numpy().copy()
        token = int(logits[:, -1].argmax(-1)); generated = bytearray()
        for j in range(actor["generation_limit"]):
            generated.append(token)
            if j + 1 < actor["generation_limit"]:
                logits, current, window = advance(model, bytes([token]), current, window, owners, extra, costs)
                token = int(logits[:, -1].argmax(-1))
        costs["decode_model_seconds"] = costs["model_seconds"] - costs["prefill_model_seconds"]
        costs["decode_forward_calls"] = costs["forward_calls"] - costs["prefill_forward_calls"]
        costs["free_generated_bytes"] = len(generated)
        costs["autoregressive_generated_byte_updates"] = max(0, len(generated) - 1)
        costs["generated_byte_updates_per_decode_second"] = ((len(generated) - 1) / costs["decode_model_seconds"] if costs["decode_model_seconds"] > 0 else None)
        costs["timing_scope"] = "CPU model calls only, one fixed continuation; first output byte comes from prefill; no production throughput claim"
        costs.update(state_bytes(current)); costs["external_stateless_window_bytes"] = len(window)
        return lp, bytes(generated), costs


def _load_model(C, payload):
    meta = payload["metadata"]
    if callable(getattr(C, "load_model", None)):
        out = C.load_model(payload)
        model = out[0] if isinstance(out, tuple) else out
    else:
        model = C.make_model(meta["arm"], meta.get("tokenizer_receipt"))
        model.load_state_dict(payload["model"], strict=True)
    model.eval()
    if callable(getattr(C, "state_digest", None)) and meta.get("model_sha256"):
        if C.state_digest(model) != meta["model_sha256"]: raise ValueError("checkpoint actual model digest differs")
    return model, meta


def _language_predictions(model, bundle, output, language_distributions=True):
    distributions = []; ids = []; costs = {"model_seconds": 0.0}; metadata = []
    for start in range(0, len(bundle.dev_windows), 4):
        group = bundle.dev_windows[start:start + 4]
        # Last byte is omitted from model observations; target-side source
        # join/CE is computed only after all prediction files are sealed.
        raw = [bundle.dev_corpus[r["file"]][r["offset"]:r["offset"] + 257] for r in group]
        tokens = torch.tensor([list(r[:-1]) for r in raw], dtype=torch.long)
        owners = tuple("language-" + str(start + j) for j in range(len(group)))
        logits, _ = call(model, tokens, None, owners, {}, costs)
        distributions.append(logits.log_softmax(-1).detach().numpy().copy()); ids.extend(r["id"] for r in group); metadata.extend(group)
    lp = np.concatenate(distributions, 0)
    path = output / "LANGUAGE_PREDICTIONS.npz"
    np.savez_compressed(path, ids=np.array(ids), log_probs=lp)
    return {"file": path.name, "sha256": sha(path), "cases": len(ids), "target_bytes_per_case": 256,
            "full_distributions_saved": True, "costs": costs, "records": metadata}


def run(args):
    if args.output.exists(): raise FileExistsError("retain original evaluation outputs")
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.threads); torch.use_deterministic_algorithms(True)
    C, D, MET, ARC = (load_local(n) for n in ("common", "data", "metrics", "arc_controls"))
    actual_checkpoint_sha = sha(args.checkpoint)
    if args.sha256 and actual_checkpoint_sha != args.sha256: raise ValueError("sealed checkpoint file SHA differs")
    if not getattr(args, "fits_seal", None):
        raise ValueError("complete 138-fit SEALED_ALL_FITS.json barrier required before DEV")
    fits_seal = json.loads(Path(args.fits_seal).read_text())
    if fits_seal.get("status") != "SEALED_ALL_FITS_BEFORE_DEV" or fits_seal.get("jobs") != 138:
        raise ValueError("complete original all-fit seal required")
    seal_keys = [(r.get("arm"), r.get("seed"), r.get("lane")) for r in fits_seal.get("checkpoints", ())]
    exact_keys = {(arm, seed, lane) for arm in C.arm_names() for seed in C.SEEDS
                  for lane in ("parameter_exposure_matched", "counted_arithmetic_matched")}
    if len(seal_keys) != 138 or len(set(seal_keys)) != 138 or set(seal_keys) != exact_keys:
        raise ValueError("all-fit seal does not cover exact preregistered 138-job matrix")
    seal_directory = Path(args.fits_seal).parent
    for filename, key in (("FINAL_MEASURED_MANIFEST.json", "measured_manifest_sha256"),
                          ("FINAL_OWNER_GATE.json", "final_gate_sha256")):
        if sha(seal_directory / filename) != fits_seal.get(key):
            raise ValueError("actual final all-fit gate/manifest differs from seal")
    final_gate = json.loads((seal_directory / "FINAL_OWNER_GATE.json").read_text())
    if final_gate.get("phase") != "final" or final_gate.get("status") != "PASS_OWNER_TWO_LANE_INTERNAL" or final_gate.get("model_rows") != 138:
        raise ValueError("actual complete measured final gate did not pass")
    if fits_seal.get("source_hashes") != C.source_hashes():
        raise ValueError("evaluation source differs from all-fit source seal")
    sealed_matches = [r for r in fits_seal.get("checkpoints", ()) if r.get("checkpoint_sha256") == actual_checkpoint_sha]
    if len(sealed_matches) != 1:
        raise ValueError("checkpoint not uniquely included in all-fit seal")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model, checkpoint_meta = _load_model(C, payload)
    sealed_key = sealed_matches[0]
    if any(checkpoint_meta.get(k) != sealed_key.get(k) for k in ("arm", "seed", "lane")):
        raise ValueError("checkpoint identity differs from all-fit seal")
    digest_before = C.state_digest(model)
    optimizer_steps = sorted({float(v["step"]) for v in (payload.get("optimizer") or {}).get("state", {}).values() if "step" in v})
    del payload
    bundle = D.load_data(args.fit, args.data)
    rows = list(bundle.dev_actors); indices = {r["id"]: i for i, r in enumerate(rows)}
    donor_specs = {r["id"]: r for r in bundle.dev_donors}; donors = []
    for row in rows:
        spec = donor_specs[row["id"]]
        if not spec["available"]: donors.append(None); continue
        other = rows[indices[spec["donor_id"]]]
        if other["id"] == row["id"] or other["domain"] != row["domain"] or len(other["acquisition_bytes"]) != len(row["acquisition_bytes"]):
            raise ValueError("donor is not original same-domain/same-observation-clock evidence")
        donors.append(other)
    conditions = tuple(args.conditions.split(",")) if args.conditions not in ("core", "full") else CORE + (SELECTIVE if args.conditions == "full" else ())
    if len(set(conditions)) != len(conditions) or any(c not in CORE + SELECTIVE for c in conditions):
        raise ValueError("unique registered condition names required")
    lp = np.full((len(conditions), len(rows), 256), np.nan, dtype=np.float32)
    generated = np.zeros((len(conditions), len(rows), 40), dtype=np.uint8)
    available = np.zeros((len(conditions), len(rows)), dtype=np.bool_)
    per_condition = []; unsupported = []
    with torch.inference_mode():
        language = _language_predictions(model, bundle, args.output)
        for ci, condition in enumerate(conditions):
            costs = []; started = time.perf_counter()
            # Missing dimensions are structural UNKNOWNs. Do not waste a full
            # acquisition pass merely to rediscover absent bank/level fields.
            structural_reason = None
            if condition in ("reset_preserve_clock", "context_reset", "lexical_reset", "hidden_reset", "fast_reset", "slow_reset", "banks_reset", "top_level_reset", "slot0_reset"):
                try: selective_reset(model, model.start(("schema-probe",)), condition)
                except UnsupportedIntervention as exc: structural_reason = str(exc)
            for i, original_actor in enumerate(rows):
                if structural_reason is not None:
                    unsupported.append({"condition": condition, "id": original_actor["id"], "status": "INCONCLUSIVE_UNSUPPORTED", "reason": structural_reason}); continue
                actor = original_actor
                if condition == "color_relabel":
                    if actor["domain"] != "arc":
                        unsupported.append({"condition": condition, "id": actor["id"], "status": "NOT_APPLICABLE", "reason": "color relabel is ARC-only"}); continue
                    actor = ARC.derived_actor(actor)
                try:
                    values, raw, receipt = one_prediction(model, actor, condition, donors[i])
                except UnsupportedIntervention as exc:
                    unsupported.append({"condition": condition, "id": original_actor["id"], "status": "INCONCLUSIVE_UNSUPPORTED", "reason": str(exc)}); continue
                lp[ci, i] = values; generated[ci, i, :len(raw)] = np.frombuffer(raw, dtype=np.uint8); available[ci, i] = True
                costs.append({"id": original_actor["id"], **receipt})
            per_condition.append({"condition": condition, "observed_cases": len(rows), "measured_cases": len(costs), "seconds": time.perf_counter() - started,
                                  "boundary": "fresh source state for reset; resets/slot lesions at acquisition-query boundary; communication and adaptive-route/code lesions through complete episode",
                                  "costs": costs})
            print(json.dumps({"arm": checkpoint_meta.get("arm"), "condition": condition, "measured": len(costs), "seconds": per_condition[-1]["seconds"]}), flush=True)
    path = args.output / "EPISODIC_PREDICTIONS.npz"
    np.savez_compressed(path, conditions=np.array(conditions), ids=np.array([r["id"] for r in rows]), log_probs=lp, generated=generated, available=available)
    model_after = C.state_digest(model)
    if model_after != digest_before: raise ValueError("evaluation changed original learned weights")
    metadata = {"schema": "langmath101.evaluation.v1", "checkpoint_sha256": actual_checkpoint_sha,
                "expected_checkpoint_digest_provided": bool(args.sha256), "checkpoint_metadata": checkpoint_meta,
                "all_fits_seal_sha256": sha(args.fits_seal), "all_fits_seal_status": fits_seal["status"],
                "model_digest_before": digest_before, "model_digest_after": model_after,
                "recorded_optimizer_slot_steps": optimizer_steps, "evaluation_optimizer_steps": 0,
                "language": language, "conditions": per_condition, "unsupported": unsupported,
                "classification": "previously exposed public DEV; internal measurement, no independent qualification",
                "state_note": "State includes all native dataclass fields; donors retain receiving owners. Adaptive pending/raw/token metadata is copied. A1window4 uses an explicitly costed last-four-byte input window. Transformer is raw-prefix recurrence and resets at clock0.",
                "source_identity": bundle.data_identity, "torch": torch.__version__, "threads": args.threads, "platform": platform.platform(),
                "raw_prediction_files": {path.name: sha(path), language["file"]: language["sha256"]},
                "no_new_primary_data": True, "protected_TEST_access": False}
    save(args.output / "PREDICTIONS_SEALED.json", {"before_target_scoring": True, "previously_exposed_DEV": True, **metadata})
    # Target-side stage begins only here. No target is passed to one_prediction.
    judge = D.load_judge(bundle); summaries = []; all_scores = []
    for ci, condition in enumerate(conditions):
        scoring_judge = judge
        if condition == "color_relabel":
            scoring_judge = {key: dict(value) for key, value in judge.items()}
            for actor in rows:
                if actor["domain"] == "arc":
                    scoring_judge[actor["id"]]["target_bytes"] = list(ARC.relabel_target_for_scoring(judge[actor["id"]]["target_bytes"]))
        raw = [bytes(generated[ci, i, :r["generation_limit"]]) for i, r in enumerate(rows)]
        scored = MET.score_episodes(lp[ci], raw, rows, scoring_judge, available[ci])
        for row in scored: row["condition"] = condition
        all_scores.extend(scored)
        summaries.append({"condition": condition, "cohorts": MET.cohort_summary(scored)})
    with np.load(args.output / language["file"], allow_pickle=False) as values:
        language_lp = values["log_probs"]
    targets = np.array([list(bundle.dev_corpus[r["file"]][r["offset"] + 1:r["offset"] + 257]) for r in bundle.dev_windows], dtype=np.int64)
    language_summary = []; language_windows = []
    for i, record in enumerate(bundle.dev_windows):
        rlp = language_lp[i].astype(np.float64); target = targets[i]
        chosen = np.take_along_axis(rlp, target[:, None], -1).squeeze(-1)
        correct = int((rlp.argmax(-1) == target).sum())
        language_windows.append({**record, "target_bytes": int(target.size),
                                 "NLL_nats_per_byte": -float(chosen.mean()),
                                 "bits_per_byte": -float(chosen.mean()) / np.log(2),
                                 "nextbyte_accuracy": correct / int(target.size), "correct_bytes": correct})
    for modality in sorted({r["modality"] for r in bundle.dev_windows}):
        selected = np.array([r["modality"] == modality for r in bundle.dev_windows])
        sel_lp = language_lp[selected].astype(np.float64); sel_y = targets[selected]
        target_log_probs = np.take_along_axis(sel_lp, sel_y[..., None], -1).squeeze(-1)
        language_summary.append({"modality": modality, "windows": int(selected.sum()), "target_bytes": int(sel_y.size),
                                 "NLL_nats_per_byte": -float(target_log_probs.mean()), "bits_per_byte": -float(target_log_probs.mean()) / np.log(2),
                                 "nextbyte_accuracy": float((sel_lp.argmax(-1) == sel_y).mean())})
    save(args.output / "SCORED_EPISODES.json", all_scores)
    save(args.output / "RESULTS.json", {"arm": checkpoint_meta.get("arm"), "seed": checkpoint_meta.get("seed"),
                                       "lane": checkpoint_meta.get("lane"), "checkpoint_sha256": actual_checkpoint_sha,
                                       "language": language_summary, "language_windows": language_windows, "episodic": summaries,
                                       "unknown_dimensions": unsupported, "new_training": False,
                                       "qualification": "internal_exposed_DEV_not_architecture_superiority_or_protected_replication"})
    print(json.dumps({"status": "COMPLETED_INTERNAL_EVALUATION", "arm": checkpoint_meta.get("arm"), "conditions": len(conditions), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--sha256"); p.add_argument("--fits-seal", type=Path, required=True); p.add_argument("--fit", type=Path, required=True); p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--threads", type=int, default=2)
    p.add_argument("--conditions", default="full", help="core, full or explicit comma-separated registered conditions")
    run(p.parse_args())
