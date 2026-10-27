#!/usr/bin/env python3
"""Frozen original-source clock-specific content audit. No fitting or TEST."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tac_osm.research_execution import validate_internal_research  # noqa: E402
from tac_osm.phone_research_ops import decide, load_proposal  # noqa: E402
from tac_osm.historical_query_metrics import window_scores, compare_errors  # noqa: E402
from scripts.original_history_trace import (  # noqa: E402
    sha_file, verify_implementation, load_original_modules, tensor_sha, verify_causal_features,
)

SPEC = ROOT / "research/mathematical-engineering/history_query015/FAIR_COMPARISON.json"


def predict_batch(model, windows, owners, clocks):
    """Actual model boundary: observed histories and legal calendars only."""
    import torch
    if set(windows) != {"values", "valid", "age", "last"}:
        raise ValueError("raw/target/future fields forbidden at prediction boundary")
    b = len(owners)
    if (any(t.shape != (b, 88, 13) for t in windows.values())
            or clocks.shape != (b, 8, 6) or windows["valid"].dtype != torch.bool):
        raise ValueError("registered original history/clock shapes required")
    state = model.initial(b)
    for step in range(88):
        state = model.step(state, *(windows[k][:, step] for k in ("values", "valid", "age", "last")), owners)
    single = model.historic_event_logits(state, owners, clocks[:, :1])[:, 0]
    multiple = model.historic_event_logits(state, owners, clocks)
    zero = state._replace(associative=torch.zeros_like(state.associative))
    zero_read = model.historic_event_logits(zero, owners, clocks)
    if (not torch.isfinite(multiple).all() or not torch.isfinite(zero_read).all()
            or not torch.allclose(single, multiple[:, 0], rtol=0, atol=2e-6)):
        raise ValueError("nonfinite or inconsistent original N1/N8 query outputs")
    return multiple, zero_read, float((single - multiple[:, 0]).abs().max())


def predictions(model, d, rows, source):
    import torch
    result = []
    for start in range(0, len(rows), 16):
        batch = rows[start:start + 16]
        owners = torch.tensor([s for s, _ in batch], dtype=torch.long)
        anchors = torch.tensor([t for _, t in batch], dtype=torch.long)
        times = anchors[:, None] + torch.arange(-23, 65)[None, :]
        windows = {k: d[k][owners[:, None], times] for k in ("values", "valid", "age", "last")}
        query_times = anchors[:, None] + torch.arange(0, 64, 8)[None, :]
        clocks = source.calendar(query_times.flatten().tolist()).reshape(len(batch), 8, 6)
        p, zero, tolerance = predict_batch(model, windows, owners, clocks)
        for i, (owner, anchor) in enumerate(batch):
            current = torch.where(d["valid"][owner, anchor + 64], d["values"][owner, anchor + 64], d["last"][owner, anchor + 64])
            result.append({"owner": owner, "anchor_hour": anchor, "query_hour": anchor + 64,
                "target_hours": query_times[i].tolist(), "predictions": p[i].tolist(),
                "zero_read_predictions": zero[i].tolist(), "current_last_PM25": float(current[0]),
                "input_history_sha256": {k: tensor_sha(v[i]) for k, v in windows.items()},
                "query_clock_sha256": tensor_sha(clocks[i]), "N1_N8_max_abs_error": tolerance})
    return result


def execute(spec, root, archive):
    import torch
    if (spec["query_offsets_hours"] != list(range(0, 64, 8))
            or spec["query_lag_hours"] != 64 or spec["batch_size"] != 16):
        raise ValueError("unsupported frozen specificity protocol")
    root = verify_implementation(root, spec["implementation_commit"])
    if sha_file(archive) != spec["original_source_sha256"]:
        raise ValueError("original source archive hash mismatch")
    for path_key, hash_key in (("original_plan_path", "original_plan_sha256"), ("original_results_path", "original_results_sha256")):
        if sha_file(root / spec[path_key]) != spec[hash_key]:
            raise ValueError("original plan/results lineage changed")
    for row in spec["models"]:
        if sha_file(root / row["path"]) != row["sha256"]:
            raise ValueError("original checkpoint bytes changed")
    plan = json.loads((root / spec["original_plan_path"]).read_text())
    rows = plan["development_rows"]
    row_hash = hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
    if len(rows) != 117 or row_hash != spec["split_policy"]["cohort_sha256"]:
        raise ValueError("original source cohort mismatch")
    source, original = load_original_modules(root)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    d = original.source_observations(archive)
    verify_causal_features(d, source, max(t + 64 for _, t in rows))
    all_seeds = []
    with torch.no_grad():
        for seed in (18018, 18019):
            models, before, outputs, model_specs = {}, {}, {}, {}
            for arm in ("assoc_historic_query", "fast_slow_historic_query"):
                row = next(r for r in spec["models"] if r["seed"] == seed and r["arm"] == arm)
                checkpoint = torch.load(root / row["path"], map_location="cpu", weights_only=False)
                if (checkpoint["seed"] != seed or checkpoint["architecture"] != arm
                        or checkpoint["steps"] != 64 or checkpoint["examples_seen"] != 1024
                        or checkpoint["sample_order_SHA256"] != row["sample_order_sha256"]
                        or checkpoint["source_real_zip_SHA256"] != spec["original_source_sha256"]
                        or checkpoint["dev_original_source_cohort_SHA256"] != row_hash):
                    raise ValueError("original checkpoint source/order/seed/step mismatch")
                model = original.create_model(seed, arm)
                model.load_state_dict(checkpoint["model"], strict=True)
                model.eval()
                models[arm], model_specs[arm] = model, row
                before[arm] = {k: tensor_sha(v) for k, v in model.state_dict().items()}
                outputs[arm] = predictions(model, d, rows, source)
            # Both original architectures have finished prediction before this
            # scorer retrieves historical outcomes or target observation masks.
            scores = {arm: [] for arm in models}
            if len({row["sample_order_sha256"] for row in model_specs.values()}) != 1:
                raise ValueError("original comparator training sample order differs")
            for i, (owner, anchor) in enumerate(rows):
                target_hours = list(range(anchor, anchor + 64, 8))
                valid = d["valid"][owner, target_hours, 0].tolist()
                truth = [float(d["values"][owner, t, 0]) if ok else None for t, ok in zip(target_hours, valid)]
                for arm in models:
                    item = outputs[arm][i]
                    item.update(observed_target_mask_AFTER_prediction=valid, source_targets_AFTER_prediction=truth)
                    measured = window_scores(item["predictions"], truth, valid)
                    measured.update(
                        H64_zero_read_abs_error=abs(item["zero_read_predictions"][0] - truth[0]),
                        H64_train_prior_abs_error=abs(truth[0]),
                        H64_current_last_abs_error=abs(item["current_last_PM25"] - truth[0]))
                    scores[arm].append(measured)
            stats = {}
            for arm in models:
                case_scores = scores[arm]
                centered = [c for c in case_scores if c["centered_MSE"] is not None]
                null = sum(c["centered_null_MSE"] for c in centered)
                h64 = sum(c["H64_abs_error"] for c in case_scores) / len(case_scores)
                if abs(h64 - model_specs[arm]["legacy_H64_normalized_MAE"]) > 2e-6:
                    raise ValueError("frozen checkpoint H64 does not reproduce original source receipt")
                stats[arm] = {k: sum(c[k] for c in case_scores) / len(case_scores) for k in (
                    "H64_abs_error", "H64_wrong_clock_mean_abs_error", "H64_collapsed_query_abs_error",
                    "H64_zero_read_abs_error", "H64_train_prior_abs_error", "H64_current_last_abs_error",
                    "all_query_MAE", "circular_wrong_query_MAE", "collapsed_query_MAE",
                    "prediction_range", "observed_target_range")}
                stats[arm].update(centered_skill=1 - sum(c["centered_MSE"] for c in centered) / null if null > 0 else None,
                    centered_eligible_windows=len(centered), observed_targets=sum(c["observed_targets"] for c in case_scores),
                    legacy_H64_MAE_absolute_difference=abs(h64 - model_specs[arm]["legacy_H64_normalized_MAE"]))
            native = scores["assoc_historic_query"]
            baseline = scores["fast_slow_historic_query"]
            owners = [s for s, _ in rows]
            comparisons = {control: compare_errors(owners, [r[control] for r in native], [r["H64_abs_error"] for r in native])
                           for control in ("H64_wrong_clock_mean_abs_error", "H64_collapsed_query_abs_error")}
            eligible = [i for i in range(len(rows)) if native[i]["centered_MSE"] is not None and baseline[i]["centered_MSE"] is not None]
            comparisons["fastslow_centered_MSE"] = compare_errors([owners[i] for i in eligible],
                [baseline[i]["centered_MSE"] for i in eligible], [native[i]["centered_MSE"] for i in eligible])
            skill = stats["assoc_historic_query"]["centered_skill"]
            gate = skill is not None and skill > 0 and all(v["gate_pass"] for v in comparisons.values())
            for arm, model in models.items():
                if before[arm] != {k: tensor_sha(v) for k, v in model.state_dict().items()} or sha_file(root / model_specs[arm]["path"]) != model_specs[arm]["sha256"]:
                    raise ValueError("original model/checkpoint changed")
            all_seeds.append({"seed": seed, "metrics_normalized_log_PM25": stats,
                "log_PM25_spread": float(d["spread"][0]), "comparisons": comparisons,
                "registered_gate_pass": gate, "cases": outputs, "checkpoint_identity": model_specs})
    return {"models": all_seeds, "registered_all_seed_gate_pass": all(m["registered_gate_pass"] for m in all_seeds),
        "source_sha256": spec["original_source_sha256"], "cohort_sha256": row_hash,
        "checks": {"checkpoint_immutability": "PASS", "source_prefix_normalization_and_caches": "PASS",
            "feature_only_per_case_88_step_inference_boundary": "PASS", "N1_N8_original_query_equivalence": "PASS",
            "whole_model_PNDS_IEEE_AD_equivalence": "OPEN", "source_file_reloaded_future_poison": "INCONCLUSIVE_NOT_IMPLEMENTED"}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--implementation-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("immutable output already exists")
    report = {"status": "BLOCKED", "execution_level": "internal_research",
        "evidence_level": "internally_measured_not_independently_qualified",
        "scientific_qualification": False, "evaluation_completed": False}
    try:
        spec = json.loads(SPEC.read_text())
        validate_internal_research(spec)
        proposal = load_proposal(Path("research/phone_queue/history-query015.json"), ROOT)
        report["compute_decision"] = decide(proposal)
        if report["compute_decision"]["blockers"] or report["compute_decision"]["backend"] != "github_actions":
            raise ValueError("approved CPU-only internal plan required")
        receipt = json.loads((ROOT / spec["real_world_source"]["source_receipt"]).read_text())
        if receipt["archive_sha256"] != spec["original_source_sha256"] or receipt["license"] != spec["real_world_source"]["license"]:
            raise ValueError("recorded public source/license mismatch")
        report["plan_sha256"] = sha_file(SPEC)
        report["execution_files"] = {p: sha_file(ROOT / p) for p in (
            "scripts/audit_content_query_specificity.py", "scripts/original_history_trace.py",
            "src/tac_osm/historical_query_metrics.py", "src/tac_osm/research_execution.py")}
        report["status"] = "READY_INTERNAL_RESEARCH"
        if args.execute:
            if not args.source or not args.implementation_root:
                raise ValueError("original source and pinned implementation required")
            report["audit"] = execute(spec, args.implementation_root, args.source)
            report.update(status="MEASURED_INTERNAL_FROZEN_QUERY_DIAGNOSTIC", evaluation_completed=True)
    except Exception as exc:
        report.update(status="FAILED_INTERNAL_DIAGNOSTIC", reason=str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as fp:
        json.dump(report, fp, indent=2, sort_keys=True, allow_nan=False)
        fp.write("\n")
    print(json.dumps({k: report[k] for k in ("status", "evaluation_completed", "scientific_qualification")}))
    return 0 if report["status"] in ("READY_INTERNAL_RESEARCH", "MEASURED_INTERNAL_FROZEN_QUERY_DIAGNOSTIC") else 2


if __name__ == "__main__":
    raise SystemExit(main())
