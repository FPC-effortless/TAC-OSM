"""Frozen LANGPERSIST096 outcome analysis; never imports a model or source file.

Only this evaluator receives target bytes. The primary estimand is a paired
within-checkpoint state intervention, not an architecture comparison. The
historical training FLOPs are unmatched. Five publisher clusters provide an
exploratory interval, not independently qualified generalization evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

CONDITIONS = ("carry", "reset", "donor", "corrupt")
GROUPS = ("positive", "negative", "uniform")
PUBLISHERS = ("django/django", "fastapi/fastapi", "Textualize/rich",
              "pydantic/pydantic", "pytest-dev/pytest")
SEEDS = (1701, 1702, 1703)
REQUIRED_MODELS = tuple((arm, seed) for arm in ("tac", "gru", "ssm")
                        for seed in SEEDS) + (("raw_history", 0),)
COST_FIELDS = ("setup_seconds", "acquisition_seconds", "query_seconds",
               "generation_seconds", "model_bytes", "state_payload_bytes",
               "serialized_metadata_bytes", "peak_rss_bytes")
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 96001


def _fingerprint(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((value.shape, value.dtype.str)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _validate_targets_records(targets, records):
    targets = np.asarray(targets)
    if (targets.ndim != 2 or targets.shape[1] != 4 or not len(targets)
            or targets.dtype.kind not in "iu" or np.any(targets < 0)
            or np.any(targets > 255)):
        raise ValueError("targets must be nonempty integer [N,4] original bytes")
    if not isinstance(records, (list, tuple)) or len(records) != len(targets):
        raise ValueError("records must align exactly with all target rows")
    identities = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError("each episode record must be a mapping")
        if record.get("group") not in GROUPS or record.get("publisher") not in PUBLISHERS:
            raise ValueError("unknown group or publisher; cohort cannot be silently changed")
        offset = record.get("target_offset")
        if (not isinstance(record.get("file"), str) or not record["file"]
                or type(offset) is not int or offset < 192):
            raise ValueError("file and integer target_offset >=192 required")
        if "index" in record and record["index"] != index:
            raise ValueError("episode index differs from its prediction row")
        identity = (record["publisher"], record["file"], offset)
        if identity in identities:
            raise ValueError("duplicate episode offset across cohort groups")
        identities.add(identity)
    groups = np.array([r["group"] for r in records])
    publishers = np.array([r["publisher"] for r in records])
    counts = {publisher: {group: int(np.sum((publishers == publisher) &
                                (groups == group))) for group in GROUPS}
              for publisher in PUBLISHERS}
    if any(n > 20 for row in counts.values() for n in row.values()):
        raise ValueError("cohort exceeds the frozen twenty-episode cell cap")
    return targets, groups, publishers, counts


def _validate_prediction(value, count, raw):
    if not isinstance(value, Mapping) or "log_probs" not in value or "greedy" not in value:
        raise ValueError("each model requires log_probs and greedy arrays")
    log_probs, greedy = np.asarray(value["log_probs"]), np.asarray(value["greedy"])
    condition_count = 1 if raw else 4
    if (log_probs.shape != (condition_count, count, 256)
            or log_probs.dtype.kind != "f" or not np.isfinite(log_probs).all()):
        raise ValueError("finite floating log_probs must have exact [conditions,N,256] shape")
    if (greedy.shape != (condition_count, count, 4) or greedy.dtype.kind not in "iu"
            or np.any(greedy < 0) or np.any(greedy > 255)):
        raise ValueError("greedy must be integer original-byte [conditions,N,4]")
    log_normalizer = np.logaddexp.reduce(log_probs.astype(np.float64), axis=-1)
    if np.max(np.abs(log_normalizer)) > 5e-5:
        raise ValueError("log_probs are not normalized first-byte distributions")
    if np.any(log_probs > 5e-5):
        raise ValueError("a log probability exceeds zero")
    if not np.array_equal(greedy[..., 0], log_probs.argmax(-1)):
        raise ValueError("greedy first byte differs from its scored distribution argmax")
    return log_probs, greedy


def _resource_report(key, supplied, count, condition_count):
    if supplied is None:
        supplied = {}
    if not isinstance(supplied, Mapping):
        raise ValueError("each model resource record must be a mapping")
    values, missing = {}, []
    for field in COST_FIELDS:
        value = supplied.get(field)
        if value is None:
            missing.append(field)
            values[field] = None
            continue
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number))
                or not math.isfinite(float(value)) or value < 0):
            raise ValueError("resource fields must be finite nonnegative numbers")
        if field.endswith("_bytes") and float(value) != int(value):
            raise ValueError("byte counts must be integral")
        values[field] = int(value) if field.endswith("_bytes") else float(value)
    seconds = (None if any(values[field] is None for field in COST_FIELDS[:4])
               else sum(values[field] for field in COST_FIELDS[:4]))
    condition_ledgers = {}
    if key[0] == "raw_history":
        condition_ledgers["carry"] = {
            "intervention_seconds": 0., "intervention_basis": "not applicable; raw control applies no intervention",
            "query_seconds": values["query_seconds"], "generation_seconds": values["generation_seconds"],
            "with_shared_setup_acquisition_seconds": seconds,
            "per_episode_with_shared_setup_acquisition_seconds": None if seconds is None else seconds / count}
    else:
        declared = supplied.get("per_condition_costs")
        if declared is not None and not isinstance(declared, Mapping):
            raise ValueError("per_condition_costs must be a condition mapping")
        if declared is not None and set(declared) != set(CONDITIONS):
            raise ValueError("per-condition timing names differ from frozen conditions")
        for condition in CONDITIONS:
            entry = {} if declared is None else declared[condition]
            if not isinstance(entry, Mapping):
                raise ValueError("each condition timing must be a mapping")
            timing = {}
            for field in ("intervention_seconds", "query_seconds", "generation_seconds"):
                value = entry.get(field)
                if value is None:
                    missing.append("per_condition_costs." + condition + "." + field)
                elif (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number))
                      or not math.isfinite(float(value)) or value < 0):
                    raise ValueError("condition timings must be finite nonnegative numbers")
                timing[field] = None if value is None else float(value)
            total = (None if any(value is None for value in timing.values()) or
                     values["setup_seconds"] is None or values["acquisition_seconds"] is None else
                     values["setup_seconds"] + values["acquisition_seconds"] + sum(timing.values()))
            condition_ledgers[condition] = {**timing,
                "with_shared_setup_acquisition_seconds": total,
                "per_episode_with_shared_setup_acquisition_seconds": None if total is None else total / count}
        if all(row["query_seconds"] is not None and row["intervention_seconds"] is not None
               for row in condition_ledgers.values()) and values["query_seconds"] is not None:
            measured = sum(row["query_seconds"] + row["intervention_seconds"]
                           for row in condition_ledgers.values())
            if not math.isclose(measured, values["query_seconds"], rel_tol=1e-7, abs_tol=1e-8):
                raise ValueError("condition query/intervention times do not reconcile with worker total")
        if all(row["generation_seconds"] is not None for row in condition_ledgers.values()) and values["generation_seconds"] is not None:
            if not math.isclose(sum(row["generation_seconds"] for row in condition_ledgers.values()),
                                values["generation_seconds"], rel_tol=1e-7, abs_tol=1e-8):
                raise ValueError("condition generation times do not reconcile with worker total")
    return {"arm": key[0], "seed": key[1], "status": "INCONCLUSIVE" if missing else "PASS",
            "missing_fields": missing, "declared_worker_costs": values,
            "recorded_total_seconds": seconds,
            "per_episode_all_conditions_seconds": None if seconds is None else seconds / count,
            "conditions_measured": condition_count,
            "per_condition_costs": condition_ledgers,
            "condition_cost_basis": "each condition ledger charges common setup/acquisition once; ledgers are not additive",
            "state_payload_bytes_per_episode": None if values["state_payload_bytes"] is None
                else values["state_payload_bytes"] / count,
            "serialized_metadata_bytes_per_episode": None if values["serialized_metadata_bytes"] is None
                else values["serialized_metadata_bytes"] / count,
            "scope": "worker totals include diagnostic wrappers; acquisition pool charged once; not serving or verification economics"}


def _summary_metrics(values, mask):
    return {name: float(np.mean(array[mask])) for name, array in values.items()}


def _cluster_interval(values, publishers, draws):
    """All rows and checkpoint seeds stay paired; resample publishers only."""
    cluster_means = np.array([np.mean(values[publishers == publisher])
                              for publisher in PUBLISHERS], dtype=np.float64)
    replicates = cluster_means[draws].mean(axis=1)
    lower, upper = np.quantile(replicates, (.025, .975))
    return {"mean": float(cluster_means.mean()), "ci95_lower": float(lower),
            "ci95_upper": float(upper),
            "publisher_effects": dict(zip(PUBLISHERS, map(float, cluster_means)))}


def _primary_report(per_episode, groups, publishers, cohort_complete):
    missing = [seed for seed in SEEDS if ("tac", seed) not in per_episode]
    if not cohort_complete or missing:
        return {"status": "INCONCLUSIVE", "reason": "complete frozen cohort and all three TAC seeds required",
                "missing_tac_seeds": missing,
                "nll_gate": {"status": "INCONCLUSIVE"},
                "exact4_utility_gate": {"status": "INCONCLUSIVE"},
                "positive_minus_negative_nll_interaction": {"status": "INCONCLUSIVE"}}
    mask = groups == "positive"
    selected_publishers = publishers[mask]
    reset_deltas, donor_deltas, exact_deltas, accuracy_deltas = [], [], [], []
    seed_effects = {}
    for seed in SEEDS:
        metrics = per_episode[("tac", seed)]
        reset = (metrics["first_byte_nll"][1] - metrics["first_byte_nll"][0])[mask]
        donor = (metrics["first_byte_nll"][2] - metrics["first_byte_nll"][0])[mask]
        exact = (metrics["exact4"][0] - metrics["exact4"][1])[mask]
        accuracy = (metrics["first_byte_accuracy"][0] - metrics["first_byte_accuracy"][1])[mask]
        reset_deltas.append(reset); donor_deltas.append(donor)
        exact_deltas.append(exact); accuracy_deltas.append(accuracy)
        seed_effects[str(seed)] = float(np.mean(reset))
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(PUBLISHERS),
                                size=(BOOTSTRAP_REPLICATES, len(PUBLISHERS)))
    effects = {}
    for name, deltas in (("reset_minus_carry_nll", reset_deltas),
                         ("donor_minus_carry_nll", donor_deltas),
                         ("carry_minus_reset_exact4", exact_deltas),
                         ("carry_minus_reset_first_byte_accuracy", accuracy_deltas)):
        effects[name] = _cluster_interval(np.mean(deltas, axis=0), selected_publishers, draws)
    all_nll_deltas = np.mean([per_episode[("tac", seed)]["first_byte_nll"][1] -
                              per_episode[("tac", seed)]["first_byte_nll"][0]
                              for seed in SEEDS], axis=0)
    interactions = np.array([
        np.mean(all_nll_deltas[(publishers == publisher) & (groups == "positive")]) -
        np.mean(all_nll_deltas[(publishers == publisher) & (groups == "negative")])
        for publisher in PUBLISHERS])
    interaction = _cluster_interval(interactions, np.array(PUBLISHERS), draws)
    nll, donor, exact = (effects[name] for name in ("reset_minus_carry_nll",
                             "donor_minus_carry_nll", "carry_minus_reset_exact4"))
    positive_publishers = sum(value > 0 for value in nll["publisher_effects"].values())
    positive_seeds = sum(value > 0 for value in seed_effects.values())
    criteria = {"nll_gain_at_least_0_01": nll["mean"] >= .01,
                "nll_ci95_lower_above_zero": nll["ci95_lower"] > 0,
                "at_least_four_publishers_positive": positive_publishers >= 4,
                "at_least_two_seeds_positive": positive_seeds >= 2,
                "donor_nll_ci95_lower_above_zero": donor["ci95_lower"] > 0}
    nll_status = "PASS" if all(criteria.values()) else "FAIL"
    return {"status": nll_status, "group": "positive", "effects": effects,
            "seed_nll_effects": seed_effects,
            "positive_publishers": positive_publishers, "positive_seeds": positive_seeds,
            "nll_gate": {"status": nll_status, "criteria": criteria},
            "exact4_utility_gate": {"status": "PASS" if exact["ci95_lower"] > 0 else "FAIL",
                                    "criterion": "paired exact4 gain ci95 lower >0"},
            "positive_minus_negative_nll_interaction": {"status": "DESCRIPTIVE_ONLY",
                "effect": interaction, "used_for_gating_or_selection": False,
                "limitation": "strata differ in source content; this does not isolate semantic identifier storage"},
            "bootstrap": {"unit": "publisher_cluster", "clusters": 5,
                "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
                "seed_handling": "average all three frozen checkpoint seeds per episode before resampling",
                "pairing": "all conditions and all episodes retained together within each sampled publisher",
                "limitation": "five clusters; exploratory internal uncertainty, not independent qualification"}}


def analyze(predictions_by_model, targets, records, costs=None):
    """Analyze already committed predictions; missing evidence never passes.

    Keys are (arm, seed). Neural arrays use carry/reset/donor/corrupt order;
    ('raw_history',0) uses one condition. Resource records use the same keys.
    This function performs no inference, training, model selection or file I/O.
    """
    report = {"schema": "tac-osm.langpersist096.analysis.v1", "study": "LANGPERSIST096",
        "status": "INCONCLUSIVE", "execution_level": "internal_research",
        "qualification": "INTERNALLY_MEASURED_NOT_INDEPENDENTLY_QUALIFIED",
        "optimizer_steps": 0, "protected_TEST": "NOT_OPENED",
        "architecture_superiority": "NOT_ELIGIBLE_HISTORICAL_TRAINING_FLOPS_UNMATCHED",
        "full_intelligence_theory": "OPEN", "verified_write_false_acceptance": "OPEN_NOT_MEASURED",
        "limits": ["lexical reuse and future byte prediction only",
                   "state interventions identify computational dependence, not world action effects",
                   "no autonomous structure discovery, transferable procedure or semantic verification claim",
                   "historically exposed publisher DEV; not blind TEST"],
        "metrics": [], "arm_means": [], "resources": []}
    try:
        targets, groups, publishers, counts = _validate_targets_records(targets, records)
        if not isinstance(predictions_by_model, Mapping) or not predictions_by_model:
            raise ValueError("nonempty prediction mapping required")
        if costs is None:
            costs = {}
        if not isinstance(costs, Mapping):
            raise ValueError("costs must be keyed by (arm,seed)")
        report["cohort"] = {"counts": counts, "episodes": len(targets),
            "complete": all(n == 20 for row in counts.values() for n in row.values()),
            "targets_sha256": _fingerprint(targets),
            "records_sha256": hashlib.sha256(json.dumps(list(records), sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode()).hexdigest()}
        per_episode = {}
        available = set()
        for key, value in predictions_by_model.items():
            if key not in REQUIRED_MODELS:
                raise ValueError("unexpected arm/seed outside frozen prediction matrix")
            log_probs, greedy = _validate_prediction(value, len(targets), key[0] == "raw_history")
            metrics = {
                "first_byte_nll": -np.take_along_axis(log_probs.astype(np.float64),
                    np.broadcast_to(targets[None, :, :1], (*log_probs.shape[:2], 1)), axis=-1)[..., 0],
                "first_byte_accuracy": (log_probs.argmax(-1) == targets[None, :, 0]).astype(float),
                "exact4": np.all(greedy == targets[None], axis=-1).astype(float),
            }
            if not all(np.isfinite(array).all() for array in metrics.values()):
                raise ValueError("nonfinite metric; no row may be silently dropped")
            per_episode[key] = metrics
            available.add(key)
            condition_names = ("raw_history",) if key[0] == "raw_history" else CONDITIONS
            for condition, name in enumerate(condition_names):
                condition_values = {metric: array[condition] for metric, array in metrics.items()}
                for group in GROUPS:
                    for publisher in ("ALL", *PUBLISHERS):
                        mask = (groups == group) & ((publishers == publisher) if publisher != "ALL" else True)
                        count = int(mask.sum())
                        if not count:
                            continue  # Explicitly missing cells remain in cohort counts and block gates.
                        report["metrics"].append({"arm": key[0], "seed": key[1],
                            "condition": name, "group": group, "publisher": publisher,
                            "episodes": count, **_summary_metrics(condition_values, mask)})
            resource = _resource_report(key, costs.get(key), len(targets), len(condition_names))
            resource["log_probs_sha256"] = _fingerprint(log_probs)
            resource["greedy_sha256"] = _fingerprint(greedy)
            report["resources"].append(resource)
        for arm in ("tac", "gru", "ssm"):
            if not all((arm, seed) in per_episode for seed in SEEDS):
                continue
            averaged = {metric: np.mean([per_episode[(arm, seed)][metric]
                        for seed in SEEDS], axis=0) for metric in
                        ("first_byte_nll", "first_byte_accuracy", "exact4")}
            for condition, name in enumerate(CONDITIONS):
                condition_values = {metric: array[condition] for metric, array in averaged.items()}
                for group in GROUPS:
                    for publisher in ("ALL", *PUBLISHERS):
                        mask = (groups == group) & ((publishers == publisher) if publisher != "ALL" else True)
                        count = int(mask.sum())
                        if count:
                            report["arm_means"].append({"arm": arm, "seeds": list(SEEDS),
                                "condition": name, "group": group, "publisher": publisher,
                                "episodes": count, **_summary_metrics(condition_values, mask)})
        report["missing_models"] = [{"arm": arm, "seed": seed}
                                     for arm, seed in REQUIRED_MODELS if (arm, seed) not in available]
        report["primary"] = _primary_report(per_episode, groups, publishers, report["cohort"]["complete"])
        complete_costs = not report["missing_models"] and all(
                    row["status"] == "PASS" for row in report["resources"])
        report["resource_accounting"] = {"status": "PASS" if complete_costs else "INCONCLUSIVE",
            "total_recorded_worker_seconds": None if not complete_costs else
                sum(row["recorded_total_seconds"] for row in report["resources"]),
            "peak_worker_rss_bytes": None if not complete_costs else
                max(row["declared_worker_costs"]["peak_rss_bytes"] for row in report["resources"]),
            "scope": "declared acquisition/query/generation/setup; no serving, training amortization or verification claims"}
        if not report["cohort"]["complete"]:
            report["status"] = "INSUFFICIENT_COHORT"
        elif report["missing_models"] or not complete_costs:
            report["status"] = "INCONCLUSIVE"
        elif report["primary"]["nll_gate"]["status"] == "PASS":
            report["status"] = "POSITIVE_INTERNAL_FINDING"
        else:
            report["status"] = "NEGATIVE_RESULT"
    except (ValueError, TypeError, IndexError, KeyError, OverflowError) as exc:
        report["status"] = "FAIL"
        report["analysis_error"] = str(exc)
    return report


def markdown_summary(report):
    """Compact narrative from the complete JSON; no winner selection."""
    lines = ["# LANGPERSIST096 frozen-checkpoint persistence", "",
             f"Outcome: **{report['status']}**.", "",
             "Internal publisher-DEV measurements; historical training FLOPs are unmatched. "
             "This is a within-checkpoint state study and establishes no architecture superiority.", ""]
    primary = report.get("primary", {})
    if "effects" in primary:
        lines += ["| Positive-cohort paired effect | Mean | Publisher-cluster 95% interval |",
                  "| --- | ---: | ---: |"]
        for name, effect in primary["effects"].items():
            lines.append(f"| {name} | {effect['mean']:.6f} | "
                         f"[{effect['ci95_lower']:.6f}, {effect['ci95_upper']:.6f}] |")
        lines += ["", f"NLL gate: **{primary['nll_gate']['status']}**. "
                  f"Exact-four utility gate: **{primary['exact4_utility_gate']['status']}**.", "",
                  "Intervals resample only five publisher clusters after averaging the three frozen TAC seeds. "
                  "They are exploratory and do not provide independent qualification.", ""]
        interaction = primary.get("positive_minus_negative_nll_interaction", {})
        if "effect" in interaction:
            value = interaction["effect"]
            lines += [f"Descriptive positive-minus-negative NLL interaction: {value['mean']:.6f} "
                      f"[{value['ci95_lower']:.6f}, {value['ci95_upper']:.6f}]. "
                      "This is excluded from gates and selection, and does not isolate semantic identifier storage.", ""]
    if report.get("analysis_error"):
        lines += ["Analysis failed: " + report["analysis_error"], ""]
    lines += ["Resource accounting: **" + report.get("resource_accounting", {}).get("status", "INCONCLUSIVE") + "**.", "",
              "Lexical reuse does not validate autonomous structure discovery, reusable procedures, "
              "action utility or semantic verification. Verified-write false acceptance remains OPEN.", ""]
    return "\n".join(lines)


def write_reports(report, json_path, markdown_path):
    """Exclusive-create both requested outputs; never replace earlier evidence."""
    json_path, markdown_path = Path(json_path), Path(markdown_path)
    if json_path.resolve() == markdown_path.resolve():
        raise ValueError("JSON and Markdown paths must differ")
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError("do not overwrite previous research evidence")
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    markdown = markdown_summary(report)
    for path in (json_path, markdown_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("x", encoding="utf-8") as handle:
        handle.write(text)
    with markdown_path.open("x", encoding="utf-8") as handle:
        handle.write(markdown)
