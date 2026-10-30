#!/usr/bin/env python3
"""Independent, read-only outcome audit for the frozen LANGACQUIRE097 trial.

This file intentionally lives outside the fitted source tree. It imports no
model, training loop, scorer, or frozen analysis implementation. It only reads
saved predictions, licensed frozen observations/targets and source-bound receipts.
No prediction, fitting, threshold selection or source modification occurs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

FAMILIES = ("tac", "gru", "ssm", "transformer")
OBJECTIVES = ("full", "query")
SEEDS = (1701, 1702, 1703, 1704, 1705)
CONDITIONS = ("carry", "reset", "donor", "corrupt", "binding_shuffle")
COUNTS = {"tac": 100521, "gru": 100661, "ssm": 100404, "transformer": 100484}
EXPECTED_SOURCE = "d21c7e7b385ca164eac9046fafca43fc9a3762b1"
BOOTSTRAPS, BOOTSTRAP_SEED = 5000, 97001


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(part)
    return hasher.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def read_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "missing/nonregular JSON: " + str(path))
    return json.loads(path.read_text())


def read_npz(path):
    require(Path(path).is_file() and not Path(path).is_symlink(), "missing prediction NPZ")
    with np.load(path, allow_pickle=False) as saved:
        payload = {key: saved[key].copy() for key in saved.files if key != "metadata"}
        metadata = json.loads(str(saved["metadata"].item()))
    return payload, metadata


def grid(raw):
    """Independent compact-grid decoder, returning immutable geometry/content."""
    raw = bytes(raw)
    if b";" not in raw:
        return None
    prefix = raw[:raw.index(b";")]
    if len(prefix) < 3 or not (49 <= prefix[0] <= 54 and 49 <= prefix[1] <= 54):
        return None
    height, width = prefix[0] - 48, prefix[1] - 48
    colors = prefix[2:]
    if len(colors) != height * width or any(not 48 <= value <= 57 for value in colors):
        return None
    return height, width, tuple(value - 48 for value in colors)


def wb_distribution(history):
    """Counter-based independent Witten--Bell order-six reference."""
    history = bytes(history)
    probability = np.repeat(1. / 256., 256)
    for order in range(min(6, len(history)) + 1):
        suffix = history[len(history)-order:] if order else b""
        followers = Counter(history[index] for index in range(order, len(history))
                            if not order or history[index-order:index] == suffix)
        observations, alternatives = sum(followers.values()), len(followers)
        if observations:
            probability *= alternatives / (observations + alternatives)
            for token, count in followers.items():
                probability[token] += count / (observations + alternatives)
    require(np.isfinite(probability).all() and abs(float(probability.sum()) - 1.) < 1e-12,
            "independent Witten--Bell normalization failed")
    return probability


def wb_prediction(acquisition, query):
    # Every registered lexical observation has96 bytes, below1912-byte capacity.
    history = bytes(acquisition) + bytes(query)
    require(len(history) == 96, "unexpected lexical reference exposure")
    first = wb_distribution(history)
    generated = []
    current = first
    for position in range(4):
        token = int(np.argmax(current)); generated.append(token)
        if position < 3:
            history += bytes([token]); current = wb_distribution(history)
    return np.log(first), np.array(generated, dtype=np.uint8)


def last_demo(acquisition):
    demonstrations = [part for part in bytes(acquisition).split(b";") if b">" in part]
    require(bool(demonstrations), "ARC reference has no observed demonstrations")
    output = demonstrations[-1].partition(b">")[2] + b";"
    require(len(output) <= 40, "ARC observed-output baseline exceeds fixed budget")
    return output.ljust(40, b" ")


def interval(matrix, clusters):
    """Prescribed five-seed average followed by source/task cluster bootstrap."""
    matrix = np.asarray(matrix, dtype=float)
    supported = np.isfinite(matrix).all(axis=0)
    if not supported.all():
        answer = {"status": "INCONCLUSIVE", "cases": matrix.shape[1],
                  "available_cases": int(supported.sum()), "unavailable_cases": int((~supported).sum()),
                  "scope": "predeclared unavailable controls retained; supported subset is diagnostic only"}
        if supported.any():
            subset_groups = [label for label, keep in zip(clusters, supported) if keep]
            if len(set(subset_groups)) >= 2:
                answer["supported_subset_effect"] = interval(matrix[:, supported], subset_groups)
            else:
                answer["supported_subset_mean"] = float(matrix[:, supported].mean())
        return answer
    cases = [float(np.mean(matrix[:, index])) for index in range(matrix.shape[1])]
    groups = sorted(set(clusters))
    group_sums = np.array([sum(value for value, label in zip(cases, clusters) if label == group)
                           for group in groups])
    group_counts = np.array([clusters.count(group) for group in groups])
    sampled = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(groups), size=(BOOTSTRAPS, len(groups)))
    bootstrap = np.sum(group_sums[sampled], axis=1) / np.sum(group_counts[sampled], axis=1)
    lower, upper = np.quantile(bootstrap, [.025, .975])
    seed_means = np.mean(matrix, axis=1)
    group_means = group_sums / group_counts
    return {"status": "MEASURED", "mean": float(np.mean(cases)),
            "ci95_lower": float(lower), "ci95_upper": float(upper),
            "cases": len(cases), "clusters": len(groups),
            "cluster_effects": dict(zip(groups, map(float, group_means))),
            "positive_clusters": int(np.sum(group_means > 0)),
            "seed_effects": list(map(float, seed_means)),
            "positive_seeds": int(np.sum(seed_means > 0))}


def positive(effect, minimum=0., consistency=False):
    if effect["status"] != "MEASURED":
        return False
    result = effect["mean"] >= minimum and effect["mean"] > 0 and effect["ci95_lower"] > 0
    if consistency:
        result = result and effect["positive_seeds"] >= 4 and effect["positive_clusters"] >= math.ceil(.75 * effect["clusters"])
    return bool(result)


def compare(got, expected, location, differences):
    """Compare all independently reconstructed numeric/decision fields."""
    if isinstance(expected, dict):
        require(isinstance(got, dict), location + ": expected an object")
        for key, value in expected.items():
            require(key in got, location + ": missing " + key)
            compare(got[key], value, location + "." + key, differences)
    elif isinstance(expected, list):
        require(isinstance(got, list) and len(got) == len(expected), location + ": list size differs")
        for index, (left, right) in enumerate(zip(got, expected)):
            compare(left, right, location + f"[{index}]", differences)
    elif isinstance(expected, (float, np.floating)):
        require(isinstance(got, (int, float)) and not isinstance(got, bool)
                and math.isfinite(float(got)), location + ": finite number required")
        delta = abs(float(got) - float(expected)); differences.append(delta)
        require(delta <= 1e-10, location + f": numeric difference {delta}")
    else:
        require(got == expected, location + f": mismatch {got!r} versus {expected!r}")


def scored_rows(payload, metadata, actor, judge):
    family, objective, seed = (metadata["checkpoint"][key] for key in ("family", "objective", "seed"))
    require(metadata["conditions"] == list(CONDITIONS), "prediction condition order changed")
    require(metadata["case_ids"] == [case["id"] for case in actor], "prediction case order changed")
    logp, generated, available, lengths = (payload[key] for key in
                                         ("log_probs", "generated_bytes", "available", "lengths"))
    require(logp.shape == (5, 72, 256) and generated.shape == (5, 72, 40)
            and available.shape == (5, 72) and lengths.shape == (72,), "prediction geometry changed")
    require(logp.dtype == np.float32 and generated.dtype == np.uint8 and available.dtype == bool,
            "prediction dtypes changed")
    require(np.isfinite(logp).all() and float(np.max(np.abs(np.logaddexp.reduce(logp.astype(float), axis=-1)))) < 5e-5,
            "nonfinite or unnormalized first-byte distributions")
    main, binding = [], []
    for index, case in enumerate(actor):
        gold = judge[case["id"]]; target = bytes(gold["target_bytes"])
        require(int(lengths[index]) == case["generation_limit"] == (4 if case["domain"] == "lexical" else 40),
                "fixed generation budget changed")
        require(case["domain"] == gold["domain"], "actor and scorer domains differ")
        require(len(target) == 4 if gold["domain"] == "lexical" else grid(target) is not None,
                "invalid frozen target")
        for position, condition in enumerate(CONDITIONS):
            if condition == "binding_shuffle" and case["domain"] != "arc":
                require(not available[position, index], "binding control incorrectly scored lexical cases")
                continue
            expected_available = gold["donor_available"] if condition == "donor" else \
                gold["binding_shuffle_available"] if condition == "binding_shuffle" else True
            require(bool(available[position, index]) == expected_available, "frozen control availability changed")
            row = dict(family=family, objective=objective, seed=seed, domain=gold["domain"],
                       case_id=case["id"], cluster_id=gold["cluster_id"], group=gold["group"], condition=condition)
            if not expected_available:
                row.update(status="INCONCLUSIVE", nll=None, exact_match=None, format_valid=None)
            else:
                prediction = bytes(generated[position, index, :int(lengths[index])])
                require(prediction[0] == int(np.argmax(logp[position, index])), "greedy first byte differs from distribution")
                if gold["domain"] == "lexical":
                    valid, correct = True, prediction == target
                    require(not np.any(generated[position, index, 4:]), "unexpected generated bytes beyond lexical budget")
                else:
                    decoded = grid(prediction); valid = decoded is not None
                    correct = valid and decoded == grid(target)
                row.update(status="MEASURED", nll=-float(logp[position, index, target[0]]),
                           exact_match=int(correct), format_valid=int(valid))
            (binding if condition == "binding_shuffle" else main).append(row)
    return main, binding


def row_key(row):
    return tuple(row[key] for key in ("family", "objective", "seed", "condition", "case_id"))


def reconstruct_family(family, table, bindings, judge, raw, arc_reference):
    result = {"family": family, "summary": [], "domains": {}}
    for domain in ("lexical", "arc"):
        ids = sorted(case for case in judge if judge[case]["domain"] == domain)
        for objective in OBJECTIVES:
            for condition in CONDITIONS[:4]:
                for group in (("positive", "negative", "uniform") if domain == "lexical" else ("unknown_rule",)):
                    cases = [case for case in ids if judge[case]["group"] == group]
                    chosen = [table[(family, objective, seed, condition, case)] for seed in SEEDS for case in cases]
                    measured = [row for row in chosen if row["nll"] is not None]
                    result["summary"].append(dict(domain=domain, group=group, objective=objective,
                        condition=condition, cases=len(cases), seeds=5, selected_outcomes=len(chosen),
                        measured_outcomes=len(measured), inconclusive_outcomes=len(chosen)-len(measured),
                        status="MEASURED" if len(chosen) == len(measured) else "INCONCLUSIVE_FULL_COHORT",
                        **{metric: float(np.mean([row[metric] for row in measured])) if measured else None
                           for metric in ("nll", "exact_match", "format_valid")}))
        endpoint_cases = [case for case in ids if domain == "arc" or judge[case]["group"] == "positive"]
        clusters = [judge[case]["cluster_id"] for case in endpoint_cases]
        require(len(set(clusters)) == (4 if domain == "lexical" else 8), "registered cluster count changed")
        def matrix(objective, condition, metric):
            return np.array([[table[(family, objective, seed, condition, case)][metric]
                              for case in endpoint_cases] for seed in SEEDS], dtype=float)
        effects = {}
        for metric, short, sign in (("nll", "nll", -1), ("exact_match", "exact", 1), ("format_valid", "format", 1)):
            carry = matrix("query", "carry", metric)
            for name, objective, condition in (("query_minus_full", "full", "carry"),
                                               ("carry_minus_reset", "query", "reset"),
                                               ("carry_minus_donor", "query", "donor"),
                                               ("carry_minus_corrupt", "query", "corrupt")):
                effects[name + "_" + short] = interval(sign * (carry-matrix(objective, condition, metric)), clusters)
        if domain == "lexical":
            names = ("query_minus_full", "carry_minus_reset", "carry_minus_donor")
            nll_pass = all(positive(effects[name+"_nll"], .0 if name == "carry_minus_donor" else .01,
                                    name != "carry_minus_donor") for name in names)
            exact_pass = all(positive(effects[name+"_exact"]) for name in names)
            gates = {"first_byte_nll_learning_and_state": nll_pass, "exact4_learning_and_state": exact_pass}
            raw_effects = {}
            for metric, short, sign in (("nll", "nll", -1), ("exact_match", "exact", 1)):
                baseline = np.array([raw[case][metric] for case in endpoint_cases])[None, :]
                raw_effects["query_minus_raw_"+short] = interval(sign * (matrix("query", "carry", metric)-baseline), clusters)
            result["raw_history_reference"] = {"effects": raw_effects,
                "quality_hurdle": "PASS" if all(positive(eff) for eff in raw_effects.values()) else "FAIL"}
        else:
            learned = all(positive(effects[name+"_exact"], consistency=True)
                          for name in ("query_minus_full", "carry_minus_reset"))
            wrong_binding = np.array([[bindings[(family, "query", seed, "binding_shuffle", case)]["exact_match"]
                                       for case in endpoint_cases] for seed in SEEDS], dtype=float)
            binding_effect = interval(matrix("query", "carry", "exact_match")-wrong_binding, clusters)
            effects["carry_minus_binding_shuffle_exact"] = binding_effect
            attribution = positive(binding_effect, consistency=True)
            gates = {"unknown_rule_exact_learning": learned, "binding_specific_attribution": attribution,
                     "unknown_rule_exact_learning_and_state": learned and attribution}
            baseline = np.array([arc_reference[case]["exact_match"] for case in endpoint_cases])[None, :]
            reference_effect = interval(matrix("query", "carry", "exact_match")-baseline, clusters)
            result["arc_past_only_reference"] = {"effect": reference_effect,
                "quality_hurdle": "PASS" if positive(reference_effect) else "FAIL"}
        result["domains"][domain] = {"group": "positive" if domain == "lexical" else "unknown_rule",
                                     "effects": effects, "gates": {key: "PASS" if value else "FAIL" for key, value in gates.items()}}
    passed = all(value == "PASS" for domain in result["domains"].values() for value in domain["gates"].values())
    result["joint_learning_gate"] = "PASS" if passed else "FAIL"
    result["practical_reference_quality_gate"] = "PASS" if passed and \
        result["raw_history_reference"]["quality_hurdle"] == "PASS" and \
        result["arc_past_only_reference"]["quality_hurdle"] == "PASS" else "FAIL"
    return result


def audit(args):
    base = args.evaluation.resolve()
    if not (base / "RESULTS.json").is_file():
        matches = list(base.rglob("RESULTS.json"))
        require(len(matches) == 1, "point --evaluation at exactly one completed evaluation artifact")
        base = matches[0].parent
    data, repo = args.data.resolve(), args.repo.resolve()
    register = read_json(repo / "research/mathematical-engineering/langacquire097/REGISTER.json")
    require(register["training"]["steps"] == 256 and register["fit_enabled"] is True, "frozen fit recipe changed")
    require(sha256(data/"MANIFEST.json") == register["data"]["manifest_sha256"], "data manifest changed")
    for name, expected in register["data"]["artifact_sha256"].items():
        require(sha256(data/name) == expected, "frozen data changed: " + name)
    actor, judge = read_json(data/"DEV_ACTOR.json"), read_json(data/"DEV_JUDGE.json")
    require(len(actor) == len(judge) == 72 and {case["id"] for case in actor} == set(judge), "DEV identity matrix changed")
    require(Counter(row["domain"] for row in actor) == {"lexical": 64, "arc": 8}, "DEV domain counts changed")
    require(sum(not value["donor_available"] for value in judge.values()) == 5, "donor availability changed")
    require(all(row["binding_shuffle_available"] for row in judge.values() if row["domain"] == "arc"), "binding availability changed")
    execution = read_json(base/"EXECUTION.json")
    require(execution["status"] == "COMPLETED_INTERNAL_RESEARCH" and execution["source_commit"] == args.expected_source,
            "evaluation did not complete at the registered fitted source")
    require(execution["models_fitted"] == 40 and execution["optimizer_steps_during_evaluation"] == 0
            and execution["gpu_minutes"] == 0 and execution["protected_repository_TEST_opened"] is False
            and execution["general_intelligence_established"] is False
            and execution["cross_architecture_superiority_eligible"] is False, "execution scope/claim changed")
    checkpoint_seal = read_json(base/"ALL_CHECKPOINTS_SEALED.json")
    require(checkpoint_seal["models_sealed"] == 40 and checkpoint_seal["source_commit"] == args.expected_source
            and checkpoint_seal["before_first_DEV_prediction"] is True
            and checkpoint_seal["before_evaluator_opens_DEV_inputs"] is True, "all-forty pre-DEV seal missing")
    checkpoint_ids = {(row["family"], row["objective"], row["seed"]): row["checkpoint_sha256"]
                      for row in checkpoint_seal["models"]}
    expected_models = {(family, objective, seed) for family in FAMILIES for objective in OBJECTIVES for seed in SEEDS}
    require(set(checkpoint_ids) == expected_models and len(checkpoint_seal["models"]) == 40, "checkpoint seal matrix incomplete")
    query_expected = [{key: row[key] for key in ("id", "domain", "query_bytes", "generation_limit")} for row in actor]
    acquire_expected = [{key: row[key] for key in ("id", "domain", "acquisition_bytes")} for row in actor]
    require(read_json(base/"actor-inputs/queries.json") == query_expected
            and read_json(base/"actor-inputs/future-poison-queries.json") == query_expected
            and read_json(base/"actor-inputs/acquisition.json") == acquire_expected,
            "source-free actor inputs differ from the frozen observations")
    query_sha, acquire_sha = sha256(base/"actor-inputs/queries.json"), sha256(base/"actor-inputs/acquisition.json")
    predictions_seal = read_json(base/"ALL_PREDICTIONS_SEALED.json")
    require(predictions_seal["models"] == 40 and predictions_seal["before_target_score_computation"] is True,
            "target-score phase boundary missing")
    replay_seals = {(row["family"], row["objective"], row["seed"]): row for row in predictions_seal["replays"]}
    require(set(replay_seals) == expected_models, "forty replay seals missing")
    main, binding = [], []
    payload_hashes = {}
    for family, objective, seed in sorted(expected_models):
        identity = family, objective, seed
        folder = base/f"{family}-{objective}-{seed}"
        saved, meta = read_npz(folder/"original.npz"); repeated, repeat_meta = read_npz(folder/"replay.npz")
        for key in ("log_probs", "generated_bytes", "available", "lengths"):
            require(np.array_equal(saved[key], repeated[key]), "fresh-process replay mismatch: " + str(identity))
        receipt = replay_seals[identity]
        require(receipt["status"] == "PASS" and receipt["state_immutable"] is True
                and receipt["prediction_sha256"] == sha256(folder/"original.npz")
                and receipt["replay_sha256"] == sha256(folder/"replay.npz"), "replay bytes differ from seal")
        cp = meta["checkpoint"]
        require((cp["family"], cp["objective"], cp["seed"]) == identity
                and cp["status"] == "SEALED_BEFORE_DEV" and cp["source_commit"] == args.expected_source
                and cp["optimizer_steps"] == 256 and meta["optimizer_steps"] == 0
                and meta["checkpoint_sha256"] == checkpoint_ids[identity], "prediction checkpoint binding differs")
        require(meta["query_sha256"] == query_sha and meta["acquisition_sha256"] == acquire_sha
                and meta["donor_sha256"] == sha256(data/"DEV_DONORS.json")
                and meta["state_file_sha256"] == sha256(folder/"STATE.pt"), "actor/state provenance differs")
        require(meta["case_ids"] == repeat_meta["case_ids"] and cp == repeat_meta["checkpoint"], "replay metadata differs")
        rows, alternate = scored_rows(saved, meta, actor, judge)
        main.extend(rows); binding.extend(alternate)
        payload_hashes[f"{family}-{objective}-{seed}"] = receipt["prediction_sha256"]
    require(len(main) == 11520 and len(binding) == 320, "complete outcome matrix not reconstructed")
    require(sum(row["status"] == "INCONCLUSIVE" for row in main) == 200,
            "the200 unavailable ARC donor rows were dropped or altered")
    saved_main, saved_binding = read_json(base/"SCORED_ROWS.json"), read_json(base/"BINDING_ROWS.json")
    numeric_differences = []
    for name, recomputed, saved in (("main", main, saved_main), ("binding", binding, saved_binding)):
        a, b = {row_key(row): row for row in recomputed}, {row_key(row): row for row in saved}
        require(len(a) == len(recomputed) and len(b) == len(saved) and set(a) == set(b), name+": row identities differ")
        for key in a:
            compare(b[key], a[key], name+str(key), numeric_differences)
    raw_payload, raw_meta = read_npz(base/"RAW_HISTORY.npz")
    require(raw_meta["optimizer_steps"] == 0 and raw_meta["state_bytes_per_stream"] == 1928,
            "raw-reference state/fit budget differs")
    require(predictions_seal["raw_sha256"] == sha256(base/"RAW_HISTORY.npz"), "raw-reference artifact changed")
    by_id = {row["id"]: row for row in actor}; raw = {}; raw_max_difference = 0.
    require(set(map(str, raw_payload["ids"])) == {case for case in judge if judge[case]["domain"] == "lexical"},
            "raw lexical identity matrix differs")
    for index, case in enumerate(map(str, raw_payload["ids"])):
        observed = by_id[case]
        probabilities, generated = wb_prediction(observed["acquisition_bytes"], observed["query_bytes"])
        delta = float(np.max(np.abs(probabilities - raw_payload["log_probs"][index])))
        require(delta <= 1e-12 and np.array_equal(generated, raw_payload["generated"][index]), "independent raw control differs")
        raw_max_difference = max(raw_max_difference, delta)
        target = bytes(judge[case]["target_bytes"])
        # Saved probabilities feed exactly the reported endpoint; independently
        # rebuilt probabilities are checked above to a strict rounding tolerance.
        raw[case] = dict(nll=-float(raw_payload["log_probs"][index, target[0]]),
                         exact_match=int(bytes(generated) == target))
    arc_reference = {}
    require(set(map(str, raw_payload["arc_ids"])) == {case for case in judge if judge[case]["domain"] == "arc"},
            "ARC reference identity matrix differs")
    for index, case in enumerate(map(str, raw_payload["arc_ids"])):
        expected = last_demo(by_id[case]["acquisition_bytes"])
        require(expected == bytes(raw_payload["arc_generated"][index]), "ARC copy reference differs from observed history")
        decoded = grid(expected)
        arc_reference[case] = dict(case_id=case, cluster_id=judge[case]["cluster_id"],
                                   format_valid=int(decoded is not None),
                                   exact_match=int(decoded is not None and decoded == grid(judge[case]["target_bytes"])))
    arc_report = read_json(base/"ARC_COPY_REFERENCE.json")
    arc_report_rows = {row["case_id"]: row for row in arc_report["rows"]}
    for case, expected in arc_reference.items():
        compare(arc_report_rows[case], expected, "ARC reference "+case, numeric_differences)
    require(arc_report["correct"] == sum(row["exact_match"] for row in arc_reference.values())
            and arc_report["cases"] == 8, "ARC reference denominators differ")
    table = {row_key(row): row for row in main}; auxiliary = {row_key(row): row for row in binding}
    report = read_json(base/"RESULTS.json")
    require(report["status"] == "COMPLETED_INTERNAL_MEASUREMENT"
            and report["cross_architecture_superiority"] == "INELIGIBLE"
            and report["general_intelligence_established"] is False
            and report["case_count"] == 72 and report["prediction_rows"] == 11520
            and report["expected_prediction_rows"] == 11520, "reported scope/outcome counts differ")
    require(report["integrity_status"] == "PASS" and all(value == "PASS" for value in report["integrity"].values()),
            "analysis has missing/failed actual-pipeline integrity")
    family_reports = {item["family"]: item for item in report["families"]}
    require(set(family_reports) == set(FAMILIES), "reported family matrix incomplete")
    independent = {}
    for family in FAMILIES:
        reconstructed = reconstruct_family(family, table, auxiliary, judge, raw, arc_reference)
        compare(family_reports[family], reconstructed, family, numeric_differences)
        independent[family] = reconstructed
    # Verify each original resource matrix and forty-checkpoint chain without
    # invoking the source validator or executing any original model code.
    measured = {}
    for family in FAMILIES:
        registration = read_json(base/("FINAL_PARITY_"+family.upper()+".json"))
        require(registration["architecture_family"] == family and registration["no_architecture_superiority_claim"] is True,
                "within-family scope changed")
        profile = read_json(repo/registration["operator_profile_path"])
        require(sha256(repo/registration["operator_profile_path"]) == registration["operator_profile_sha256"], "profile changed")
        for model in registration["models"]:
            identity = family, model["name"], model["seed"]; actual = model["measured"]
            initial_profile = next(item for item in profile["profiles"] if item["seed"] == model["seed"] and item["objective"] == model["name"])
            require(identity in expected_models and identity not in measured, "resource model identity differs")
            require(actual["source_commit"] == args.expected_source and actual["status"] == "SEALED_BEFORE_DEV"
                    and actual["checkpoint_sha256"] == checkpoint_ids[identity]
                    and actual["trainable_parameters"] == actual["deployed_parameters"] == COUNTS[family]
                    and actual["actual_steps"] == 256 and actual["actual_examples"] == 1024
                    and actual["actual_dense_compute_tokens"] == 278528
                    and actual["actual_tokens"] == model["budget"]["tokens_seen"]
                    and actual["actual_sample_order_sha256"] == model["training_sample_order_sha256"], "actual fit budget/source differs")
            require(actual["actual_initial_model_sha256"] == actual["initial_model_sha256"] == initial_profile["weights_before"]
                    and actual["final_model_sha256"] != actual["initial_model_sha256"], "live initialization or weight update differs")
            require(len(actual["forward_backward_operator_step_sha256"]) == 256
                    and set(actual["forward_backward_operator_step_sha256"]) == {initial_profile["operator_sha256"]},
                    "actual forward/backward operator coverage incomplete")
            optimizer_steps = actual["optimizer_operator_step_sha256"]
            require(len(optimizer_steps) == 256 and optimizer_steps[0] == digest_json(actual["optimizer_operators_first"])
                    and set(optimizer_steps[1:]) == {digest_json(actual["optimizer_operators_warm"])}, "actual optimizer coverage incomplete")
            require(actual["replay"]["cold_reload_bitwise_equal"] is True
                    and all(actual["acquisition_gradient"][domain]["status"] == "PASS" for domain in ("lexical", "arc")),
                    "actual cold replay/acquisition gradient evidence missing")
            measured[identity] = actual
        require(execution["family_final_parity"][family]["status"] == "PASS", "final family gate not passed")
    require(set(measured) == expected_models, "forty final actual receipts missing")
    for family in FAMILIES:
        for seed in SEEDS:
            a, b = measured[(family, "full", seed)], measured[(family, "query", seed)]
            require(a["initial_model_sha256"] == b["initial_model_sha256"], "actual paired initialization differs")
            for key in ("actual_tokens", "actual_examples", "actual_steps", "actual_sample_order_sha256", "actual_dense_compute_tokens"):
                require(a[key] == b[key], "paired actual data/exposure differs")
            require(a["optimizer_operators_first"] == b["optimizer_operators_first"]
                    and a["optimizer_operators_warm"] == b["optimizer_operators_warm"], "paired optimizer operators differ")
    checkpoint_bytes = {"status": "INCONCLUSIVE", "reason": "--fits not supplied; seals/resources checked but original trained bytes not rehashed"}
    if args.fits:
        paths = list(args.fits.rglob("FIT_RECEIPT.json")); require(len(paths) == 40, "--fits must identify exactly forty assembled receipts")
        seen = set()
        for path in paths:
            actual = read_json(path); identity = actual["family"], actual["objective"], actual["seed"]
            require(identity in measured and identity not in seen, "original checkpoint matrix duplicated")
            require(actual == measured[identity], "original receipt differs from final parity evidence")
            require(sha256(path.parent/"MODEL.pt") == actual["checkpoint_sha256"]
                    and sha256(path.parent/"INITIAL_MODEL.pt") == actual["initial_checkpoint_sha256"], "original checkpoint bytes changed")
            seen.add(identity)
        require(seen == expected_models, "original checkpoint bytes incomplete")
        checkpoint_bytes = {"status": "PASS", "original_trained_and_initial_files_rehashed": 80}
    poison = read_json(base/"SOURCE_FUTURE_POISON.json")
    require(poison["status"] == "PASS_FIXED_SOURCE_OBSERVATION_NONINTERFERENCE" and poison["episodes"] == 72
            and poison["source_originals_unchanged"] is True and poison["eligibility_recomputed"] is False
            and len(poison["original_hashes"]) == len(poison["poisoned_hashes"]) == 12
            and all(value != poison["poisoned_hashes"][key] for key, value in poison["original_hashes"].items()),
            "source-future poison receipt incomplete")
    return {"schema": "langacquire097.independent-audit.v1", "status": "PASS_OUTCOME_AND_RECEIPT_AUDIT",
            "expected_source": args.expected_source, "audit_script_sha256": sha256(__file__),
            "evaluation_path": str(base), "data_path": str(data),
            "main_rows_recomputed": len(main), "binding_rows_recomputed": len(binding),
            "explicit_unavailable_main_rows": 200, "matched_model_replays": 40,
            "raw_lexical_cases_independently_recomputed": 64, "ARC_copy_cases_independently_recomputed": 8,
            "raw_max_log_probability_difference": raw_max_difference,
            "max_report_numeric_difference": max(numeric_differences, default=0.),
            "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAPS, "lexical_source_clusters": 4,
                          "ARC_task_clusters": 8, "model_seeds_averaged_within_case": 5},
            "forty_model_seal_and_actual_resource_receipts": "PASS", "original_checkpoint_bytes": checkpoint_bytes,
            "prediction_artifact_hashes": payload_hashes, "independent_family_results": independent,
            "neural_forward_calls": 0, "optimizer_steps": 0, "source_files_changed": False,
            "scope": "Independent recomputation of preregistered saved outcomes/receipts; no general-intelligence, safety or architecture-superiority certificate"}


def self_test():
    require(grid(b"221203;extra") == (2, 2, (1, 2, 0, 3)), "grid fixture")
    require(grid(b"2212;") is None and grid(b" 221203;") is None and grid(b"221203") is None, "invalid-grid fixture")
    require(last_demo(b"R;111>112;113>114;") == b"114;".ljust(40, b" "), "past-only reference fixture")
    require(abs(float(wb_distribution(b"abracadabra").sum())-1.) < 1e-12, "WB fixture")
    matrix = np.repeat([[1., -1., 1., -1.]], 5, axis=0)
    result = interval(matrix, ["a", "a", "b", "b"])
    require(result["mean"] == result["ci95_lower"] == result["ci95_upper"] == 0., "cluster-not-seed bootstrap fixture")
    matrix[:, 0] = np.nan
    result = interval(matrix, ["a", "a", "b", "b"])
    require(result["status"] == "INCONCLUSIVE" and result["unavailable_cases"] == 1, "unavailable-control fixture")
    print("Independent audit self-tests PASS; no real-data/model predictions executed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--data", type=Path, default=Path("/workspace/langacquire097-blinded"))
    parser.add_argument("--repo", type=Path, default=Path("/workspace/TAC-OSM-live"))
    parser.add_argument("--fits", type=Path)
    parser.add_argument("--expected-source", default=EXPECTED_SOURCE)
    parser.add_argument("--output", type=Path, default=Path("/workspace/langacquire097-independent-audit.json"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    parser.error("--evaluation is required") if args.evaluation is None else None
    require(not args.output.exists(), "never overwrite an earlier audit")
    try:
        result = audit(args)
    except Exception as error:
        result = {"schema": "langacquire097.independent-audit.v1", "status": "FAIL",
                  "error": type(error).__name__, "reason": str(error),
                  "neural_forward_calls": 0, "optimizer_steps": 0, "source_files_changed": False}
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False); handle.write("\n")
        print(json.dumps(result)); raise
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False); handle.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "main_rows_recomputed", "binding_rows_recomputed",
                                                   "max_report_numeric_difference", "raw_max_log_probability_difference")}))


if __name__ == "__main__":
    main()
