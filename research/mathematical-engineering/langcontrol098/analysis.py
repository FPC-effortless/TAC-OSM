"""Prospective within-family objective effects; actor inputs never enter here.

NLL is the first target-byte score. Exact outcomes use fixed-budget, free-running
generation. Source owners/tasks, rather than bytes or seed copies, are the
bootstrap units. No result from this module establishes architecture superiority.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from collections.abc import Mapping

import numpy as np

SEEDS = (1701, 1702, 1703, 1704, 1705)
FAMILIES = ("tac", "gru", "ssm", "transformer")
OBJECTIVES = ("full", "query")
CONDITIONS = ("carry", "reset", "donor", "corrupt")
DOMAINS = ("lexical", "arc")
REQUIRED_INTEGRITY = ("source_provenance", "split_isolation", "prediction_boundary",
                      "fresh_process_replay", "parameter_budget", "training_exposure",
                      "within_family_compute", "gradient_to_acquisition")
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 97001


def _contract(records):
    if isinstance(records, Mapping):
        records = [{**value, "case_id": key} for key, value in records.items()]
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("frozen DEV case identities are required")
    result = {}
    for record in records:
        metadata = record.get("metadata", {})
        case = record.get("case_id", record.get("id"))
        domain = record.get("domain")
        cluster = record.get("cluster_id", metadata.get("cluster_id"))
        group = record.get("group", metadata.get("group", "arc"))
        if (not isinstance(case, str) or not case or case in result or domain not in DOMAINS
                or not isinstance(cluster, str) or not cluster
                or record.get("partition", "DEV") != "DEV"
                or (domain == "lexical" and group not in ("positive", "negative", "uniform"))):
            raise ValueError("invalid, duplicated or non-DEV frozen case")
        donor = metadata.get("donor_available", record.get("donor_available", True))
        binding = metadata.get("binding_shuffle_available", record.get("binding_shuffle_available", False))
        if type(donor) is not bool or type(binding) is not bool:
            raise ValueError("frozen control availability must be boolean")
        result[case] = dict(domain=domain, cluster_id=cluster,
                            group="unknown_rule" if domain == "arc" else group,
                            donor_available=donor, binding_shuffle_available=binding)
    if {r["domain"] for r in result.values()} != set(DOMAINS):
        raise ValueError("both registered learning domains are required")
    for domain in DOMAINS:
        selected = [r for r in result.values() if r["domain"] == domain
                    and (domain != "lexical" or r["group"] == "positive")]
        if len({r["cluster_id"] for r in selected}) < 2:
            raise ValueError("at least two original source/task clusters per endpoint required")
    return result


def _number(value, name):
    if (not isinstance(value, (int, float, np.number)) or isinstance(value, (bool, np.bool_))
            or not math.isfinite(float(value))):
        raise ValueError("finite numeric " + name + " required")
    return float(value)


def _binary(value, name):
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    number = _number(value, name)
    if number not in (0., 1.):
        raise ValueError("binary " + name + " required")
    return number


def _effect(matrix, clusters):
    """matrix[seed, case]: average seeds, then resample complete source clusters."""
    matrix = np.asarray(matrix, dtype=np.float64)
    case_effect = matrix.mean(0)
    names = sorted(set(clusters))
    sums = np.array([case_effect[np.asarray(clusters) == name].sum() for name in names])
    counts = np.array([sum(cluster == name for cluster in clusters) for name in names])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = rng.integers(0, len(names), size=(BOOTSTRAP_SAMPLES, len(names)))
    draws = sums[sampled].sum(1) / counts[sampled].sum(1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    seed_effects = matrix.mean(1)
    cluster_effects = sums / counts
    return {"status": "MEASURED", "mean": float(case_effect.mean()), "ci95_lower": float(lo), "ci95_upper": float(hi),
            "cases": int(matrix.shape[1]), "clusters": len(names),
            "cluster_effects": dict(zip(names, map(float, cluster_effects))),
            "positive_clusters": int((cluster_effects > 0).sum()),
            "seed_effects": list(map(float, seed_effects)),
            "positive_seeds": int((seed_effects > 0).sum())}


def _positive(effect, *, minimum=0.0, consistency=False):
    if effect.get("status") != "MEASURED":
        return False
    passed = effect["mean"] >= minimum and effect["mean"] > 0 and effect["ci95_lower"] > 0
    if consistency:
        passed &= (effect["positive_seeds"] >= 4 and
                   effect["positive_clusters"] >= math.ceil(0.75 * effect["clusters"]))
    return bool(passed)


def _available_effect(matrix, clusters):
    matrix = np.asarray(matrix, dtype=float)
    supported = np.isfinite(matrix).all(0)
    if supported.all():
        return _effect(matrix, clusters)
    result = {"status": "INCONCLUSIVE", "cases": matrix.shape[1],
              "available_cases": int(supported.sum()), "unavailable_cases": int((~supported).sum()),
              "scope": "predeclared unavailable controls retained; supported subset is diagnostic only"}
    if supported.any():
        subset_clusters = [name for name, keep in zip(clusters, supported) if keep]
        if len(set(subset_clusters)) >= 2:
            result["supported_subset_effect"] = _effect(matrix[:, supported], subset_clusters)
        else:
            result["supported_subset_mean"] = float(matrix[:, supported].mean())
    return result


def analyze(rows, expected_cases, integrity=None, *, families=FAMILIES, seeds=SEEDS,
            raw_rows=None, binding_rows=None, arc_reference_rows=None):
    """Analyze an entire frozen matrix, or return explicit FAIL/INCONCLUSIVE.

    Required row fields: family, objective, seed, domain, case_id, cluster_id,
    condition, nll, exact_match, format_valid; lexical rows also have group.
    Integrity statuses are caller-supplied actual-pipeline receipts, not inferred
    from these numeric arrays. A unit-test ledger supplies no empirical evidence.
    """
    report = {"schema": "langcontrol098.analysis.v1", "status": "INCONCLUSIVE",
              "objective_semantics": {"full": "lambda=1 prefix plus future",
                                      "query": "lambda=0 future only",
                                      "future_coefficients_fixed": True,
                                      "domain_weights": {"lexical": 0.5, "arc": 0.5},
                                      "reduction": "fixed_domain_case_segment_means",
                                      "normalize_by_total_coefficient_mass": False},
              "evidence_level": "INTERNALLY_MEASURED_NOT_INDEPENDENTLY_QUALIFIED",
              "cross_architecture_superiority": "INELIGIBLE", "general_intelligence_established": False,
              "primary_family": "tac", "families": [],
              "bootstrap": {"unit": "original_source_owner_or_original_ARC_task",
                            "samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED,
                            "seed_handling": "average five model seeds inside each case before resampling",
                            "scope": "exploratory DEV intervals with few original clusters"},
              "limits": ["ARC is an authored benchmark diagnostic, not naturally observed field outcomes",
                         "First-byte NLL and exact free-running outcomes have separate gates",
                         "External scorer verification is not an acquired autonomous verifier",
                         "False-write acceptance, continual improvement and full intelligence remain OPEN"]}
    try:
        contract = _contract(expected_cases)
        families, seeds = tuple(families), tuple(seeds)
        if (not families or len(set(families)) != len(families)
                or any(f not in FAMILIES for f in families) or seeds != SEEDS):
            raise ValueError("registered families and all five frozen seeds required")
        table = {}
        for row in rows:
            case = row.get("case_id")
            key = (row.get("family"), row.get("objective"), row.get("seed"),
                   row.get("condition"), case)
            if (case not in contract or key in table or key[0] not in families
                    or key[1] not in OBJECTIVES or key[2] not in seeds
                    or type(key[2]) is not int or key[3] not in CONDITIONS):
                raise ValueError("unregistered, duplicated or invalid prediction row")
            for field in ("domain", "cluster_id"):
                if row.get(field) != contract[case][field]:
                    raise ValueError("case metadata differs from frozen contract")
            row_group = row.get("group", "unknown_rule")
            if row.get("domain") == "arc" and row_group == "arc":
                row_group = "unknown_rule"
            if row_group != contract[case]["group"]:
                raise ValueError("case cohort differs from frozen contract")
            available = not (key[3] == "donor" and not contract[case]["donor_available"])
            if not available:
                if row.get("status") != "INCONCLUSIVE" or any(row.get(k) is not None for k in ("nll", "exact_match", "format_valid")):
                    raise ValueError("unavailable donor must remain explicitly inconclusive")
                table[key] = dict(nll=None, exact_match=None, format_valid=None)
                continue
            if row.get("status", "MEASURED") not in ("MEASURED", "PASS"):
                raise ValueError("available case lacks a measured outcome")
            nll = _number(row.get("nll"), "first-byte NLL")
            exact = _binary(row.get("exact_match"), "exact match")
            valid = _binary(row.get("format_valid"), "format validity")
            if nll < 0 or exact not in (0., 1.) or valid not in (0., 1.) or exact > valid:
                raise ValueError("invalid scored outcome")
            table[key] = dict(nll=nll, exact_match=exact, format_valid=valid)
        expected = len(families) * len(OBJECTIVES) * len(seeds) * len(CONDITIONS) * len(contract)
        report["case_count"] = len(contract)
        report["prediction_rows"] = len(table)
        report["expected_prediction_rows"] = expected
        if len(table) != expected:
            report["reason"] = "incomplete frozen model/condition/case matrix; no rows dropped"
            return report
        integrity = integrity or {}
        report["integrity"] = {key: integrity.get(key, "INCONCLUSIVE") for key in REQUIRED_INTEGRITY}
        clean = all(status == "PASS" for status in report["integrity"].values())
        failed_integrity = any(status == "FAIL" for status in report["integrity"].values())
        report["integrity_status"] = "PASS" if clean else "FAIL" if failed_integrity else "INCONCLUSIVE"
        if failed_integrity:
            report["status"] = "FAIL"
            report["reason"] = "actual-pipeline integrity failure; no capability inference"
            return report
        bindings = {}
        if binding_rows is not None:
            for row in binding_rows:
                case = row.get("case_id")
                key = (row.get("family"), row.get("objective"), row.get("seed"), case)
                if (case not in contract or contract[case]["domain"] != "arc" or key in bindings
                        or key[0] not in families or key[1] not in OBJECTIVES or key[2] not in seeds
                        or row.get("condition") != "binding_shuffle" or row.get("domain") != "arc"
                        or row.get("cluster_id") != contract[case]["cluster_id"]):
                    raise ValueError("invalid structural-binding control row")
                if not contract[case]["binding_shuffle_available"]:
                    if row.get("status") != "INCONCLUSIVE" or any(row.get(k) is not None for k in ("nll", "exact_match", "format_valid")):
                        raise ValueError("unavailable binding control must remain inconclusive")
                    bindings[key] = dict(nll=None, exact_match=None, format_valid=None)
                else:
                    nll = _number(row.get("nll"), "binding-control NLL")
                    exact = _binary(row.get("exact_match"), "binding-control exact match")
                    valid = _binary(row.get("format_valid"), "binding-control format validity")
                    if nll < 0 or exact > valid:
                        raise ValueError("invalid binding-control outcome")
                    bindings[key] = dict(nll=nll, exact_match=exact, format_valid=valid)
            expected_binding = len(families) * len(OBJECTIVES) * len(seeds) * sum(value["domain"] == "arc" for value in contract.values())
            if len(bindings) != expected_binding:
                raise ValueError("incomplete structural-binding control matrix")
        raw = {}
        if raw_rows is not None:
            for row in raw_rows:
                case = row.get("case_id")
                if case not in contract or contract[case]["domain"] != "lexical" or case in raw:
                    raise ValueError("invalid raw-history reference case")
                nll = _number(row.get("nll"), "raw-history NLL")
                exact = _binary(row.get("exact_match"), "raw-history exact match")
                if nll < 0 or exact not in (0., 1.):
                    raise ValueError("invalid raw-history outcome")
                raw[case] = dict(nll=nll, exact_match=exact)
            if set(raw) != {k for k, value in contract.items() if value["domain"] == "lexical"}:
                raise ValueError("raw-history reference must cover every lexical DEV case")
        arc_reference = {}
        if arc_reference_rows is not None:
            for row in arc_reference_rows:
                case = row.get("case_id")
                if case not in contract or contract[case]["domain"] != "arc" or case in arc_reference:
                    raise ValueError("invalid ARC past-only reference case")
                arc_reference[case] = _binary(row.get("exact_match"), "ARC reference exact match")
            if set(arc_reference) != {k for k, value in contract.items() if value["domain"] == "arc"}:
                raise ValueError("ARC past-only reference must cover every ARC DEV case")
        for family in families:
            item = {"family": family, "summary": [], "domains": {}}
            for domain in DOMAINS:
                domain_cases = sorted(k for k, value in contract.items() if value["domain"] == domain)
                for objective in OBJECTIVES:
                    for condition in CONDITIONS:
                        group_names = ("positive", "negative", "uniform") if domain == "lexical" else ("unknown_rule",)
                        for group in group_names:
                            cases = [k for k in domain_cases if contract[k]["group"] == group]
                            if not cases:
                                continue
                            selected_values = [table[(family, objective, seed, condition, case)]
                                      for seed in seeds for case in cases]
                            values = [value for value in selected_values if value["nll"] is not None]
                            item["summary"].append(dict(domain=domain, group=group,
                                objective=objective, condition=condition, cases=len(cases), seeds=len(seeds),
                                selected_outcomes=len(selected_values), measured_outcomes=len(values),
                                inconclusive_outcomes=len(selected_values)-len(values),
                                status="MEASURED" if len(values)==len(selected_values) else "INCONCLUSIVE_FULL_COHORT",
                                **{metric: float(np.mean([value[metric] for value in values])) if values else None
                                   for metric in ("nll", "exact_match", "format_valid")}))
                cases = [k for k in domain_cases if domain != "lexical" or contract[k]["group"] == "positive"]
                clusters = [contract[k]["cluster_id"] for k in cases]
                def values(objective, condition, metric):
                    return np.array([[table[(family, objective, seed, condition, case)][metric]
                                      for case in cases] for seed in seeds], dtype=float)
                effects = {}
                for metric, label in (("nll", "nll"), ("exact_match", "exact"), ("format_valid", "format")):
                    sign = -1 if metric == "nll" else 1
                    carry = values("query", "carry", metric)
                    comparisons = {"query_minus_full": values("full", "carry", metric),
                                   "carry_minus_reset": values("query", "reset", metric),
                                   "carry_minus_donor": values("query", "donor", metric),
                                   "carry_minus_corrupt": values("query", "corrupt", metric)}
                    for name, other in comparisons.items():
                        effects[name + "_" + label] = _available_effect(sign * (carry - other), clusters)
                if domain == "lexical":
                    nll_gate = all(_positive(effects[name + "_nll"], minimum=.01 if name != "carry_minus_donor" else 0.,
                                            consistency=name != "carry_minus_donor")
                                   for name in ("query_minus_full", "carry_minus_reset", "carry_minus_donor"))
                    utility_gate = all(_positive(effects[name + "_exact"])
                                       for name in ("query_minus_full", "carry_minus_reset", "carry_minus_donor"))
                    gates = {"first_byte_nll_learning_and_state": nll_gate,
                             "exact4_learning_and_state": utility_gate}
                    raw_effects = {}
                    if raw:
                        for metric, label, sign in (("nll", "nll", -1), ("exact_match", "exact", 1)):
                            raw_values = np.array([raw[case][metric] for case in cases])[None, :]
                            raw_effects["query_minus_raw_" + label] = _effect(
                                sign * (values("query", "carry", metric) - raw_values), clusters)
                    item["raw_history_reference"] = {"effects": raw_effects,
                        "quality_hurdle": "INCONCLUSIVE" if not raw else
                            "PASS" if all(_positive(effect) for effect in raw_effects.values()) else "FAIL",
                        "scope": "untrained reference, not parameter/FLOP-matched architecture evidence"}
                else:
                    learning = all(
                        _positive(effects[name + "_exact"], consistency=True)
                        for name in ("query_minus_full", "carry_minus_reset"))
                    if bindings:
                        alternate = np.array([[bindings[(family, "query", seed, case)]["exact_match"]
                                               for case in cases] for seed in seeds], dtype=float)
                        binding_effect = _available_effect(values("query", "carry", "exact_match")-alternate, clusters)
                    else:
                        binding_effect = {"status": "INCONCLUSIVE", "reason": "structural binding-control rows absent"}
                    effects["carry_minus_binding_shuffle_exact"] = binding_effect
                    attribution = None if binding_effect["status"] == "INCONCLUSIVE" else _positive(binding_effect, consistency=True)
                    gates = {"unknown_rule_exact_learning": learning,
                             "binding_specific_attribution": attribution,
                             "unknown_rule_exact_learning_and_state": False if not learning else attribution}
                    reference_effect = None
                    if arc_reference:
                        reference_values = np.array([arc_reference[case] for case in cases])[None, :]
                        reference_effect = _effect(values("query", "carry", "exact_match")-reference_values, clusters)
                    item["arc_past_only_reference"] = {
                        "method": "copy last observed demonstration output; no query target",
                        "effect": reference_effect,
                        "quality_hurdle": "INCONCLUSIVE" if reference_effect is None else
                            "PASS" if _positive(reference_effect) else "FAIL",
                        "scope": "fixed external exact-grid reference; no invented NLL"}
                item["domains"][domain] = {"group": "positive" if domain == "lexical" else "unknown_rule",
                    "effects": effects,
                    "gates": {name: ("PASS" if passed else "FAIL") if clean and passed is not None else "INCONCLUSIVE"
                              for name, passed in gates.items()}}
            passed = all(status == "PASS" for domain in item["domains"].values()
                         for status in domain["gates"].values())
            all_statuses = [status for domain in item["domains"].values() for status in domain["gates"].values()]
            item["joint_learning_gate"] = ("PASS" if clean and passed else "FAIL" if clean and "FAIL" in all_statuses else "INCONCLUSIVE")
            reference_gates = [item["raw_history_reference"]["quality_hurdle"],
                               item["arc_past_only_reference"]["quality_hurdle"],
                               item["joint_learning_gate"]]
            item["practical_reference_quality_gate"] = ("PASS" if clean and all(value == "PASS" for value in reference_gates)
                else "FAIL" if clean and "FAIL" in reference_gates else "INCONCLUSIVE")
            report["families"].append(item)
        report["status"] = "COMPLETED_INTERNAL_MEASUREMENT" if clean else "INCONCLUSIVE"
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
        report["status"] = "FAIL"
        report["analysis_error"] = str(error)
    return report


def markdown_summary(report):
    lines = ["# LANGCONTROL098 objective-learning results", "",
             "Outcome: **" + report["status"] + "**.", "",
             "Within-family objective comparisons; no architecture ranking or AGI certificate.", "",
             "Full/query mean lambda=1/0 with fixed future-target coefficients, equal domain weights and per-case segment means.",
             "These labels do not denote the original token-averaged LANGACQUIRE097 losses.", "",
             "| Family | Lexical NLL gate | Lexical exact-four gate | ARC transfer gate | Joint gate | Reference quality hurdle |",
             "| --- | --- | --- | --- | --- | --- |"]
    for item in report["families"]:
        lex, arc = item["domains"]["lexical"]["gates"], item["domains"]["arc"]["gates"]
        lines.append("| " + " | ".join([item["family"], lex["first_byte_nll_learning_and_state"],
            lex["exact4_learning_and_state"], arc["unknown_rule_exact_learning_and_state"],
            item["joint_learning_gate"], item["practical_reference_quality_gate"]]) + " |")
    lines += ["", "Effects average five model seeds within each case, then resample original source/task clusters.",
              "ARC is a secondary authored-benchmark mechanism diagnostic. Its first-byte NLL is a formatting diagnostic.",
              "All failed/insufficient cases remain in the complete JSON; few DEV clusters limit generalization.", ""]
    if "analysis_error" in report:
        lines += ["Analysis failed: " + report["analysis_error"], ""]
    return "\n".join(lines)


def write_reports(report, json_path, markdown_path):
    paths = [Path(json_path), Path(markdown_path)]
    if paths[0].resolve() == paths[1].resolve() or any(path.exists() for path in paths):
        raise FileExistsError("preserve every prior report")
    texts = [json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", markdown_summary(report)]
    for path, text in zip(paths, texts):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
