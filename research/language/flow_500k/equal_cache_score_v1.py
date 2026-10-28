"""GRAPH012: offline paired native-backbone vs identical-cache scoring.

Consumes frozen, source-bound prediction traces. This is a scoring/qualification
utility, NOT a causal-proof or a model/SSM training runner. Rows are strictly
paired; shared retrieval posteriors are computed upstream from observed source.
No training, checkpoints, target-derived routing, or external reads here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

SCHEMA = "graph012-equal-cache-v1"
VOCAB = 128
REQUIRED_MODEL_FIELDS = (
    "architecture", "active_parameters", "training_updates", "training_tokens",
    "training_flops", "fit_sample_sha256", "checkpoint_sha256",
)


def _digest(value):
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _positive_probability(x, name):
    if not isinstance(x, list) or len(x) != VOCAB:
        raise ValueError(f"{name}: expected 128 probabilities")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 or v > 1 for v in x):
        raise ValueError(f"{name}: finite strictly-positive probabilities required")
    if abs(math.fsum(x) - 1.0) > 2e-5:
        raise ValueError(f"{name}: probability law not normalized")
    return x


def _mix(base, source, weight):
    return [(1.0 - weight) * b + weight * s for b, s in zip(base, source)]


def _mean(xs):
    return math.fsum(xs) / len(xs)


def _score_rows(doc):
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise ValueError("unsupported GRAPH012 schema")
    policy = doc.get("policy", {})
    if (policy.get("exact_weight") != 0.6 or policy.get("approx_weight") != 0.1 or
        policy.get("cache_capacity") != 512 or policy.get("cache_order") != 3 or
        policy.get("tau") != 0.05 or policy.get("query_topk") != 8 or
        policy.get("query_temperature") != 0.25):
        raise ValueError("cache must match frozen GRAPH009/011 policy")
    if not _digest(policy.get("query_checkpoint_sha256")) or not _digest(policy.get("fit_prior_sha256")):
        raise ValueError("frozen shared query and FIT-prior identities required")
    m = doc.get("models", {})
    if not isinstance(m, dict) or set(m) != {"tac", "ssm"}:
        raise ValueError("paired TAC and SSM metadata required")
    for label, md in m.items():
        if not isinstance(md, dict) or any(k not in md for k in REQUIRED_MODEL_FIELDS):
            raise ValueError(f"{label}: model training/identity receipts missing")
        if (type(md["architecture"]) is not str or not md["architecture"] or
            any(type(md[x]) is not int or md[x] < 1 for x in
                ("active_parameters", "training_updates", "training_tokens", "training_flops")) or
            not _digest(md["fit_sample_sha256"]) or not _digest(md["checkpoint_sha256"])):
            raise ValueError(f"{label}: invalid training or checkpoint receipt")
    data = doc.get("data", {})
    if (not isinstance(data, dict) or data.get("split") not in ("FIT", "DEV", "TEST") or
        not _digest(data.get("source_manifest_sha256")) or
        not _digest(data.get("scoring_protocol_sha256")) or
        data.get("prior_test_exposures") not in (True, False)):
        raise ValueError("source provenance and split disclosure required")
    prior = _positive_probability(doc.get("fit_prior"), "FIT prior")
    prior_digest = hashlib.sha256(json.dumps(prior, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    if prior_digest != policy["fit_prior_sha256"]:
        raise ValueError("FIT prior bytes do not match shared prior hash")
    rows = doc.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("nonempty paired prediction rows required")
    seen = set()
    losses = defaultdict(list)
    by_source = defaultdict(lambda: defaultdict(list))
    routing = {"exact": 0, "approx": 0, "no_retrieval": 0}
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"row {i}: invalid")
        owner, source, ordinal, y = (row.get(k) for k in ("owner_id", "source_sha256", "ordinal", "target"))
        if (type(owner) is not str or not owner or not _digest(source) or
            type(ordinal) is not int or ordinal < 0 or type(y) is not int or not 0 <= y < VOCAB):
            raise ValueError(f"row {i}: invalid owner/source/ordinal/target")
        key = (owner, source, ordinal)
        if key in seen:
            raise ValueError(f"row {i}: duplicate source owner ordinal")
        seen.add(key)
        exact_hit, approx_hit = row.get("exact_hit"), row.get("approx_hit")
        if type(exact_hit) is not bool or type(approx_hit) is not bool or (exact_hit and approx_hit):
            raise ValueError(f"row {i}: invalid retrieval gate or exact priority violation")
        exact = _positive_probability(row.get("exact_posterior"), f"row {i} exact posterior") if exact_hit else None
        approx = _positive_probability(row.get("approx_posterior"), f"row {i} approx posterior") if approx_hit else None
        if not exact_hit and row.get("exact_posterior") is not None:
            raise ValueError(f"row {i}: unexpected exact cache on miss")
        if not approx_hit and row.get("approx_posterior") is not None:
            raise ValueError(f"row {i}: unexpected approximate posterior on miss")
        routing["exact" if exact_hit else "approx" if approx_hit else "no_retrieval"] += 1
        for name in ("tac", "ssm", "prior"):
            base = prior if name == "prior" else _positive_probability(row.get("native", {}).get(name), f"row {i} {name} native")
            after_exact = _mix(base, exact, 0.6) if exact_hit else base
            after_both = _mix(after_exact, approx, 0.1) if approx_hit else after_exact
            for variant, p in (("native", base), ("exact", after_exact), ("same_cache", after_both)):
                v = -math.log(p[y])
                losses[(name, variant)].append(v)
                by_source[source][(name, variant)].append(v)
    results = {name: {variant: _mean(losses[(name, variant)]) for variant in ("native", "exact", "same_cache")}
               for name in ("tac", "ssm", "prior")}
    differences = {
        "tac_minus_ssm_native_nll": results["tac"]["native"] - results["ssm"]["native"],
        "tac_minus_ssm_equal_cache_nll": results["tac"]["same_cache"] - results["ssm"]["same_cache"],
        "equal_cache_interaction_vs_native": (results["tac"]["same_cache"] - results["ssm"]["same_cache"])
                                               - (results["tac"]["native"] - results["ssm"]["native"]),
        "tac_native_value_vs_shared_prior_cache": results["prior"]["same_cache"] - results["tac"]["same_cache"],
        "ssm_native_value_vs_shared_prior_cache": results["prior"]["same_cache"] - results["ssm"]["same_cache"],
    }
    a, b = m["tac"], m["ssm"]
    gaps = []
    if abs(a["active_parameters"] - b["active_parameters"]) / max(a["active_parameters"], b["active_parameters"]) > .01:
        gaps.append("learned parameters differ by >1%")
    if abs(a["training_flops"] - b["training_flops"]) / max(a["training_flops"], b["training_flops"]) > .01:
        gaps.append("training FLOP budgets differ by >1%")
    for field in ("training_updates", "training_tokens", "fit_sample_sha256"):
        if a[field] != b[field]:
            gaps.append(f"{field} not paired")
    if data["prior_test_exposures"] or data["split"] != "TEST":
        gaps.append("not a genuinely untouched TEST split")
    # Evidence authenticity, trained checkpoint replay and target-free source scan
    # MUST be checked by an independent runner; this scorer cannot certify them.
    gaps.append("independent original checkpoint/source/causal replay not attested by offline scorer")
    trace_digest = hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    return {"schema": SCHEMA, "evidence_status": "SCORING_ONLY_NOT_ARCHITECTURE_QUALIFIED",
            "paired_trace_sha256": trace_digest,
            "rows": len(rows), "unique_sources": len(by_source), "routing": routing,
            "nll": results, "paired_differences_nats_per_char": differences,
            "fairness_and_promotion_gaps": gaps,
            "source_nll": {key:{name:{variant:_mean(values[(name,variant)]) for variant in ("native", "exact", "same_cache")}
                                 for name in ("tac","ssm","prior")} for key,values in sorted(by_source.items())},
            "limitations": ["does not run/freeze models or verify source trace causality",
                            "shared query and cache probabilities must come from independent target-blind source scan",
                            "owner_id is a label, not an authentication credential",
                            "does not establish equal tuning effort, quality-matched cost or semantic long memory"]}


def score_document(doc):
    """Pure scoring function; validates before computing the first result."""
    return _score_rows(doc)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", type=Path, help="source-owned JSON export of paired frozen backbone predictions and COMMON cache")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    doc = json.loads(args.trace.read_text(encoding="utf-8"))
    result = score_document(doc)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Scored {result['rows']} paired rows; status={result['evidence_status']} (not a qualified head-to-head)")


if __name__ == "__main__":
    main()
