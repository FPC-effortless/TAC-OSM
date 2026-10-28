"""GRAPH012: fail-closed five-seed, source-clustered summary of paired score traces.

This module analyzes PROVIDED predictions; it never loads checkpoints, scores a
protected source, trains a model, or promotes architecture superiority. It
preserves the primary scorer's UNQUALIFIED status under every circumstance.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from equal_cache_score_v1 import score_document

SCHEMA = "graph012-five-seed-rollup-v1"
MIN_SEEDS = 5
MIN_SOURCES_FOR_INTERVAL = 8


def _avg(values):
    if not values:
        raise ValueError("missing paired source observations")
    return math.fsum(values) / len(values)


def _quantile(values, p):
    values = sorted(values)
    k = (len(values) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] * (hi - k) + values[hi] * (k - lo)


def aggregate(docs, *, resamples=2000, bootstrap_seed=12012):
    """Five seeds must share identical (owner, source digest, ordinal, label) rows.

    Cluster bootstrap samples whole real source documents and whole seeds,
    preserving within-document token correlations. It does NOT provide a
    valid independent confidence interval if source identities themselves
    are not independently sampled, or if the documents have been inspected.
    """
    if not isinstance(docs, list) or len(docs) != MIN_SEEDS:
        raise ValueError("require exactly five complete seed traces")
    if type(resamples) is not int or not (100 <= resamples <= 100000):
        raise ValueError("resamples must be an integer in [100,100000]")
    if type(bootstrap_seed) is not int:
        raise ValueError("fixed integer bootstrap seed required")
    scored = []
    seen_seeds = set()
    expected_rows = None
    expected_data = None
    expected_prior = None
    source_seed_effects = []
    for doc in docs:
        seed = doc.get("seed") if isinstance(doc, dict) else None
        if type(seed) is not int or seed in seen_seeds:
            raise ValueError("five distinct integer seed IDs required")
        seen_seeds.add(seed)
        score = score_document(doc)
        # This digest binds target alignment, not authenticity of the source.
        identity = {(r["owner_id"],r["source_sha256"],r["ordinal"],r["target"])
                    for r in doc["rows"]}
        if expected_rows is None:
            expected_rows = identity
            expected_data = doc["data"]
            expected_prior = doc["fit_prior"]
        elif identity != expected_rows:
            raise ValueError("different source/target/owner/ordinal pairs between seeds")
        if doc["data"] != expected_data or doc["fit_prior"] != expected_prior:
            raise ValueError("data split/manifest or FIT prior differ between seeds")
        if any(md["fit_sample_sha256"] != doc["models"]["tac"]["fit_sample_sha256"]
               for md in doc["models"].values()):
            raise ValueError("per-seed fit sample schedules must match across architectures")
        effects = {}
        for source, value in score["source_nll"].items():
            effects[source] = value["tac"]["same_cache"] - value["ssm"]["same_cache"]
        source_seed_effects.append(effects)
        scored.append((seed,score,doc))
    scored.sort(key=lambda z:z[0])
    # The source-level score table collapses owner windows sharing source hash.
    # All seeds must be evaluated on the same physical source set.
    source_names = sorted(source_seed_effects[0])
    if any(sorted(d) != source_names for d in source_seed_effects):
        raise ValueError("source membership differs between seeds")
    # Reorder bootstrap arrays to the deterministic sorted-seed order.
    by_seed = {doc["seed"]: eff for doc,eff in zip(docs,source_seed_effects)}
    matrix = [[by_seed[s][src] for src in source_names]
              for s,_,_ in scored]
    per_seed = []
    gaps = set()
    for seed, result, doc in scored:
        gaps.update(result["fairness_and_promotion_gaps"])
        per_seed.append({
            "seed":seed,
            "paired_trace_sha256":result["paired_trace_sha256"],
            "tac_native_nll":result["nll"]["tac"]["native"],
            "ssm_native_nll":result["nll"]["ssm"]["native"],
            "tac_equal_cache_nll":result["nll"]["tac"]["same_cache"],
            "ssm_equal_cache_nll":result["nll"]["ssm"]["same_cache"],
            "prior_equal_cache_nll":result["nll"]["prior"]["same_cache"],
            "tac_minus_ssm_equal_cache_nll":result["paired_differences_nats_per_char"]["tac_minus_ssm_equal_cache_nll"],
            "routing":result["routing"],
            "fairness_gaps":result["fairness_and_promotion_gaps"],
        })
    pooled = _avg([x["tac_minus_ssm_equal_cache_nll"] for x in per_seed])
    # Documents, not overlapping 128-character windows or tokens, are the
    # bootstrap resampling unit. Seeds are also drawn as whole experiments.
    ci = None
    if len(source_names) >= MIN_SOURCES_FOR_INTERVAL:
        rng = random.Random(bootstrap_seed)
        sample = []
        for _ in range(resamples):
            seeds = [rng.randrange(MIN_SEEDS) for _ in range(MIN_SEEDS)]
            documents = [rng.randrange(len(source_names)) for _ in source_names]
            sample.append(_avg([_avg([matrix[s][src] for src in documents]) for s in seeds]))
        ci = [_quantile(sample, .025),_quantile(sample, .975)]
    # Ordinary pooled NLL weights source documents by number of real targets.
    # Report macro separately: equal weight per original source.
    macro = _avg([_avg([row[j] for row in matrix]) for j in range(len(source_names))])
    return {
        "schema": SCHEMA,
        "evidence_status": "SCORING_ONLY_NOT_ARCHITECTURE_QUALIFIED",
        "distinct_seed_count":len(seen_seeds),
        "distinct_source_count":len(source_names),
        "per_seed":per_seed,
        "pooled_tac_minus_ssm_equal_cache_nll":pooled,
        "source_macro_tac_minus_ssm_equal_cache_nll":macro,
        "source_clustered_two_level_bootstrap_95pct_interval":ci,
        "bootstrap_seed":bootstrap_seed,
        "bootstrap_resamples":resamples if ci is not None else 0,
        "fairness_and_promotion_gaps":sorted(gaps),
        "limitations":[
            "this summary cannot independently certify original checkpoints, source legality, authentication or causal trace generation",
            "source bootstrap is descriptive, not a proof of untouched TEST or independent corpus sampling",
            "all traces must use the exact shared cache at each token; this is not verified from supplied arrays",
            "five training seeds do not make overlapping source-document windows independent",
            "no architectural promotion, even when a numerical confidence interval excludes zero",
        ],
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path,nargs="+",help="five source-bound prediction JSON files, one per seed")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--resamples",type=int,default=2000)
    args=parser.parse_args()
    docs=[json.loads(f.read_text(encoding="utf-8")) for f in args.traces]
    out=aggregate(docs,resamples=args.resamples)
    args.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("GRAPH012 scoring-only rollup; original checkpoint replay is NOT certified")


if __name__=="__main__":main()
