#!/usr/bin/env python3
"""Exploratory noisy-alias retrieval diagnostics, NOT confirmatory INTEGRATION-002."""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from tac_osm.integration_002_benchmark import diagnostic_rows  # noqa: E402

ID = "INTEGRATION-002-DEV-NOISY-ALIAS-001"
OUT = ROOT/"artifacts"/(ID+".json")


def main() -> None:
    rows=diagnostic_rows()
    if not rows:
        raise RuntimeError("no development episodes")
    if not all(r["exact_model_visible_identity_reuse"] is False for r in rows):
        raise AssertionError("exact-key shortcut leaked")
    def mean(name):
        return statistics.fmean(r[name] for r in rows)
    by_strength={}
    for strength in (0.0,0.25,0.5,1.0):
        subset=[r for r in rows if r["hard_fraction"]==strength]
        by_strength[str(strength)]={
            "cosine_mean":statistics.fmean(r["cosine_accuracy"] for r in subset),
            "euclidean_mean":statistics.fmean(r["euclidean_accuracy"] for r in subset),
            "euclidean_recall_at_1_mean":statistics.fmean(
                r["euclidean_recall_at_1"] for r in subset),
        }
    summary={
        "cell_rows":len(rows),
        "development_pair_count":sum(r["pairs"] for r in rows),
        "evaluated_episode_count":2*sum(r["pairs"] for r in rows),
        "oracle_accuracy":1.0,
        "mean_cosine_accuracy":mean("cosine_accuracy"),
        "mean_euclidean_accuracy":mean("euclidean_accuracy"),
        "mean_euclidean_recall_at_1":mean("euclidean_recall_at_1"),
        "worst_euclidean_accuracy":min(r["euclidean_accuracy"] for r in rows),
        "best_euclidean_accuracy":max(r["euclidean_accuracy"] for r in rows),
        "per_interference_level":by_strength,
    }
    if summary["mean_euclidean_accuracy"] >= 1.0:
        raise AssertionError("pilot remains an exact-match shortcut")
    path=ROOT/"src"/"tac_osm"/"integration_002_benchmark.py"
    record={
        "experiment_id":ID,
        "status":"development-only",
        "confirmatory_status":"NOT_REGISTERED_OR_RUN",
        "trained_model":False,
        "source_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "limitations":[
            "Oracle-side target_slot is withheld from model inputs but the generator defines ground truth",
            "No recurrent shared-state or learned addressing model is trained",
            "Simple classical cosine and Euclidean scorers are dense O(H+D) baselines",
            "No Bayes optimality or irreducible error ceiling is proven",
            "Noise levels and difficulty are exploratory, not frozen confirmatory settings",
            "No claim about real-world semantics, long-term memory or sublinear total cost",
        ],
        "summary":summary,
        "raw_cells":rows,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
    print(json.dumps(summary,indent=2,sort_keys=True))
    print("INTEGRATION-002 PROSPECTIVE DESIGN ONLY. NO CONFIRMATORY RESULT.")
    print("Artifact:",OUT)


if __name__=="__main__":
    main()
