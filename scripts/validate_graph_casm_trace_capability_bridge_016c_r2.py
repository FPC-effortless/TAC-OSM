#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from tac_osm.contract import load_contract

EXPERIMENT_ID = "TACOSM-GRAPH-CASM-TRACE-CAPABILITY-BRIDGE-016C-R2"
ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    args = parser.parse_args()

    result = json.loads(Path(args.result).read_text(encoding="utf-8"))
    contract = load_contract(EXPERIMENT_ID)

    assert result["experiment_id"] == EXPERIMENT_ID
    assert result["status"] == "measured"
    assert result["protocol"]["seeds"] == list(contract.seeds)
    assert result["protocol"]["M_levels"] == list(contract.h_levels)
    assert result["protocol"]["tasks_per_seed_M"] == contract.eval_steps
    assert result["protocol"]["channels"] == [a.name for a in contract.arms]
    assert result["protocol"]["B"] == 8
    assert result["protocol"]["primary_M"] == 512

    expected_tasks = len(contract.seeds) * len(contract.h_levels) * contract.eval_steps
    raw = result["results"]
    assert set(map(int, raw)) == set(contract.seeds)

    observed = 0
    for seed in contract.seeds:
        assert set(raw[str(seed)]) == set(map(str, contract.h_levels))
        for m in contract.h_levels:
            trials = raw[str(seed)][str(m)]
            assert len(trials) == contract.eval_steps
            for trial in trials:
                observed += 1
                for channel in contract.arms:
                    row = trial[channel.name]
                    assert row["target_bucket_size"] >= 1
                    assert row["shortlist_size"] <= 8
                    assert row["candidate_scan_operations"] == m
                    assert row["signature_construction_operations"] == m * 16
                    assert row["target_evidence_read_operations"] == 1
                    assert row["probe_selection_operations"] == 1
                    assert row["verification_operations"] == row["execution_operations"]
                    assert row["total_operations"] == (
                        row["signature_construction_operations"]
                        + row["candidate_scan_operations"]
                        + row["probe_selection_operations"]
                        + row["target_evidence_read_operations"]
                        + row["verification_operations"]
                    )
                    assert row["target_in_shortlist"] in (0.0, 1.0)
    assert observed == expected_tasks

    checks = result["checks"]
    for seed in contract.seeds:
        c = checks[str(seed)]
        assert c["train_eval_structure_disjoint"] is True
        assert c["train_eval_truth_disjoint"] is True
        assert c["target_identity_available_to_selector"] is False
        assert c["target_index_available_to_selector"] is False
        assert c["target_evidence_available_before_selection"] is False
        assert c["verifier_labels_available_to_selector"] is False

    primary = result["summary"]["primary"]
    assert primary["M"] == 512
    assert len(primary["seed_level_paired_delta"]) == len(contract.seeds)

    recomputed = []
    for seed in contract.seeds:
        rows = raw[str(seed)]["512"]
        trace = statistics.fmean(r["activation_trace"]["verified_success"] for r in rows)
        scalar = statistics.fmean(r["scalar_row"]["verified_success"] for r in rows)
        recomputed.append(trace - scalar)

    assert all(abs(a - b) < 1e-12 for a, b in zip(
        recomputed, primary["seed_level_paired_delta"]
    ))
    assert abs(
        statistics.fmean(recomputed) - primary["mean_paired_delta"]
    ) < 1e-12

    print(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "validated_trials": observed,
                "primary_mean_delta": primary["mean_paired_delta"],
                "seed_bootstrap_95ci": primary["seed_bootstrap_95ci"],
                "leakage_gates": "PASS",
                "operation_accounting": "PASS",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
