#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tac_osm.learned_address_001_benchmark import (
    BASE_E2E008_GENERATOR_SHA256,
    GENERATOR_VERSION,
    generator_hash,
)


def main(path: str):
    artifact_path = Path(path)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    contract_path = ROOT / "contracts" / "TACOSM-PLM-LEARNED-ADDRESS-001.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    assert artifact["experiment_id"] == "TACOSM-PLM-LEARNED-ADDRESS-001"
    assert artifact["status"] == "measured"
    assert artifact["provenance"]["contract_sha256"] == hashlib.sha256(
        contract_path.read_bytes()
    ).hexdigest()
    assert artifact["provenance"]["benchmark_sha256"] == generator_hash()
    assert artifact["provenance"]["base_e2e008_generator_sha256"] == BASE_E2E008_GENERATOR_SHA256
    assert artifact["protocol"]["benchmark_generator_version"] == GENERATOR_VERSION
    assert artifact["protocol"]["entity_id_in_model_input"] is False
    assert artifact["protocol"]["operator_dispatch"] == "fixed query-conditioned oracle"

    rows = artifact["seed_results"]
    assert len(rows) == 5
    assert len({r["evaluation_episode_fingerprint"] for r in rows}) == 5
    assert all(r["training_evaluation_semantic_overlap"] == 0 for r in rows)
    assert all(r["q2_target_slot_counts"] == {"0": 200, "1": 200, "2": 200} for r in rows)

    summary = artifact["summary"]
    assert summary["classifier_decision_integrity_pass"] is True
    assert artifact["leakage_audit"]["entity_id_in_model_input"] is False
    assert artifact["leakage_audit"]["address_key_independent_of_payload"] is True
    assert artifact["leakage_audit"]["query_contains_no_answer"] is True
    assert artifact["leakage_audit"]["observation_order_randomized"] is True
    assert artifact["leakage_audit"]["q2_target_slot_balanced"] is True
    assert artifact["leakage_audit"]["evaluation_fingerprints_distinct"] is True
    assert artifact["leakage_audit"]["oracle_address_evaluation_only"] is True

    q2 = [r["q2_accuracy"] for r in rows]
    address = [r["address_selection_accuracy"] for r in rows]
    oracle = [r["oracle_address_q2_accuracy"] for r in rows]
    expected = (
        statistics.fmean(q2) >= 0.80
        and min(q2) >= 0.40
        and statistics.fmean(address) >= 0.90
        and min(address) >= 0.80
        and statistics.fmean(oracle) >= 0.80
        and statistics.fmean(oracle) - statistics.fmean(q2) <= 0.10
    )
    assert summary["learned_address_supported"] == expected

    print(json.dumps({
        "q2_mean": summary["q2_mean"],
        "q2_min": summary["q2_min"],
        "q2_ci95": summary["q2_composition_or_seed_bootstrap_ci95"],
        "address_selection_mean": summary["address_selection_mean"],
        "address_selection_min": summary["address_selection_min"],
        "address_selection_ci95": summary["address_selection_seed_bootstrap_ci95"],
        "oracle_address_q2_mean": summary["oracle_address_q2_mean"],
        "learned_minus_oracle_q2_gap": summary["learned_minus_oracle_q2_gap"],
        "corrupted_address_q2_mean": summary["corrupted_address_q2_mean"],
        "shuffle_observation_order_q2_mean": summary["shuffle_observation_order_q2_mean"],
        "learned_address_supported": summary["learned_address_supported"],
    }, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
