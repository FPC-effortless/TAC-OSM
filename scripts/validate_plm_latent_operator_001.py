#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tac_osm.latent_operator_001_benchmark import (
    BASE_E2E008_GENERATOR_SHA256,
    GENERATOR_VERSION,
    generator_hash,
)


def main(path: str) -> None:
    artifact_path = Path(path)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    contract_path = ROOT / "contracts" / "TACOSM-PLM-LATENT-OPERATOR-001.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    assert artifact["experiment_id"] == "TACOSM-PLM-LATENT-OPERATOR-001"
    assert artifact["status"] == "measured"
    assert artifact["provenance"]["contract_sha256"] == hashlib.sha256(
        contract_path.read_bytes()
    ).hexdigest()
    assert artifact["provenance"]["benchmark_sha256"] == generator_hash()
    assert artifact["provenance"]["base_e2e008_generator_sha256"] == BASE_E2E008_GENERATOR_SHA256
    assert artifact["protocol"]["benchmark_generator_version"] == GENERATOR_VERSION
    assert artifact["protocol"]["operator_id_in_model_input"] is False
    assert artifact["protocol"]["operator_induction_loss"] == "q2 task loss only"

    rows = artifact["seed_results"]
    assert len(rows) == 5
    assert len({r["evaluation_episode_fingerprint"] for r in rows}) == 5
    assert all(r["training_evaluation_semantic_overlap"] == 0 for r in rows)

    summary = artifact["summary"]
    assert summary["classifier_decision_integrity_pass"] is True
    assert artifact["leakage_audit"]["operator_id_in_model_input"] is False
    assert artifact["leakage_audit"]["operator_id_in_training_loss"] is False
    assert artifact["leakage_audit"]["support_context_contains_only_axy"] is True
    assert artifact["leakage_audit"]["support_order_randomized"] is True
    assert artifact["leakage_audit"]["operator_inducer_gradient_gate_pass"] is True
    assert artifact["leakage_audit"]["operator_synthesis_not_claimed"] is True

    q2 = [r["q2_accuracy"] for r in rows]
    ops = [r["operator_selection_accuracy"] for r in rows]
    expected = (
        statistics.fmean(q2) >= 0.80
        and min(q2) >= 0.40
        and statistics.fmean(ops) >= 0.90
        and min(ops) >= 0.80
    )
    assert summary["operator_induction_supported"] == expected

    print(json.dumps({
        "q2_mean": summary["q2_mean"],
        "q2_min": summary["q2_min"],
        "q2_ci95": summary["q2_composition_bootstrap_ci95"],
        "operator_selection_mean": summary["operator_selection_mean"],
        "operator_selection_min": summary["operator_selection_min"],
        "operator_selection_ci95": summary["operator_selection_composition_bootstrap_ci95"],
        "oracle_operator_q2_mean": summary["oracle_operator_q2_mean"],
        "shuffle_support_q2_mean": summary["shuffle_support_q2_mean"],
        "no_memory_q2_mean": summary["no_memory_q2_mean"],
        "operator_induction_supported": summary["operator_induction_supported"],
    }, indent=2))


if __name__ == "__main__":
    import statistics
    main(sys.argv[1])
