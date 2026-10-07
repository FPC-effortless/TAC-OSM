#!/usr/bin/env python3
from __future__ import annotations
import hashlib, inspect, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"src"))
from tac_osm.integrated_e2e_005 import FunctionalMultimodalPLM, FunctionalConfig
from tac_osm.integrated_e2e_006_benchmark import (
    GENERATOR_VERSION, HELDOUT, TRAIN_COMBOS, SEALED_E2E005, DEV_COMBOS,
    generator_hash,
)
EXPERIMENT_ID="TACOSM-PLM-INTEGRATED-E2E-006"
CONTRACT=ROOT/"contracts"/f"{EXPERIMENT_ID}.json"
x=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
c=json.loads(CONTRACT.read_text(encoding="utf-8"))
assert x["experiment_id"]==EXPERIMENT_ID and x["status"]=="measured"
assert c["status"]=="pre-registered"
assert x["protocol"]["hidden_dim"]==64==c["protocol"]["hidden_dim"]
assert x["protocol"]["seeds"]==c["seeds"]
assert x["protocol"]["registered_heldout"]==[list(v) for v in HELDOUT]
assert x["protocol"]["train_combos_count"]==len(TRAIN_COMBOS)
assert x["protocol"]["benchmark_generator_version"]==GENERATOR_VERSION
assert x["provenance"]["contract_sha256"]==hashlib.sha256(CONTRACT.read_bytes()).hexdigest()
assert x["provenance"]["benchmark_sha256"]==generator_hash()
assert x["summary"]["oracle_q2_accuracy"]==1.0
assert x["summary"]["gradient_surface_pass"] is True
assert x["summary"]["classifier_decision_integrity_pass"] is True
assert x["leakage_audit"]["training_evaluation_semantic_overlap_zero"] is True
assert x["leakage_audit"]["heldout_compositions_excluded_from_training"] is True
assert x["leakage_audit"]["evaluation_generated_after_training"] is True
assert x["leakage_audit"]["payload_auxiliary_supervision"] is False
assert x["leakage_audit"]["classifier_action_decision_consistency"] is True
qsig=list(inspect.signature(FunctionalMultimodalPLM.query).parameters)
assert qsig==["self","memory","entity","i","j","op"]
src=inspect.getsource(FunctionalMultimodalPLM.query)
assert "answer" not in src
assert "def forward_episode" not in inspect.getsource(FunctionalMultimodalPLM)
assert set(HELDOUT).isdisjoint(SEALED_E2E005)
assert set(HELDOUT).isdisjoint(DEV_COMBOS)
assert set(HELDOUT).isdisjoint(set(TRAIN_COMBOS))
for row in x["seed_results"]:
    assert row["training_evaluation_semantic_overlap"]==0
    assert row["normal_q2_decision_mismatches"]==0
    assert row["oracle_q2_accuracy"]==1.0
mean=sum(r["normal_q2_accuracy"] for r in x["seed_results"])/len(x["seed_results"])
assert abs(mean-x["summary"]["primary_q2_mean"])<1e-12
assert x["summary"]["primary_pass"] == bool(mean>=0.80 and min(r["normal_q2_accuracy"] for r in x["seed_results"])>=0.40)
print("PASS: E2E-006 independent provenance, split, leakage, interface and result validation")
