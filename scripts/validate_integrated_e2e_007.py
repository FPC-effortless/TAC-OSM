#!/usr/bin/env python3
from __future__ import annotations
import hashlib,inspect,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from tac_osm.integrated_e2e_005 import FunctionalMultimodalPLM,FunctionalConfig
from tac_osm.integrated_e2e_007_benchmark import HELDOUT,TRAIN_COMBOS,SEALED_E2E005,DEV_COMBOS,E2E006_HELDOUT,generator_hash
ID="TACOSM-PLM-INTEGRATED-E2E-007";CONTRACT=ROOT/"contracts"/f"{ID}.json"
x=json.loads(Path(sys.argv[1]).read_text());c=json.loads(CONTRACT.read_text())
assert x["experiment_id"]==ID and x["status"]=="measured"
assert c["status"]=="pre-registered"
assert c["protocol"]["hidden_dim"]==64 and c["protocol"]["state_write_mode"]=="residual_linear"
assert x["protocol"]["hidden_dim"]==64 and x["protocol"]["state_write_mode"]=="residual_linear"
assert x["protocol"]["seeds"]==c["seeds"] and x["protocol"]["steps"]==c["steps"]
assert x["protocol"]["registered_heldout"]==[list(v) for v in HELDOUT]
assert x["provenance"]["contract_sha256"]==hashlib.sha256(CONTRACT.read_bytes()).hexdigest()
assert x["provenance"]["benchmark_sha256"]==generator_hash()
assert x["summary"]["oracle_q2_accuracy"]==1.0
assert x["summary"]["gradient_surface_pass"] is True
assert x["summary"]["classifier_decision_integrity_pass"] is True
assert x["leakage_audit"]["training_evaluation_semantic_overlap_zero"] is True
assert x["leakage_audit"]["heldout_compositions_excluded_from_training"] is True
assert x["leakage_audit"]["prior_e2e005_e2e006_dev_sets_excluded"] is True
assert x["leakage_audit"]["payload_auxiliary_supervision"] is False
assert x["leakage_audit"]["evaluation_generated_after_training"] is True
assert x["leakage_audit"]["controls_reuse_exact_same_episode_objects"] is True
assert list(inspect.signature(FunctionalMultimodalPLM.query).parameters)==["self","memory","entity","i","j","op"]
assert "answer" not in inspect.getsource(FunctionalMultimodalPLM.query)
assert FunctionalConfig().hidden_dim==40
assert set(HELDOUT).isdisjoint(set(SEALED_E2E005))
assert set(HELDOUT).isdisjoint(set(DEV_COMBOS))
assert set(HELDOUT).isdisjoint(set(E2E006_HELDOUT))
assert set(HELDOUT).isdisjoint(set(TRAIN_COMBOS))
for row in x["seed_results"]:
 assert row["training_evaluation_semantic_overlap"]==0
 assert row["normal_q2_decision_mismatches"]==0
 assert row["oracle_q2_accuracy"]==1.0
mean=sum(r["normal_q2_accuracy"] for r in x["seed_results"])/len(x["seed_results"])
assert abs(mean-x["summary"]["primary_q2_mean"])<1e-12
assert x["summary"]["primary_pass"]==bool(mean>=.8 and min(r["normal_q2_accuracy"] for r in x["seed_results"])>=.4)
print("PASS: E2E-007 provenance, contract, split, leakage, interface and result validation")
