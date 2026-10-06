#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "TACOSM-PLM-INTEGRATED-E2E-005"

import sys
sys.path.insert(0, str(ROOT / "src"))

from tac_osm.contract import load_contract
from tac_osm.integrated_e2e_005 import FunctionalMultimodalPLM
from tac_osm.integrated_e2e_005_benchmark import GENERATOR_VERSION, HELDOUT, SEEDS, STEPS, generator_hash

parser = argparse.ArgumentParser()
parser.add_argument("result")
args = parser.parse_args()

result_path = Path(args.result)
x = json.loads(result_path.read_text(encoding="utf-8"))
contract_path = ROOT / "contracts" / f"{EXPERIMENT_ID}.json"
contract_raw = json.loads(contract_path.read_text(encoding="utf-8"))
c = load_contract(EXPERIMENT_ID)

assert x["experiment_id"] == EXPERIMENT_ID
assert x["status"] == "measured"
assert c.status == "pre-registered"
assert x["protocol"]["seeds"] == list(SEEDS) == contract_raw["seeds"]
assert x["protocol"]["steps"] == STEPS == contract_raw["protocol"]["train_steps"]
assert x["protocol"]["batch_size"] == contract_raw["protocol"]["batch_size"]
assert x["protocol"]["evaluation_episodes_per_seed"] == contract_raw["protocol"]["evaluation_episodes_per_seed"]
assert x["protocol"]["registered_heldout"] == [list(x) for x in HELDOUT]
assert [a["name"] for a in contract_raw["arms"]] == x["protocol"]["registered_controls"]
assert x["protocol"]["benchmark_generator_version"] == GENERATOR_VERSION
assert x["protocol"]["representation_auxiliary_supervision"] is False
assert x["protocol"]["operator_dispatch"] == "fixed_query_conditioned"

expected_contract_hash = hashlib.sha256(contract_path.read_bytes()).hexdigest()
assert x["provenance"]["contract_sha256"] == expected_contract_hash
assert x["provenance"]["benchmark_sha256"] == generator_hash()

rows = x["seed_results"]
assert len(rows) == len(SEEDS)
assert {r["seed"] for r in rows} == set(SEEDS)

for row in rows:
    for key in (
        "normal_q1_accuracy",
        "normal_q2_accuracy",
        "normal_verifier_accuracy",
        "no_memory_q2_accuracy",
        "shuffle_image_q2_accuracy",
        "text_only_q2_accuracy",
        "image_only_q2_accuracy",
        "audio_only_q2_accuracy",
        "oracle_q2_accuracy",
    ):
        assert 0.0 <= row[key] <= 1.0
    assert row["training_evaluation_semantic_overlap"] == 0
    assert row["oracle_q2_accuracy"] == 1.0

assert x["summary"]["oracle_q2_accuracy"] == 1.0
assert x["summary"]["gradient_surface_pass"] is True
assert x["leakage_audit"]["q1_q2_entities_distinct"] is True
assert x["leakage_audit"]["heldout_compositions_excluded_from_training"] is True
assert x["leakage_audit"]["evaluation_generated_after_training"] is True
assert x["leakage_audit"]["training_evaluation_semantic_overlap_zero"] is True
assert x["leakage_audit"]["controls_reuse_exact_same_episode_objects"] is True
assert x["leakage_audit"]["payload_auxiliary_supervision"] is False
assert x["leakage_audit"]["post_action_outcome_only_feedback"] is True
assert x["leakage_audit"]["fixed_operator_dispatch_not_learned_discovery"] is True
assert x["leakage_audit"]["pre_action_query_signature"] == [
    "self", "memory", "entity", "i", "j", "op"
]

forward_source = inspect.getsource(FunctionalMultimodalPLM.forward_episode)
assert "bit_loss" not in forward_source
assert "entity_loss" not in forward_source
assert "payload" not in forward_source

assert list(inspect.signature(FunctionalMultimodalPLM.query).parameters) == [
    "self", "memory", "entity", "i", "j", "op"
]

normal = [r["normal_q2_accuracy"] for r in rows]
mean_q2 = sum(normal) / len(normal)
assert abs(x["summary"]["primary_q2_mean"] - mean_q2) < 1e-12
assert abs(x["summary"]["all_seed_min_q2"] - min(normal)) < 1e-12
assert x["summary"]["primary_pass"] == bool(
    mean_q2 >= c.primary_endpoint.threshold and min(normal) >= 0.40
)

print("PASS: E2E-005 independent contract, fingerprint, leakage, and result validation")
