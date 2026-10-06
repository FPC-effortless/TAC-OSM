#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tac_osm.contract import load_contract

EXPERIMENT_ID = "TACOSM-PLM-INTEGRATED-E2E-003"

p = argparse.ArgumentParser()
p.add_argument("result")
args = p.parse_args()

x = json.loads(Path(args.result).read_text(encoding="utf-8"))
c = load_contract(EXPERIMENT_ID)

assert x["experiment_id"] == EXPERIMENT_ID
assert x["status"] == "measured"
assert x["protocol"]["seeds"] == list(c.seeds)
assert x["protocol"]["steps"] == c.steps
assert x["protocol"]["eval_episodes"] == c.eval_steps
assert x["summary"]["primary_pass"] == (
    x["summary"]["explicit_both_q2_mean"] >= 0.80
    and x["summary"]["explicit_both_min_seed"] >= 0.40
)
assert len(x["summary"]["explicit_both_q2_seed_bootstrap_ci95"]) == 2
assert len(x["protocol"]["evaluation_episode_fingerprints"]) == len(c.seeds)
assert len(set(x["protocol"]["evaluation_episode_fingerprints"])) == len(c.seeds)

arm_results = x["arm_seed_results"]
required = {"explicit_write","explicit_read","explicit_both"}
assert set(arm_results) == required

fingerprints = {}
for arm in required:
    rows = arm_results[arm]
    assert len(rows) == len(c.seeds)
    assert {r["seed"] for r in rows} == set(c.seeds)
    fingerprints[arm] = tuple(r["evaluation_episode_fingerprint"] for r in rows)
    assert all(len(fp) == 64 for fp in fingerprints[arm])

assert len(set(fingerprints.values())) == 1

for arm in required:
    data = arm_results[arm]
    for row in data:
        assert 0.0 <= float(row[arm]) <= 1.0

assert x["provenance"]["arm_artifacts"]
for arm, prov in x["provenance"]["arm_artifacts"].items():
    assert prov.get("git_commit") not in {None, "", "unknown"}

assert x["decision_rule"]["integrity"] == "passed before arm execution"
print(json.dumps({
    "experiment_id": EXPERIMENT_ID,
    "status": x["status"],
    "primary_mean_q2": x["summary"]["explicit_both_q2_mean"],
    "primary_ci95": x["summary"]["explicit_both_q2_seed_bootstrap_ci95"],
    "min_seed_q2": x["summary"]["explicit_both_min_seed"],
    "memory_drop": x["summary"]["memory_drop"],
    "alignment_drop": x["summary"]["alignment_drop"],
    "paired_episode_fingerprints": "PASS",
    "leakage_provenance": "PASS",
}, indent=2, sort_keys=True))
