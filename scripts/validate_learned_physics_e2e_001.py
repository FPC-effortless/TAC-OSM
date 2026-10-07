#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

EXPERIMENT_ID = "TACOSM-PLM-LEARNED-PHYSICS-E2E-001"

artifact = Path(sys.argv[1])
data = json.loads(artifact.read_text(encoding="utf-8"))
root = artifact.parents[1]
contract = root / "contracts" / f"{EXPERIMENT_ID}.json"
benchmark = root / "src" / "tac_osm" / "learned_physics_e2e_benchmark.py"

assert data["status"] == "measured"
assert data["experiment_id"] == EXPERIMENT_ID
assert data["provenance"]["contract_sha256"] == hashlib.sha256(contract.read_bytes()).hexdigest()
assert data["provenance"]["benchmark_sha256"] == hashlib.sha256(benchmark.read_bytes()).hexdigest()

rows = data["seed_results"]
assert set(rows) == {"learned_data_only", "learned_physics_prior"}
assert len(rows["learned_data_only"]) == len(rows["learned_physics_prior"]) == 5
assert [row["seed"] for row in rows["learned_data_only"]] == [0, 1, 2, 3, 4]
assert [row["seed"] for row in rows["learned_physics_prior"]] == [0, 1, 2, 3, 4]
assert all(
    row["training_evaluation_overlap"] == 0
    for arm in rows.values()
    for row in arm
)
assert [
    row["evaluation_episode_fingerprint"]
    for row in rows["learned_data_only"]
] == [
    row["evaluation_episode_fingerprint"]
    for row in rows["learned_physics_prior"]
]
assert len({
    row["evaluation_episode_fingerprint"]
    for row in rows["learned_physics_prior"]
}) == 5
assert len({
    row["parameter_count"]
    for arm in rows.values()
    for row in arm
}) == 1
assert data["integrity"]["no_physical_state_supervision"] is True
assert data["integrity"]["no_entity_ids"] is True
assert data["integrity"]["no_fixed_operator_table"] is True
assert data["summary"]["physics_prior_delta_ci95"] is not None

print(json.dumps(data["summary"], indent=2, sort_keys=True))
