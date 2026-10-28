# GRAPH006 original replay intake — engineering increment

Internally measured engineering checks, **not independently qualified**.
No original language-data/model evaluation was performed.

The offline intake runner returns exit **2**, `state=BLOCKED` and
`REPO_REPLAY=BLOCKED`. Its full deterministic output is READINESS.json.
The reviewed inventory, publisher/license/original document hashes, 36 original
FIT/previously exposed DEV documents, 25 original checkpoint payloads, original
protocol, train/evaluation/diagnostic recipes and original per-seed metrics are
not supplied to the runner. No unknown hashes or signatures were filled in.
The original source code hashes in CONTRACT.json are calculated from the pinned
PR529 Git commit; they do not authenticate any unavailable dataset/checkpoint.

Implemented: byte-only intake with pinned inventory digest and committed Git
membership, exact family/seed and FIT/DEV coverage, source-version hashes,
recorded publisher/license, original recipe/metric assets, missing-vs-changed
receipts and safe relative paths. It rejects TEST partitions before payload
reads, changed hashes, omissions, symlink/hardlink aliases and escaped paths.
It does not deserialize PyTorch, import supplied recipes, fit, score, extract
archives or access the network. Original binary contents (including optimizer
and RNG structure) are **NOT INSPECTED**; those require the later lawful,
source-pinned cold restore. A valid byte inventory only means READY_FOR_REPLAY,
never replay PASS or independently verified authenticity.

Validation: 19 constructed intake tests plus existing probe/execution-level
checks: **50 PASS**. First full suite: **1 FAIL, 1246 PASS, 4 SKIPPED** (stale
mutable test count); updated the current count to1251, preserving freeze327.
Final full suite: **1247 PASS, 4 SKIPPED**. Fixture bytes
and example.invalid provenance are engineering controls, never real evidence.
The router selects bounded GitHub Actions CPU, zero GPU minutes; its checked-in
COMPUTE_DECISION.json is reproduced in the dedicated CI workflow. The workflow
expects BLOCKED and labels it explicitly in its summary and artifact. Green
engineering CI must not be interpreted as successful original score replay.

Original evidence: [PR529 failure diagnosis](https://github.com/FPC-effortless/TAC-OSM/pull/529)
reports useful signal loss and failed lag64/192 retrieval. These are preserved
reported results, not new independent measurements. Earlier PR483/history_audit014
H64 negative results/exact replay and PR505 negatives are unchanged. This change
adds no learned model, architecture, mathematical graph or evaluation adapter;
PNDS54/full Torch/IEEE/AD refinement remains OPEN. The larger archived language
model is read-only; no new fit is authorized outside the user's 95k–105k scale.

Next: recover the **original** PR529 archive from its existing lawful owner,
review/commit the inventory with exact original hashes and allowed FIT/DEV
source/license evidence, and preserve original payloads inside this repository
(or restored Git LFS). Then preregister actual score replay, record its router
plan, and restore/score in clean-checkout CI. Do not retrain substitutes, open
TEST, or treat external signers as prerequisites for ordinary internal work.

## Interface for the eventual original inventory

Use schema `tac-osm.replay-inventory.v1`, study_id `TAC-LANG-GRAPH-006`:
`publisher` has url/license/license_evidence; `sources` records path/sha256 and
partition fit/dev; `checkpoints` records path/sha256/family/seed; `assets` records
path/sha256/role. Exact groups and counts are in CONTRACT.json. Each record's
path must refer to a distinct, committed original file beneath --root. The
inventory itself must be committed and its reviewed SHA supplied explicitly.
No external signer is required for this internal metadata inspection.

```
python scripts/check_original_replay_intake.py \
  --contract research/replay-readiness/language_graph006/CONTRACT.json \
  --root PATH_TO_CHECKOUT_WITH_ORIGINALS \
  --inventory REPO_RELATIVE_REVIEWED_INVENTORY \
  --inventory-sha256 REVIEWED_INVENTORY_SHA256 \
  --output intake-results/READINESS.json
```

No inventory is committed now because original identities are unavailable.
Exit0=byte intake ready (replay still BLOCKED); exit2=missing prerequisites;
exit1=invalid metadata/changed bytes/runtime failure. The gate alone grants no
training, protected evaluation, certification, promotion or resource exception.
