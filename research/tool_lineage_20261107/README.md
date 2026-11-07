# TOOL_LINEAGE_20261107 — exact original TRAIN byte lineage

Internal source measurement; **not independently qualified**. No neural inference,
training, tokenizer fitting, tool execution, protected TEST or GPU use.

## Verified unfinished work and priorities

Reviewed master-plan commit `af307318a18d7fbd97b94ada9f383ff5ed614a4d`, current
repository/FPC/real-world/phone-lab rules, Issue #472, drafts #721/#722 and PNDS
#115–117. Draft #721 already supplies the optional FP64-memory remedy; its 0/84
new-path result does not erase the original 11/84 FP32 failures. Draft #722 already
implements the source/BPE/schema pipeline. This increment does not repeat either.

1. Complete original-byte traceability of the unfinished source pipeline.
2. Resolve semantic-template leakage, missing no-call coverage and operational
   evidence before any capability claim or new fit.
3. Qualify bounded memory/streaming inference on an actual offline device, with
   physical latency/RAM/energy evidence rather than inferred efficiency.

Source provenance and causal boundaries apply across language/System 1,
Enterprise, Temporal, Physical, Edge-Lite and Counterfactual work. The public
publisher's **synthetic** requests cannot substitute for real operational outcomes
in any track. This work adds no neural architecture and authorizes no fit.

## Concrete gap and additive remedy

The frozen parent `csl125_training_data.digest(row)` computes a sorted, compact
UTF-8 JSON object hash. It is a useful canonical identity, but is not the SHA-256
of the publisher's original line. Source encoding, whitespace and key order can
change without changing that identity. Every existing `source_sha256`, source ID,
label, grouping, partition, catalogue and quarantine decision remains intact.

`jsonl_byte_lineage.py` adds an optional sidecar with logical record, physical
line, exact half-open byte offsets, raw-line hash including line ending, canonical
object hash and adapter disposition. It fails closed on wrong file pins, invalid
JSON/UTF-8, incomplete/nonunique disposition accounting, canonical identity drift,
mutated parsed records, wrong split and invalid duplicate representatives. This
is exact byte provenance to a supplied pin, not a cryptographic proof of source
rights, semantic label truth or absence of hash collisions.

## Preregistration and authentic source replay

Protocol and no-fit CPU router receipt were published before new execution in
commit `92e3e9bc6bef18f11062957f7099533630fe39db`.

- Original acquisition: [run 42262433533](https://github.com/FPC-effortless/TAC-OSM/actions/runs/42262433533), artifact `13524439750`.
- Original artifact ZIP SHA-256: `1961ffdf01f7979d61c0ff9b04093dea3d5a2b7afb1e7c46c8ea6966cf667dc7`.
- Publisher: `mllmTeam/DroidCall`, revision `42563ae614280d2891d57f1e7057c4bc50dd27bd`; publisher card declares Apache-2.0 and synthetic data.
- Only README.md, annotated_api.jsonl and DroidCall_train.jsonl are opened; all three exact file pins are verified. No TEST file is read.
- Original TRAIN: 12,923,060 bytes / 10,071 records. Every raw-line hash differs from the existing canonical-object hash.
- All 10,071 records mapped once: 7,837 accepted (7,065 FIT / 772 DEV), 2,234 quarantined, zero duplicates. Adapter decisions replay unchanged: `fc3f290b760223e10bee1eaf8f6a4fc69cb3c77fac53f08e269cba7b0c2c385f`.
- Local tests: **17 passed**. These are constructed engineering fixtures, separate from the original TRAIN audit above.
- Local source summary: `LOCAL_RESULTS.json`. Dedicated hosted replay is recorded separately when available.

The full sidecar contains hashes/offsets/dispositions only, not request/answer
contents. Replay uses the old adapter without retraining BPE, executing a model or
modifying original source/results. Standard 95k–105k / matched <=1% parameter and
identical data/training/tuning gates are unchanged; 500k exploratory history is
separate and no 125M fit exemption is introduced.

## Reproduction

```sh
python -B scripts/phone_lab.py plan research/phone_queue/tool_lineage_20261107.json
PYTHONPATH=src:. python -m pytest -q tests/test_jsonl_byte_lineage.py
python -B scripts/audit_tool_jsonl_lineage.py --publisher-dir ORIGINAL/publisher --output NEW_OUTPUT
```

The dedicated workflow reuses the exact prior acquisition, verifies all allowed
publisher file hashes and emits a source-only evidence artifact. Outputs refuse
overwrite. Source or tokenizer preparation is not training approval.

## Scientific and mathematical limits / next step

No operational tool capability, fresh-language generalization, long-history
retrieval, safety/reward certification, checkpoint promotion or intelligence gain
was measured. Zero no-call examples and semantic-template leakage remain open.
PNDS scalar/operator results are not end-to-end parser/provenance/IEEE/AD proofs;
those obligations remain **OPEN**. There is no new neural operation requiring a
new theorem/variant here. Original H64 and all prior negative evidence stay intact.

Actual target device, operator support, numerical agreement, physical p50/p95,
RAM, energy and airplane-mode measurements: **NOT RUN** (no physical target in
this source-only audit). No deployment or remote inference fallback is added.
Next: review/adopt the sidecar in the prepared data bundle, then preregister a
label-blind semantic-template separation audit before considering a bounded fit.
