# HISTORY-AUDIT014: original-source historical-memory audit

**Status: internally measured, not independently qualified. The H64 benefit gate failed for both seeds; source trace checks and byte-identical replay passed.**

[Measured results and limits](INTERNAL_RESULTS.md). Internal research runs under the operator's 2026-10-27 execution-level amendment. Ordinary public FIT diagnostics do not require
external signers. Independent qualification remains a distinct guarded level.
The prior blocked receipt is historical evidence of the old policy, not the
current internal prerequisite. No original data, model or historical result is
rewritten by this authorization change.

This follows [draft PR #477](https://github.com/FPC-effortless/TAC-OSM/pull/477).
`FAIR_COMPARISON.json` preregisters a **noncomparative frozen diagnostic**,
not a new fit or qualified architecture comparison. Its four original model
identities come from LATFUSE008's immutable original receipts; the original
implementation is pinned to `0d9c9fbc8c5ed94762c9da129379158d36125e7f`.

## Source-code audit completed

The pinned code path is:

1. `scripts/latfuse002_real_fit.py:57` authenticates the original UCI501
   archive hash and enumerates original station CSVs. The FIT branch checks
   year **before numeric observation conversion**, stopping at 2015. Archive
   hashing is not independent publisher authentication. Reading the archive
   bytes does not mean its protected numerical observations were scored.
2. `scripts/latfuse008_paired_real_training.py:63` derives its final input
   normalization strictly before 2014-03-01 and builds previous-value/age
   caches forward in time. The underlying LATFUSE002 loader computes an
   intermediate later-FIT normalization, but LATFUSE008 discards those
   normalized values and recomputes from original raw FIT. The audit must
   verify the *returned actual features*, rather than infer leakage or safety
   solely from either normalization's name.
3. `src/tac_osm/latent_fusion_004_factorial.py:107` computes the actual encoder
   value `u`, projects the source key, value, retention and write/router gates,
   and writes to the two-head matrix only where observations are available.
   An hourly row is therefore not by itself a memory-write identity.
4. `src/tac_osm/latent_fusion_007_historic_query.py:41` forms native historical
   keys from requested past calendar and current fast/slow state, reads the
   matrix and applies `historic_event_head`. Original historical source truth
   is not a native query argument. An oracle historical write-key is privileged
   offline diagnostic information and cannot establish legal native recall.
5. `scripts/latfuse013_real_pm25_content_probe.py:77` constructs its wrong key
   by swapping deliberately high/ordinary pair members. This is a directly
   label-correlated comparator. Its original results remain unchanged; its
   favorable wrong-key score is not evidence arbitrary wrong addresses work
   better than true addresses.

The original cohort is label-selected, previously exposed May–June 2014 FIT,
and the parent models trained on that FIT era. Removing labels from **key
selection** does not make the cohort label-independent or an unseen holdout.
Every case resets state 23 hours before its anchor, reproducing the original
case protocol; it does not audit uninterrupted multi-year streaming state.

## Implemented actual-source trace path

`scripts/original_history_trace.py` verifies the pinned checkout's Python files
before importing the original graph. It rejects extra (including ignored),
modified, missing and symlinked Python files. After the internal public-source checks (or external permission in qualification
mode), it authenticates source/checkpoint bytes and loads original weights.

The audit rebuilds the source's permitted normalization/last/age prefix and
compares it with the actual features. Both predictors receive a truncated
feature-only mapping: **no raw targets and no feature rows after query time**.
Each source case has one actual owner and one independent recurrent state.

The associative path hooks actual `key_map`, `val_map`, retention, write and
router module invocations inside `step`. It records input/encoder/key/matrix
digests and original owner/hour, reconstructs the observed delta update, and
requires agreement within 2e-6 absolute FP32 error. A control candidate must
have an actual nonzero directional matrix write. PR #477's label-blind
selection then resolves its selected hours into those **captured** keys.
Missing or duplicate targets/candidates block; no paired-label fallback exists.

The native read is independently contracted and passed through the original
event head; its logit must match the original historical-query method within
2e-6. Four wrong-time keys and the true historical write-key are explicitly
privileged controls. They do not enter the native predictor. Prediction of
historical events and original +6-hour forecast finishes for both original
architectures before the scorer consults factual source outcomes.

No new learned decoder is fitted. The primary diagnostic is H64 event Brier,
with original nonassociative checkpoint, same-checkpoint zero-read lesion,
balanced-cohort prior and causal current-last threshold baselines. All use the
same original cases; report both seeds separately. The full training-FLOP
comparison was never established historically, so this is not architecture
superiority. Zero-read is an OOD lesion and the deterministic threshold is a
practical control, not a capacity-matched trained architecture. Historical
binary-event prediction does not establish continuous PM2.5 reconstruction.
Cosine geometry is secondary and cannot establish source-specific content.

## Execution levels and CPU run

[Execution-level policy](../../RESEARCH-EXECUTION-LEVELS-2026-10-27.md) applies.
The operator explicitly authorized internally measured public FIT/DEV research
without external signers. `PUBLIC_SOURCE_RECEIPT.json` records the UCI publisher
page, CC-BY-4.0 license, DOI, official download bundle and exact nested original
archive hash. Other bundle members are not opened. The FIT loader stops numeric
source parsing before 2015, preserving protected 2016–17 TEST.

```sh
python scripts/phone_lab.py plan research/phone_queue/original-history-audit014.json
python scripts/validate_research_parity.py \
  --manifest research/mathematical-engineering/history_audit014/FAIR_COMPARISON.json \
  --phase preflight
python scripts/audit_original_history.py --execute \
  --implementation-root /clean/pinned-original-code \
  --source /authorized/PRSA2017-original.zip \
  --output /new/path/RESULTS.json
```

The amended router selects GitHub CPU, 30-minute estimate, zero GPU budget.
Internal preflight validates source/license metadata without keys; runtime
checks the actual original source, implementation, frozen cohort and all
checkpoint hashes before scoring. Output explicitly records `internal_research`
and `internally_measured_not_independently_qualified`; scientific qualification
and checkpoint promotion remain false. No neural fitting or model graph change.

`--execution-level independent_qualification` retains external source/evaluator
receipt verification, using independently pinned public keys and the exact
plan/code/source/checkpoint binding. Existing reward-integrity certification,
safety and promotion gates remain strict. Internal mode is rejected for
protected evaluation, TEST, reward certification, safety claims or promotion.
No signatures are fabricated or needed for the ordinary internal run.

The dedicated workflow separates constructed engineering tests from the actual
internal source evaluation. On this PR branch it downloads the publisher's
original archive, verifies its frozen hash, runs the frozen CPU diagnostic,
then repeats it from a fresh Python process and requires byte-identical output.
Full source traces and replay evidence are retained as Actions artifacts;
concise results belong in the final report. Any failure remains visible.

## Limitations and preserved history

`BLOCKED_PREFLIGHT.json`, `COMPUTE_DECISION.json` and the original static review
are retained snapshots from the previous signer-only policy. The authorization
amendment records the old plan hash/commit. It changes no scientific cases,
metrics, seeds, control selection, primary threshold or original checkpoint.

Prefix reconstruction uses rtol 1e-6 / atol 2e-6 for FP32 normalized values,
exact masks and exact causal last/age comparisons. Full source-file reload with
future-only perturbations remains **INCONCLUSIVE**; downstream feature-poison
fixtures do not certify that upstream boundary. Mathematical whole-model Lean /
Python / IEEE / AD equivalence remains **OPEN**, with no graph change here.

The cohort is exposed FIT, label-selected and small. Full historical training
FLOP parity is unestablished. Oracle keys carry historical observation content
and are privileged; oracle scores/uncentered cosine alone cannot demonstrate
stored information. Concurrent PRs #478–#480 are related prior research, not
independent qualification. Preserve all negative results and qualify any
reported memory benefit to this frozen internal diagnostic only.
