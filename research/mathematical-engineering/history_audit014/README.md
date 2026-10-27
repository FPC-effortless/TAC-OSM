# HISTORY-AUDIT014: original-source historical-memory audit

**Status: PREREGISTERED / BLOCKED INDEPENDENT AUTHORIZATION.** The operator
confirmed on 2026-10-27 that independent source/evaluator receipts do not exist.
No original numerical data or trained checkpoint has been evaluated by this
increment. No source download, neural training, TEST access or GPU use occurs.

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
modified, missing and symlinked Python files. Only after independent permission
does it authenticate source/checkpoint bytes and load original weights.

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

## Authorization and compute

```sh
python scripts/phone_lab.py plan research/phone_queue/original-history-audit014.json
python scripts/validate_research_parity.py \
  --manifest research/mathematical-engineering/history_audit014/FAIR_COMPARISON.json \
  --phase preflight
python scripts/audit_original_history.py --output /new/path/preflight.json
```

The compute router selects GitHub CPU, estimated 30 minutes, **zero GPU budget**,
and reports evidence blockers. The parity gate returns `PASS_DIAGNOSTIC_ONLY`;
that is not evaluation permission. Preflight returns exit 2 and an immutable
`BLOCKED_INDEPENDENT_AUTHORIZATION` receipt before any original-data read,
checkpoint deserialization or model import.

For execution the independent source assessor and evaluator must each provide
a distinct Ed25519-signed authorization, using separately operator-pinned
`TAC_OSM_SOURCE_VERIFIER_PUBLIC_KEY_HEX` and
`TAC_OSM_EVALUATOR_VERIFIER_PUBLIC_KEY_HEX`. Neither key is taken from the PR
or a receipt. These are public verification keys; private signers must remain
outside agent/model access. The CLI has no fixture/force bypass or signing API.

Each signed envelope contains exactly `payload` and `signature_hex`. Payload:
`schema: tac-osm.history-audit-authorization.v1`, role `source` or `evaluator`,
the exact preflight `authorization_binding`, all role-specific `checks` in
`history_audit_authorization.CHECKS` with `PASS`, and a timezone-aware future
`expires_at`. The binding covers the full preregistration/proposal, execution
file digests, original implementation/source and every arm/seed/checkpoint.
The preflight artifact supplies the binding for independent review, not proof
the assessor completed the investigation. Signatures alone do not establish
real publisher provenance, actual ACL enforcement or evaluator independence.

Only after that separate work, the trusted CPU runner may invoke:

```sh
python scripts/audit_original_history.py --execute \
  --source-authorization /trusted/source-authorization.json \
  --evaluator-authorization /trusted/evaluator-authorization.json \
  --implementation-root /clean/pinned-original-code \
  --source /authorized/PRSA2017_original.zip \
  --output /new/path/original-history-results.json
```

This narrow frozen diagnostic authorization does not replace the existing
independent reward-runner signatures, 100k/parity gates or PNDS obligations for
any new fit, graph change, checkpoint promotion or capability claim. Output
always has `scientific_qualification: false`. Original source/evaluator
attestation checks cannot be self-signed by this agent.

## Verification and unresolved checks

The dedicated read-only CI checks the **original code** with constructed
inputs and ephemeral test-only signatures. It never downloads the source ZIP,
deserializes original checkpoints or executes the original-source audit.
Tests exercise actual projection capture, native decomposition, selected-key
resolution, absent-write rejection, future feature poison, tampered/missing
authorizations, code/plan/checkpoint binding, and altered code checkouts.
Fixture PASS is engineering evidence only.

Source-prefix reconstruction uses rtol 1e-6 / atol 2e-6 for normalized FP32
values, exact masks and exact causal last/age comparisons. Full source-file
reload with future-only source perturbations remains **INCONCLUSIVE**; a
constructed downstream feature poison is not that upstream integration test.
Source-qualified trace execution, fitted-checkpoint replay and source outcome
comparisons remain **NOT RUN / BLOCKED** until independent authorization.

The existing PNDS rules and reciprocal partial kernel records were inspected.
No model graph or weight changes are introduced. Full original-model Lean /
Python / IEEE / AD equivalence remains **OPEN**, and no kernel identity or
fixture test in this PR is represented as completing it. Original checkpoint,
data/compute parity and scientific failures remain immutable.
