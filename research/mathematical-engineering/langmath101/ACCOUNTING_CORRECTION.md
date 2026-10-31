# Accounting correction after the first optimizer updates

Run 41169500006, source 3a83fd79069c13b8f6aea37a86062e96fe37ee1e,
passed preflight and attempted all 138 fits. Every job completed one AdamW
update, then failed in ActualTrainingLedger.__exit__ because AdamW emits
`profiler::_record_function_enter_new` and `profiler::_record_function_exit`.
The validator had incorrectly required every dispatch event to be ATen.
All 138 original job traces were inspected; the appended audit preserves each
failure location and all 138 original artifact identities/digests.

The old FAILURE.json counts len(journal), which is zero when ledger exit
raises. Its zero-update report is incorrect: the exact source executes
opt.step() before that exit. There were 138 failed first updates, not zero.
No model sealed, no candidate DEV predictions were scored, and the fits are
excluded from every comparison. The old runner retained initial models and
failure reports but **did not retain the updated model/optimizer bytes**.
Those lost states cannot be audited or reconstructed as original checkpoints.
The original artifacts remain unchanged; no fabricated replay is supplied.

The correction accepts only the two explicitly named, tensor-free profiling
markers. It retains their complete rows in the fingerprint and uncharged
operator list. Unknown namespaces and tensor-bearing markers still fail.
The mm/addmm/bmm formula, model graphs, data, tokenizers, schedules, recipe,
parameter limits, counted target, stopping rule and evaluation gates are
unchanged. This remains partial dense arithmetic, not complete training FLOPs.

The runner now increments completed updates immediately after opt.step(),
separately records attempted updates and ledger entries, and saves failed
model, optimizer, RNG and actual per-slot step counters on any training error.
A failed fit remains ineligible for sealing and evaluation.

PROFILE_INDEX.json is preserved byte-for-byte. The new
PROFILE_INDEX_ACCOUNTING_V2.json explicitly inherits its zero-update model
measurements and records the four instrumentation/source changes. It does not
pretend these measurements were made on new hardware or erase prior updates.
Before corrected optimization, Linux CI must recheck all 69 exact original
FIT inputs, initial tensor identities, gradients and full forward/backward
operator fingerprints. The original initializations were generated on Linux;
a local Windows attempt failed initial tensor identity and is not accepted as
profile reproduction. No weights or tolerances were changed to pass it.

Local Windows verification: 203 reference, candidate, causal-interface, parity,
probe and accounting tests passed. The new constructed optimizer regressions
check exact first/warm AdamW weight and moment equality against untraced
updates, unchanged dense counts, retained marker fingerprints, strict rejection
of unknown/tensor-bearing markers, and failure preservation after a completed
update but before ledger completion. These fixture tests are engineering
evidence, not research fits or capability results.

The ordinary preflight and CPU planner pass for the corrected source bindings.
Only Linux CI can establish the required frozen-profile recheck. A corrected
fit attempt is allowed by the owner's continuation request; it restarts every
arm from its frozen initialization and preserves the failed attempt separately.
No candidate, recipe, data, target or endpoint selection used DEV results.

The existing PNDS graph/reference pins remain unchanged because no candidate
computation graph or mathematical formula changed. Whole-model IEEE/AD remains
OPEN. Protected TEST, independent qualification, promotion and merge remain
unauthorized. Results are internally measured, not independently qualified.
