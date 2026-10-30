# LACUNA-ADDRESS005 — specificity not established on six new documents

Internally measured, not independently qualified. Three unchanged original
640-update checkpoints, six prospectively registered PSF/Rust documents.
The original scientific run passed all source/integrity checks on its first
execution. No fit, tuning, GPU, protected TEST, promotion or merge.

The primary formal specificity gate is **INCONCLUSIVE_SUPPORT_OR_CONTROL**:
all document support minima were met, but some required age-matched irrelevant
controls did not exist. Independently of that missing-control issue, the fully
observed correct-versus-hard-wrong margin was only **0.00001715 NLL**, below
the preregistered0.001minimum, and was negative in two of six documents. Neither
the native-query selection gate nor all-query prediction-utility gate passed.
No prior scientific gate is reopened; training remains stopped.

## Prospective design and data

Registration d43ae53a104c520d27e67c27a0bd021f8760562e preceded text acquisition.
Only publisher tree metadata was read first. Exact pinned Git blobs, sizes,
SHA256 and licenses are preserved in SOURCE_RECEIPT.json/source bundle.
Sources: CPython decimal/itertools/statistics and Rust book closures/interior
mutability/futures. These were new documents in the inspected research lineage;
publisher/phrase novelty or absence from all external pretraining is not proved.

Observe4096bytes, queries every4clocks t256..4095, same prefix-only four-byte
context selection and bins8–31/32–127/128–239. No future-label agreement selects
the cohort. Six owners, two publisher groups, three seeds. Average within owner
then equally across owners, with publisher/seed means reported separately.
Seeds repeat source targets; they never increase independent source support.

2598source/query/bin cells repeat across3checkpoints=7794measurements, involving
1661distinct current positions. The all-query denominator is5760source positions
and17280checkpoint measurements, including no-match positions.

The same fixed-span projection is applied to correct/wrong/shuffled/irrelevant
and two generic random-direction controls. It preserves historical, total slow
and actual mixed fast/slow read norms. Native trajectory/current write/query/
gates/fast read and decoder are unchanged; patched states are discarded. All14
candidate/control arms are retained. Supplied keys remain evaluator privileges.

## Primary address-specificity endpoint: ages8–31

Delta=NLL_wrong−NLL_correct; positive favors the intended supplied address.

| Source | Distinct short-age cells | Correct-versus-hard-wrong NLL margin |
| --- | ---: | ---: |
| CPython decimal | 36 | −0.00021166 |
| CPython itertools | 293 | +0.00009937 |
| CPython statistics | 106 | +0.00019073 |
| Rust closures | 33 | +0.00005292 |
| Rust interior mutability | 21 | −0.00004021 |
| Rust futures | 21 | +0.00001173 |
| Equal-document macro | 510total | **+0.00001715** |

All6owners exceed the fixed20-cell minimum. The prescribed margin was0.001
both overall and within each publisher. Actual publisher margins were0.00002614
(CPython) and0.00000815(Rust). All three seed macro margins were positive
(0.00000628,0.00002405,0.00002111), but two document means were negative.

The correct-address gain versus the original model was+0.00098458, with two
negative seed means and one negative document mean. It did not meet the
separately registered practical-utility gate. Improvement versus the original
alone remains insufficient evidence of correct historical content use.

## Required control availability and interpretation

The hard-wrong control was valid for every primary measurement, and there were
no numerical projection fallbacks. However,219of1530primary measurements lacked
a qualifying SAME-age irrelevant record:73source cells (54itertools,19statistics),
repeated across seeds. The implementation retained these rows with explicit
INCONCLUSIVE_MISSING_CONTROL placeholders. They block the full primary gate
as registered; no posthoc exclusion, source replacement or threshold relaxation
was used to claim a pass. This is missing control eligibility, not low document
support or a failed norm calculation.

Do not interpret the aggregate irrelevant-control score including placeholders
as a valid all-row contrast. A separately labeled **descriptive complete-control
subset** contains437source cells/1311measurements. Its macro correct-minus-wrong
margin is0.00002556, correct-minus-shuffle0.00017684, and correct-minus-irrelevant
is−0.00017784. These saved-data summaries do not replace the original gate.

Across all primary rows, the complete shuffled-value comparison is+0.00014662
and the generic random-direction mean comparison+0.00026047. Both are below
0.001; publisher-level contrasts are unfavorable for Rust. The observed data
therefore do not provide the intended meaningful correct-address separation.
No iid token/seed significance or broad negative theorem is claimed.

## Secondary ages and normal query

| Age | Correct-vs-wrong macro NLL | Supplied-key/raw gain vs original | Specificity gate |
| --- | ---: | ---: | --- |
| 8–31 primary | +0.00001715 | +0.00098458 | Inconclusive: missing required controls |
| 32–127 | +0.00005573 | +0.00108773 | Not passed |
| 128–239 | +0.00003672 | +0.00121704 | Not passed |

The older bins had complete relevant controls but did not meet specificity
or practical-utility requirements. They cannot be substituted for the primary.
Native-matrix secondary conditions also did not pass; all scores are retained.

At EVERY scheduled query, native q selected its highest-cosine retained writer
key before evaluator context selection. Its context-hit rate was0.1620%, versus
2.3785% for the fixed uniform-random key baseline:−2.2164percentage points,
against a required+1point advantage. Both publisher averages and every seed
average favored random; document effects varied. This concerns matching this
specific four-byte writer-key representation, not general semantic retrieval.

Native-query/raw C on all queries gained+0.00028890NLL versus the original,
below0.001, and lost0.00034783 versus mean generic random direction. Both
publishers favored the random-direction mean over C; original-model gain was
negative for two seeds and two documents. Neither query-selection nor all-query
conditional utility passed. Conditional native-query performance is not an
independent autonomous architecture rollout.

## Geometry, integrity and engineering verification

- Original100521parameters and1920Btensor state plus logicalclock unchanged;
  frozen weight hashes passed. Original9checkpoint archive/FIT replay passed
  within the documented missing historical-output/086RNG limits.
- All3norms verified numerically during every prepared read. Maximum float64
  errors: historical/slow1.12e-16,mixed2.23e-16; nativecast historical/slow<1.49e-8,
  mixed<4.00e-8. Thresholds1e-10/3e-6. Saved matched-case vectors independently
  recheck all3norms without model execution.
- Native A identity maxlogiterror5.73e-6; source slow_zero0; conditional transport
  <1.66e-7; reconstructed key/value<3.28e-7/8.95e-8; independent projection
  reference<2.78e-17. All preregistered source tolerances passed.
-18first-query prefix-poison checks on native-query/random preparation passed,
  max7.63e-6.18local composed projection→source-decoder VJP/FD checks passed,
  max0.000406<0.002. These sampled/local checks are not a global noninterference
  or all-parameter neural proof.
-51targeted Actions tests passed. Final full local suite1651passed/4skipped,
  existing profiler warning only. Initial local/full-CI integration check caught
  the not-yet-added report-verifier file; it is now supplied, the failing test
  passes, and full local suite was rerun successfully. Initial failed source CI
  41015666492 is preserved; scientific inference was not rerun for that repair.
- PNDS93first scoped Lean/Python CI41014601686 passed; bridgeCI41015661715passed.
  Proofs concern exact-real conditional norm algebra. Full NN IEEE/AD,
  independent qualification and canonical legacy-registry migration remainOPEN.

## Memory, compute and original artifacts

Native slow bank256B; reference raw ring256B includes8Bclock/248bytes. Actual
evaluator also holds the original1920Btensor state+logicalclock, ring,786432B
decomposition arrays, reconstructed keys/values, cache and reports. Equal bank
payloads do not establish equal-total-state autonomous efficiency.

Main stream73728observed-byte source steps.166032decoder attempts yielded148398
actual main decoder calls after exact perclock read caching; separate integrity
calls are outside that count.65754raw attention calls are derived from2perall-
query position,4permatchedmeasurement, plus18poison checks. Process CPU510.4454s
with2threads. Recorded peak RSS before final output serialization624197632B;
not a mobile latency, whole-process maximum or deployment efficiency result.

[Original successful run41015666434](https://github.com/FPC-effortless/TAC-OSM/actions/runs/41015666434),
job123931358743, artifact12960000689. Execution
8e2b160cc014448384a52c3521db6b4456677ba6; source code
10a32f0f556c101e1b50062cf33a604190ac20ef. Original ZIP SHA256
30642784198b3288ed69f5d857ef21111b7a3bf12061c864f34b73f21ae5dc98.

The Git-backed29.7MBoriginal ZIP preserves the full exact126.2MBRESULTS.json,
including all-case and all-query rows/vectors. SUMMARY.json excludes those rows
but retains complete group results and integrity receipts. The stdlib verifier
authenticates original ZIP and publisher sources; reconstructs the full prefix
cohort and exclusions; checks saved norms; independently recomputes group means,
support and all gates. It does not independently recompute neural logits or
establish scientific qualification. No inference was repeated for reporting.

The6documents are now exposed DEV. Preserve the formal inconclusive primary
status, small observed core margin, failed secondary/autonomous gates and all
earlier negatives/READNORM004exploratory signal. Keep training stopped. This
study does not establish that memory recall is impossible, nor select KDA,
addressing or decoder fusion as a trained architectural winner.
