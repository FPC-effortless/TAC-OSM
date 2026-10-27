# PAR-MEM-006 — authentic source keyed historical retrieval: associative-memory failure

**Status: original SHA-replayable real-source exploratory result; NO qualified
long-history recall, NO fair architecture superiority, NO action policy.**

[Registered protocol](https://github.com/FPC-effortless/TAC-OSM/blob/research/parmem006-source-history-query-2026-10-27/research/parallel-six/PAR-MEM-006-PROTOCOL.json) |
[Source-query and optimizer seed plan](https://github.com/FPC-effortless/TAC-OSM/blob/research/parmem006-source-history-query-2026-10-27/research/parallel-six/PAR-MEM-006-FROZEN-PLAN.json) |
[10 head original optimizer+RNG+SHA receipt](https://github.com/FPC-effortless/TAC-OSM/blob/research/parmem006-source-history-query-2026-10-27/research/replay-bundles/PAR-MEM-006/TRAIN_RECEIPT.json) |
[Original individual five-seed measured DEV scores](https://github.com/FPC-effortless/TAC-OSM/blob/research/parmem006-source-history-query-2026-10-27/research/replay-bundles/PAR-MEM-006/DEV_RESULTS.json).

## Genuine source & legal query

The exact publisher UCI PRSA2017 Beijing 12-station source is bound by
SHA256 `d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8`.
Observed historical event identifiers are only source station and
its already-recorded chronological event index. The task was to recover
the real recorded PM2.5 value of an event **16, 64 or 128 genuinely
observed source events earlier**. At the query event, the past value
is never supplied in query metadata or directly in the head's input.
Only actual previously observed source time, time-of-day/day-of-week
features, event index and lag are available. The old value **was**
observed during earlier sequence processing, so the task is a
retrospective source-reconstruction problem on real measurements,
not real service adoption or future forecasting.

The frozen pretrained 96,639-parameter time-aware TAC core comes from
real-source PAR-TIME-005. For each query, a new 3,913-trainable-parameter
head reads:
- frozen 88D cumulative source context;
- 14D source current known observations / availability;
- query-key generated from **only historical source event time**;
- optionally the 8D read of the 8x8 historical associative matrix.

Total 100,552 deployed learned params per model (within 95k–105k).
Five pretrained true-source source seeds × 2 query-head arms:
**10 trained retrieval heads**, 64 AdamW steps at batch64 each,
same original training source identities and sample selection order.
Original sampling cases/optimizer orders committed BEFORE new training,
all 10 checkpoint/optimizer+RNG files committed BEFORE opening the
historically-exposed 2014 May–June DEV source answers. Per-case original
DEV predictions frozen to disk for clean-checkout replay.

**Important control precision:** the arm named `history_blind`
in source code turns off the **associative matrix read only** and
still retains the 88D cumulative TAC fast state, today's measured
observations, and time metadata. It is therefore strictly
ASSOCIATIVE-BLIND, NOT fully stateless. Both heads have exactly the
same non-dummy active weight count and query-operator shapes.
Complete original **full training forward+backward+optimizer FLOP
parity is NOT yet certified**, so do not call this a certified
fair neural architecture comparison.

The original 2014 DEV source was previously exposed by other TAC
studies; measured data here are genuine but NOT an independent holdout.
No original source numerical years 2015–2017 were accessed.

## 5-seed measured source historical reconstruction results

Normalized PM2.5 MAE; lower is better. Exactly **1,032 actual
source-measured historical queries per lag, seed and arm**.

| Past source event lag | With associative read | Associative-blind | Copy current source PM2.5 |
|---|---:|---:|---:|
| 16 | 1.06938368 | **1.05220737** | **0.69097710** |
| 64 | 1.09265335 | **1.05849376** | **0.80004185** |
| 128 | 1.11924877 | **1.07976314** | **0.81426674** |

Relative associative penalty compared with otherwise identical
trained ASSOC-BLIND control: **+1.632% (lag16)**,
**+3.227% (lag64)**, **+3.657% (lag128)** on this exposed DEV
source cohort. Only 1/5 assoc heads win over their paired
assoc-blind peer at lag16; 1/5 at lag64; 2/5 at lag128.
Source-measured exact-history oracle = 0 MAE, but **requires keeping
the actual history externally** and does not share TAC's fixed-size
154-float internal memory budget. Never compare oracle as if fair.

This is an unambiguous failure to recover historical values with
this particular pretrained memory/key representation and trained
timestamp query. Trained true-source 768-event persistence **did not**
result in useful exact historical retrieval.

## What we learn, without overclaim

1. **Software persistent state is real:** carried 8×8 matrix and
   input summary survive 768+ source events, as shown by PAR-TIME-005.
   Memory is neither disabled nor inherently transient because of
   explicit timestamps.
2. **Observed readout usefulness is not established:** the learned
   query key derived only from source event time cannot reliably
   address keys that were originally computed from **event content**
   (the pretrained `core.address(e_t)` includes measurements).
   The head can instead learn correlations with current measurements
   and time; the associative matrix read may add interference.
3. **Query/read key misalignment is a candidate mechanism, not proven
   sole cause.** Information loss and overwriting in the small 8×8
   superposed matrix, narrow residual, undertraining, task mismatch,
   and incomplete source observational coverage are viable contributors.
4. **The control is not stateless.** Its 88D prefix does retain
   past input information. A truly stateless 100k model and
   source-legal feature-only baseline are separate next controls.
5. **Not a fair total-compute architecture claim.** Although all
   learned active parameters/optimizer steps/data/splits match, full
   nonlinear/AD/optimizer mathematical FLOP parity was not audited to
   1%. CPU wall time is not a fairness constraint. Independent
   untouched real-source qualification remains OPEN.

## Next decisive diagnostics (not silently claimed as completed)

Preregister all before fitting:
- A **source-permitted privileged oracle-key diagnostic**
  that uses the key ACTUALLY WRITTEN by the existing frozen model
  to query a past source event, compared with the timestamp-only
  query. This may expose query-address misalignment without granting
  a deployable ability; mark oracle privileged and DIAGNOSTIC only.
- A fixed **same-byte budget** learned time-indexed write and read
  address map (both sides trained, no target supplied to query),
  with matched learned associative-blind and truly stateless controls.
- Genuine source event delays 16/64/128/256 and matched data,
  frozen optimizer source order, complete forward/back/AdamW FLOP
  parity, and independent source-heldout evaluation. Do not reuse
  the historically exposed May–June development as a fresh TEST.
- Analyze query source selection, per-key collisions, retrieved
  physical PM2.5 target information and confidence/calibration.
  Measure whether extra memory helps *after controlling current
  value and source time*, not merely whether a matrix has nonzero norm.
- Preserve original 10 checkpoints, trained frozen source base,
  source legal provenance and full exactly-replayable predictions.

**Engineering:** the independent read-only replay workflow
`.github/workflows/parmem006-independent-replay.yml` reconstructs
original source history, matrix and head outputs without training.
Pass status must be confirmed from GitHub before claiming cold replay.

No original trained checkpoint was overwritten. All six research
families, previous negative results and heldout protections preserved.
