# PAR-MEM-011 — source-authentic trained shared-key memory, historical retrieval still unqualified

**2026-10-27 | Genuine source experimental development; 10 original
checkpointed neural models and cold clean-checkout replay PASS.**

- [Original study protocol](PAR-MEM-011-PROTOCOL.json)
- [Frozen source owner/time and all-seed fit schedule](PAR-MEM-011-FROZEN-PLAN.json)
- [Original five-seed ten-model optimizer/checkpoint SHA receipt](../replay-bundles/PAR-MEM-011/TRAIN_RECEIPT.json)
- [Full original individual seed scores](../replay-bundles/PAR-MEM-011/DEV_RESULTS.json)
- [Original GitHub Actions source training and clean-checkout replay](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40534079644)
- [Independent training full op-shape matched source preflight](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40533311469)

## Why this experiment follows the master research plan

PAR-MEM-010 reintroduced mechanisms already implemented in historical
DIAGKEY004 shared known-key identity, ME100 observed-only source admission,
LATFUSE004 multihead soft writing and LATFUSE018 supervised real physical
historical content reconstruction. This test, PAR-MEM-011, actually **trains
the integrated 98,326-parameter, 410-FP32-state model** on physically
observed historical PM2.5 values at exact known source event IDs.

Past targets are genuine UCI PRSA2017 Beijing air sensor measurements in the
SHA-verified original publisher source (SHA256
`d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8`).
Only 2013–2014 actual observations were parsed; original normalizer fitted
before 2014-03-01. Fits occur on original March–April 2014 observations
at 12 station identities. Development covers May–June 2014, which was
**previously viewed** in earlier TAC-OSM studies and is NOT an untouched
cohort / TEST or independently qualified capability.

## Original registered learned intervention

Two physically identical networks per seed, each 98,326 trainable
parameters and 410 FP32 source memory values, share the same original
pretrained source-time backbone, input events, actual timestamps,
source-known ordinal Fourier addressing, 129 physically observed event
history windows, training optimizer/hyperparameters, registered
sample order and training updates.

- `correct_shared_source_key`: final historical query addresses the actual
  known previous **observed event** 16, 64 or 128 events earlier.
- `adjacent_wrong_source_key`: query addresses the **next genuinely observed
  historical event** of the *same source station*, also earlier than the
  current event. This still uses a learned source-admitted associative memory
  and still updates all learned parameters. It is an ADDRESS-TARGET control,
  **not a stateless/zero-memory architecture comparison**.

Five original seeds: 271028–271032. Each model received 18 genuine
source-supervised AdamW updates at batch 4 source-identity windows,
cycling lags 16/64/128, with historical normalized PM2.5 SmoothL1
loss plus 0.15 times source-measured next-hour masked physical loss.
Training history window is 129 source-observed events per case.
The registered 10×18-optimizer-step exposure is modest and not a claim
of training convergence or exhaustive optimization.

Source prefit original chronological query/owner/sampling schedule was
committed to Git BEFORE original training; all 10 original complete
model+AdamW+RNG/source/code SHA checkpoints committed BEFORE opening
May–June development metrics. Every original per-case prediction, source
identity and target tensor was then saved, committed, and independently
recomputed from a clean checkout with no optimizer in that replay.

### Training operator fairness

The actual source graph preflight compared **1,926** distinct ATen
operator/input-shape signatures in a source-real forward, backward,
gradient-clip and AdamW step of the two arms: **zero differences**.
Torch profiler recognized exactly **636,061,480** FLOPs in each arm,
a **0% gap in the supported subset**. The two models literally use
the same differentiable architecture and source input shapes, with
the target event ordinal offset being the only semantic intervention.
However **the full scalar-operation-weighted forward/backward/optimizer
mathematical FLOP audit is not yet certified** across unsupported
non-matmul operations and all five source schedules. This study does not
claim independently qualified fair neural architecture superiority.
CPU wall time parity is *not* a scientific fairness requirement.

## First original genuine source DEV scores

For each lag, the same 120 real source-station/anchor queries per
seed/arm (12 stations × 10 fixed recorded anchors); old PM2.5 was a
genuinely previously observed numerical source value, legally available
at its original write and intentionally not passed to the query input.
Metric: mean normalized historical PM2.5 MAE, **smaller is better**.

| Exact actual source-recorded lag | Correct source-known write/read query | Adjacent wrong source-event query | Copy CURRENT PM2.5 (weak context-only baseline) | Correct-key seeds beating wrong |
| --- | ---: | ---: | ---: | ---: |
| 16 events | 0.604188 | 0.604963 | 0.734462 | 4/5 |
| 64 events | 0.508405 | 0.507507 | 0.573954 | 1/5 |
| 128 events | 0.574039 | 0.573246 | 0.665352 | 1/5 |

- Shared-address model improves historical-target MAE versus simply copying
  the current observed PM2.5 by about 17.7%, 11.4%, and 13.7% at
  lags 16/64/128 on the selected real DEV anchors. This is a **weak
  observational baseline** and does NOT by itself identify memory benefit.
- Correct historical source address beats WRONG adjacent historical
  source address by only **0.128%** at 16 events.
- It is **0.177% WORSE** than wrong-address at lag 64 and **0.138% WORSE**
  at lag 128 (both 1/5 paired seed wins).
- On genuine next-hour PM2.5 among the same evaluation source anchors,
  mean normalized error was ~0.163041 for correct and ~0.163044 for
  wrong address, an insignificant descriptive difference. This is **not**
  an independently qualified physical forecast superiority comparison;
  no stronger persistence/GRU/SSM comparator was trained here.
- The original publisher source true past-value oracle yields 0 historical
  error only by storing/retrieving the **entire actual event history**.
  This is NOT an equal-memory fixed-size neural comparator.

**Primary conclusion: REAL historical-target fitting is feasible in the
integrated new 410-value TAC source-memory architecture, and the simple
current-value copy baseline is exceeded on these source cases. But the
memory readout remains insensitive to the CORRECT historical event's
address: exact event-specific long-delay retrieval at 64/128 events
is NOT demonstrated.** The trained wrong-key control doing as well
suggests the head may exploit temporal/weather priors, source context,
collapsed values, or nonlinear stored summary rather than retrieving
specific event content. A high-rank source-known Fourier address does
not guarantee actual retained value/retrieval benefit.

## Independent limits and next study selection gates

This does not qualify a seventh TAC model family, deployment, active
physical control, confidence/probability calibration, causally
beneficial event memory, or independent unseen-site/period
generalization. The 18-update fit is short; longer training may matter,
but repeatedly tuning on this previously exposed 2014 DEV cannot
create an independent real-world success claim.

Before spending more fit budget:
1. **Attribution first:** frozen original trained checkpoints, equal
   genuine case cohort, source-time controlled wrong-key and zero-read
   lesions, real original-event value reconstruction. Add near/far
   off-address discrimination and per-head write/read entropy/collisions;
   do not train yet another unregistered microvariant on exposed DEV.
2. **Strong controls:** genuinely stateless source/time model, no-reader
   TAC with useful 88D fast/slow summary, seasonal/linear recent-context
   regression, matched GRU/SSM and correct fixed-source prior.
3. **Original source authorization:** future cohort must have stable
   source-verified event IDs, observed- and available-at clocks, and
   old targets genuinely observed before the query. Keep 2015–2017
   numerical source years unopened until independently authorized
   and protocol-frozen; do not relabel old DEV as virgin TEST.
4. **Strict science:** 95k–105k actual learned weights with ≤1%
   cross-arm mismatch, full mathematical training FLOP parity,
   equivalent data/splits/optimizer budget, five seeds, all original
   model/optimizer/RNG/state and source schedule replay committed
   before any new TEST scoring. CPU runtime not forced equal.
5. **PNDS:** operator-level exact-real source key equality and
   gated-delta/owner query safety are covered in PNDS companion #49,
   but the full four-block affine scan's FP32 refinement, full
   autodiff and probability-law refinement remain incomplete.

Old trained models and checkpoints untouched, original negative
comparisons preserved. PR remains draft/unmerged.
