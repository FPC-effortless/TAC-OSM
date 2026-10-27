# PAR-TIME-005 — 768-event real persistent-stream training: evidence and limits

**Current scientific status: completed genuine-source single-trained-architecture
development, NOT independently qualified and NOT a matched learned-model win.**

[Source protocol](https://github.com/FPC-effortless/TAC-OSM/blob/research/partime005-long-continuous-state-training-2026-10-27/research/parallel-six/PAR-TIME-005-PROTOCOL.json) |
[Source-time plan](https://github.com/FPC-effortless/TAC-OSM/blob/research/partime005-long-continuous-state-training-2026-10-27/research/parallel-six/PAR-TIME-005-FROZEN-PLAN.json) |
[Original optimizer/checkpoint SHA receipt](https://github.com/FPC-effortless/TAC-OSM/blob/research/partime005-long-continuous-state-training-2026-10-27/research/replay-bundles/PAR-TIME-005/TRAIN_RECEIPT.json) |
[Full original five-seed real development results](https://github.com/FPC-effortless/TAC-OSM/blob/research/partime005-long-continuous-state-training-2026-10-27/research/replay-bundles/PAR-TIME-005/DEV_RESULTS.json).

## What was built and physically observed

- Original immutable UCI Beijing PRSA2017 multi-station source archive,
  SHA256 d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8.
  The original source reader parses measured pollutant/weather channels
  from 2013–2014 ONLY. 2015–2017 numerical values are untouched.
- Training-only normalization: real records ending before 2014-03-01.
  Source-observed March–April 2014, 12 genuine station entity streams.
- Five trained original models, 96,639 active parameters each, original
  seeds 271028–271032. Exactly **24 AdamW optimizer updates x 32
  chronological source observations x 12 independent real stations**
  = 768 observations per station and 9,216 station-observation steps
  per model, 46,080 station-observation steps over five runs.
- Every model's returned state is *carried from one update to the next*
  without reset. Automatic differentiation IS truncated and state tensors
  detached after each 32-event optimizer block; the recurrent STATE
  VALUES are not cleared between optimization blocks. Learned weights
  change each block; this is ordinary truncated-BPTT, not equivalence
  to full 768-step backpropagation.
- State is constant-size per entity: 88 float input summary values,
  64 associative matrix float values, 1 elapsed-time clock, 1 previous
  write clock = **154 FP32 scalars** (616 nominal bytes), plus one
  Boolean and an integer event counter (integer actual storage/runtime
  not included), trained parameters and process/device allocations
  excluded.
- Source-current `observed_action` covariates are genuine availability
  flags and known hour phase, NOT actual executed agent commands.
  Forecast targets are next clock-hour real measurements, masked when
  the channel was physically missing. No protected label known at
  current prediction is an input.
- All five source time schedules were frozen in a Git commit before
  any optimizer update. All original weights/AdamW optimizer/RNG/final
  persistent state snapshots and SHA hashes were frozen in another commit
  before 2014 May–June development scoring. Original full per-model
  forecasting sequences subsequently saved for cold replay.

## Real 2014 May–June age-banded development outcomes

The model was restarted only at the BEGINNING of each station's
development stream, then state was retained for 830 chronological
source-observed events per site. May–June 2014 had already been examined
in earlier TAC research; this is **development only, not new TEST**.

Five-seed mean true 13-channel normalized next-clock-hour MAE,
where smaller is better:

| Observations since fresh start | Time-aware trained TAC | Genuine last-observation persistence | Relative TAC difference |
| --- | ---: | ---: | ---: |
| 32–63 | 0.31475728 | 0.31466861 | +0.0282% worse |
| 64–127 | 0.25951956 | 0.25938348 | +0.0525% worse |
| 128–255 | 0.23534247 | 0.23526712 | +0.0320% worse |
| 256–511 | 0.25507272 | 0.25502255 | +0.0197% worse |
| 512+ (through 830) | 0.27353900 | 0.27346459 | +0.0272% worse |

**All five seeds lose on aggregate normalized multi-sensor MAE in every
age band.** This is genuine negative evidence; do not cherry-pick
PM2.5 or the occasional marginal PM2.5 win to claim overall success.
PM2.5 normalized forecasts have a few very small improvements in
individual bands, but those do not overturn the overall result.

Age-band metrics from original real observed source cohorts should NOT be
interpreted as a direct causal effect of longer memory: early and late
time ranges contain different atmospheric events. To establish causal
memory benefit, compare the same source anchors with controlled
no-memory/time-blind policies and equal registered compute.

### What persistence IS and IS NOT

**Proven at implemented model/software level:** a real source-trained
model can keep its recurrent matrix, source summary and clocks from
one source observation through 768 consecutive training events and
830 consecutive evaluation events, even across optimizer updates.
It has neither a per-token KV history cache nor a hidden matrix that
expands with the source history length. Checkpoints preserve original
final training persistent states.

**NOT proven:** recovery of individual past facts across 768+ real
events, causal benefit of its associative matrix for these predictions,
predictive improvement over source persistence, retention of memory
across application/server restarts without explicitly serializing and
restoring source-scoped state, or low-latency full-device deployment.

**Why this negative result matters:** a model with
\(\hat{x}_{t+1}=x_t+\delta_\theta\) can simply learn a near-zero
residual and approximate last-observation persistence on slowly evolving
source channels. That lets model training run and produce valid weights
without supplying incentive for historical content retrieval. The
solution is NOT to relabel software persistence as demonstrated useful
long-term memory.

## Next research gate

1. From immutable source-time qualified true longitudinal data, choose
   an actual historical **source-dependent** prediction/reconstruction
   query where earlier real records determine the answer and current
   input alone cannot. First verify ground-truth event identity and
   source-known-at-query legality; never use privileged keys or future
   outcomes. Reuse existing negative LATFUSE reports as controls.
2. Train true streaming with matched parameter counts (within 1%),
   source/splits, optimizer budget and COMPLETE training FLOPs for
   associative/no-read and temporal/time-blind comparators. Torch
   profiler-supported matmuls only are insufficient for full parity.
   CPU wall time need not be made equal.
3. Register 32/64/128/256 event lag breakdown and independently untouched
   source cohorts with all five checkpoint+optimizer+RNG+state resume
   materials. Strict no-read inference lesions are useful DIAGNOSTICS
   but do NOT replace equal-trained-controls.
4. Add a safe source-scoped state snapshot/restart test at a true
   stream boundary, checking that a cold-process resume matches
   uninterrupted predictions; no claims of automatic cloud persistence.

### Independence and replay boundary

PAR-TIME-005 original training workflow freezes exact source chronological
orders, trained checkpoints and full development forecasts. The separate
read-only GitHub Actions workflow
`.github/workflows/partime005-independent-replay.yml` reconstructs
each genuine source trajectory without fitting. Do not mark a final
clean-checkout replay success until the standalone workflow passes.

All earlier TAC family designs, original trained checkpoints and
negative research records are unchanged; this remains an unmerged
draft PR. No GPU quota, external test set, physical actuator, protected
years or new model families were consumed.
