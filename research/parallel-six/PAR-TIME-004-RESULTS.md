# PAR-TIME-004 — trained TAC has persistent state, but real FIT predictive benefit remains negligible

**Status:** Genuine authentic-source 2014 March–April FIT-period diagnostic only,
after five original PAR-TIME-003 real-source trained models. All original
checkpoint SHA values remain unchanged. This is a read-only frozen-model
intervention, not independent validation or a fair matched neural comparison.

## What is actually persistent?

The unmodified TimeAwareParallelResidualWorld streaming API carries:
- 88 FP32 **running historical input summary** values;
- one 8x8 FP32 **associative matrix** (64 scalars);
- one FP32 elapsed time since the start of the current stream;
- one FP32 last memory-write timestamp;
- one Boolean write-seen flag and integer event index/count.

The data-independent learned network weights are not counted in the state.
**154 FP32 scalars = 616 nominal bytes**, plus a Boolean and integer, at
the default neural geometry. Stream-step returns a new state to be passed
into the next call. There is no unbounded key/value history cache.

Fast and slow feature vectors at each observation are **derived** from
current embedding, previous summary and matrix read. They are NOT additional
independently persisted fast/slow recurrent tensors in this candidate.
The 12-event PAR-TIME-003 training and source scoring **reset the state**
at each training example. This demonstrates model-capable persistence
inside an observed stream, but NOT that weights learn useful memory over
days/weeks/months or across unrelated sessions. Durable external storage,
entity association, restart recovery, expiry and source consent would be
application responsibilities.

Exact-real PNDS v1.7 affiliate scan/stream mathematics only proves the
externally fixed affine memory schedule under its documented assumptions,
not neural/FP AD correctness or long-horizon memory utility.

## Preregistered 5-seed frozen source-FIT-only intervention

[Protocol](https://github.com/FPC-effortless/TAC-OSM/blob/research/partime004-frozen-fit-state-lesions-2026-10-27/research/parallel-six/PAR-TIME-004-PROTOCOL.json)
and [full original per-seed tensor source metrics](https://github.com/FPC-effortless/TAC-OSM/blob/research/partime004-frozen-fit-state-lesions-2026-10-27/research/parallel-six/PAR-TIME-004-RESULTS.json).

64 actual source FIT-period histories: 32 with at least one authentic
>1-hour gap and 32 with all observed prior deltas one hour,
selected **only by original chronological source inputs**, independent of
next-hour physical targets. This is the original training PERIOD and should
not be described as a new independent data holdout.

Original checkpoint seeds: 271028–271032; 5 frozen models x 64 histories,
each of length 12 source-recorded chronological inputs. All source
measurements and original normalizers are from the immutable SHA-verified
original publisher PRSA air source; protected 2015–17 numerical target
records and previously-exposed 2014 DEV outcomes NOT read by this study.

Temporary time-projection column lesions are applied in volatile memory
to pretrained inference and undone exactly; associative-read lesion zeros
the carried matrix BEFORE each source timestep while leaving fast-prefix
summary, previous timestamp, observed inputs and train weights in place.
This preserves zero-history matrix reads without fabricating new source data.
No alternative models are fitted for these inference interventions.
The trained model's source covariate STILL contains known hour-of-day
sine, so eliminating the new explicit time projection does not remove
ALL time information.

### Actual observed 13-channel normalized source-FIT MAE, lower is better

| Frozen intervention | Mean across 5 original models |
| --- | ---: |
| Unmodified time-aware TAC | 0.2492290154 |
| Temporal projection all 7 columns zeroed | 0.2492234811 |
| Previous associative matrix cleared each step | 0.2492326766 |
| Last-real-observation persistence | 0.2489932409 |

- The original model is 0.0947% WORSE than simple persistence.
- Removing the learned temporal projection has a tiny *favorable*
  MAE effect on these FIT source histories (-0.0022% relative error).
- Removing historical associative reads has an extremely small unfavorable
  effect (+0.00147% relative MAE). This is not robust evidence that
  matrix memory improves useful real forecasting.
- Both lesions DO change original output: mean absolute normalized
  change 5.3217e-05 (all new temporal columns disabled) and
  1.2638e-04 (associative matrix read cleared). These mechanisms are
  computationally active but their observed influence on genuine FIT
  predictions is very small.
- Original 8x8 matrices have nonzero entries on these source histories,
  mean absolute matrix magnitude after 12 events ~0.05483.
- Exact fitted batch/stream output tolerance checks PASS across all five
  source-trained seeds; tensor state saved after six observations using a
  typed weights-only in-memory serialized payload restored losslessly;
  predictions after resuming match the uninterrupted same-process stream
  EXACTLY in this diagnostic. This is a technical snapshot/resume property,
  **not** cross-process service persistence or multi-day recall evidence.
- Under **all-write** source records, \(\log(1+\mathrm{lastWriteAge})\)
  is numerically identical to \(\log(1+\Delta t)\): max measured
  discrepancy **0**, leaving strong collinearity. Thus the independent
  benefit of previous-write age cannot be identified from this protocol.

No optional model was chosen after inspecting favorite seeds; all five
original checkpoints and unfavorable measurements are preserved.

### What this means for next research

The failure to beat persistence is not primarily proof of failure of the
persistence *data structure*. The research question is whether the
training objective supplies enough gradient pressure to **store and
retrieve relevant older content**. The current objective uses only
next-hour change, from 12-event windows, with an unchanged
source-normalized residual scale and strong persistence connection.
On slowly moving atmospheric measurements that task can be solved
almost as well by copying the last observation, which weakens
pressure to learn historical contributions.

The scientifically important next capability-bearing trial should:
1. Freeze a source-authorized **genuinely longitudinal** data split with
   target-relevant lag at 12, 32, 64 or more *recorded* events, and
   labels revealed only after source-legal causal queries; avoid
   retroactively opening protected/previously-exposed numerical years.
2. Train continuous per-entity streams with explicit recoverable state,
   not only disjoint 12-event resets, then measure state survival across
   process restarts. Avoid source-permission assumptions.
3. Compare genuinely trained no-read/time-blind/temporal variants
   within 1% active parameters and audited full training-operator FLOP
   budgets, equal optimizer/data/tuning/splits and >=5 seeds. CPU
   wall-time is separately an efficiency result, not an enforced parity.
4. Include *strong* persistence, linear/ridge, GRU and SSM comparators;
   assess error by delay, source-group confidence, genuine information
   reconstruction and uncertainty with no cherry-picked channels.
5. Freeze original trained model+optimizer+source timing/order hashes
   and replay from fresh repository checkout BEFORE any new protected
   TEST scoring.

Source code in scripts/partime004_frozen_fit_lesions.py and original
source-FIT proof-of-operation CI; new standalone read-only replay CI checks
all previously committed 5-seed metrics without writing any trained model.
All six active research families and all earlier failures stay intact.
