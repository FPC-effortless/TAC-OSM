# PAR-TIME-001 — positional order, source elapsed time, and prior-write age

**Status: untrained optional version; NO real-data capability or superiority claim.**

## Why this is a distinct prospective version
Base: draft PAR-FFN-001 in TAC PR #500 and PNDS PR #46.
This proposal adds seven **input-only** temporal features to the learned 48D
observation embedding; they reach the address, content, pre-scan fast/slow
proposals and input-conditioned matrix gates. It retains the exact inherited
post-scan bounded residual FFN without a seventh model family.

At current source event index j with nonnegative source-provided delta dt_j,
clock t_j=sum_{i<=j}dt_i, and last previous source-admitted write clock L_{j-1}:

    phi_j = [ log1p(j), sin(j/16), cos(j/16),
              log1p(dt_j), log1p(t_j),
              log1p(t_j - L_{j-1}), seen_write_before_j ]

    e_j = tanh(encoder([observed_x_j, recorded_action_j])
               + temporal_projection(phi_j))

The 7x48 temporal projection starts at zero, preserving the unmodified model
prediction at initialization WHEN all events are admitted and learned weights
are copied from the parent. Its parameters remain fully differentiable.

**Age means only the time since the most recent PRIOR ADMITTED write.** A
superposed associative matrix has no exact per-record age or identity,
and this model does NOT claim one. The explicit source-owned Boolean
write mask is applied after computing today's query, so current write does
not affect current memory read or current age. An all-true mask means each
source observation contributes to the matrix; an all-false mask creates no
new associative writes. The mask must not come from future labels or target
knowledge. It is not a learned optimal selective retention policy.

The affine scheduled matrix update is unchanged and computed exclusively from
e_j and current input-only prefix evidence. The only chronological recurrence
during streaming is the carried input prefix, matrix, clock values, seen flag
and event count. Neither past/future teacher labels nor downstream readouts
flow backward into the same scan gate schedule. This preserves legitimate
associative chronological prefix composition, subject to numeric tolerances.

## Exact model accounting

Default dimensions: observed state14, recorded action14, width88,
readout decoder132, residual FFN hidden12, time features7, embedding48.

- Parent PAR-FFN-001: **96,303 learned/deployed parameters**.
- Time projection: **7x48 = 336 live new parameters**, no bias.
- Time-aware candidate: **96,639 parameters**, within the compulsory
  95k–105k target and 0.3489% over the parent parameter count.
- Parent streaming state: 88 floating prefix + 64 matrix floats = 152.
- Time-aware streaming state: 88 + 64 + 1 elapsed + 1 last-write
  = **154 FP32 scalar values (616 nominal bytes)** per entity,
  plus one Boolean seen flag and one integer event count.
- Learned weights, source metadata, entity IDs, runtime tensors, clocks'
  units/precision and future potential cache memory are separate and must
  be counted in total device deployment measurements.

When clock deltas vary by entity, batch input uses [B,T] deltas and [B,T]
Boolean write masks. Streaming consumes [B] per step. Every delta must
be finite and nonnegative and reflect timestamp information actually known
when the prediction is requested. The implementation forbids silently
assuming a fixed one-second interval. Time units (seconds, hours, frames)
are domain-specific and must be fixed by authentic source registration.
The model does NOT handle absolute calendar-time effects or per-record
missing time; those need a separate typed source adapter.

## Required scientific controls — not executed

1. Constructed no-fit tests of source time validation, chronological
   order sensitivity, explicit previous-write age, exact no-write matrix
   isolation, prefix future-poison independence, masked scan/stream output
   and parameter-gradient parity. These are not performance evidence.
2. Parent-preserving initial projection check: copied base weights with
   all-write mask must reproduce the parent output. Explicit projection
   gradients must be nonzero on constructed fixtures.
3. Prospective authentic time-stamped sensors/physical events, independent
   entity/source groups, train-only normalization, sealed independent
   future heldout, preregistration and replayable weights/optimizer/RNG.
4. Run time-blind parent (96,303), time-aware treatment (96,639) with
   parameter mismatch <1%, plus matched GRU, SSM/Mamba-style comparator
   and practical persistence/constant-velocity/ridge baselines.
   **Full forward/backward/optimizer FLOP parity has not yet been verified.**
   If workloads differ >1% in registered training FLOPs, match or explain
   without claiming a fair superiority result. CPU runtime need NOT match.
5. Independent ablations: position-only, delta-time-only,
   elapsed-clock-only, age-since-prior-write-only, all-features, shuffled
   legal clocks, no associative read and no-write controls. Do not turn
   corrupted future source time into a legal model input.
6. Primary real-data outcomes: proper point MAE/MSE, uncertainty NLL and
   source-group coverage, performance by event delay and write density,
   long-horizon retention, actual training/stream latency, and total
   CPU/GPU resource accounting. No score from constructed fixtures may
   be called real improvement.

## Mathematical verification boundary

PNDS counterpart: ParallelTimeAge.lean and parallel_time_age_math.py.
Theorems establish **exact-real scalar** clock monotonicity under
nonnegative dt, age growth without writes, write-time reset, and integer
index progression. They do not verify logarithm/trig numerical precision,
whole neural Torch/Lean AD, the probability laws, bounded index collision
at very large steps, timestamps for each mixed memory item, or trained
real-world skill.

This work is stacked as separate drafts on the previous PAR-FFN-001
branches; original six families and 17 indexed research records, all
trained weights and mixed/failed historical gates are unchanged.
