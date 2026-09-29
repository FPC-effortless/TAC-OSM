# TACOSM-C5-002 — Result: INSTRUMENT_INVALID / representability gate failed

Run: GitHub Actions `36536249559` (workflow `TACOSM-C5-002 compute`)
Measurement commit: `def57ed2e044844576ba84599c56d031fa9d1fd9`
Contract fingerprint: see `contracts/TACOSM-C5-002.json`
Artifact: `TACOSM-C5-002-def57ed2e044844576ba84599c56d031fa9d1fd9`
(artifact `11018687523` — the frozen checkpoint and the run record are retained
so this verdict is auditable rather than asserted)

**STATUS: INSTRUMENT_INVALID.** No capability claim. No negative capability
claim. The pre-registered gate terminated the run *before* the task stream, so
unlike `TACOSM-C5-001` this experiment produced no capability table at all —
`"cells": []` — which is the design working rather than failing.

The successor is not yet registered. Closing the loop requires diagnosing the
instrument first; see "The route back" below.

## Why the run is instrument-invalid, in one sentence

The bridge model that reports 84.4% absolute accuracy on held-out circuits
separates the relation's satisfier from its non-satisfier in **59 of 256**
held-out pairs (0.2305) — below the pre-registered minimum of 0.5, and at or
below the ~0.25 expected of a model whose output carries no information about
the relation — so the gate refused to let the task stream run and no C5
capability number was emitted.

## The measured numbers

Only gate-level quantities exist. The run was terminated before any arm, cell,
seed or H level was measured, so there is no capability table and no seed
spread to report — which is the point of the gate's position in the run.

| quantity | value | pre-registered minimum | reading |
|---|---:|---:|---|
| gate accuracy (fraction of held-out pairs separated) | **0.2305** (59/256) | 0.5 | failed |
| gate threshold (the verifier's own `>= 0.5`) | 0.5 | — | as registered |
| bridge validation accuracy (absolute output accuracy) | 0.84375 | not a gate | **not the criterion** |
| bridge final train loss | 0.17973150312900543 | not a gate | not the criterion |
| structures executed in the task stream | 0 | — | terminated |

Gate detail, as registered: 512 gate structures (256 satisfying / violating
pairs), bridge seed `20260929`, gate seed `20260930` (a stream disjoint from
the task stream's seeds, so the gate cannot be calibrated on the evaluation
population), CASM-S commit `c315544…`.

Bridge calibration was byte-identical to `TACOSM-C5-001` by design — 512
training examples, 128 validation, 100 epochs, `lr = 2e-3`, seed `20260929`,
final train loss 0.1797, validation accuracy 0.84375. A different calibration
would move the instrument and leave the gate's verdict unattributable. The
identity is what makes the next paragraph a measurement rather than a guess.

## The finding: the instrument's accuracy score is blind to its separation score

The two numbers come from the same frozen model on the same graph family in the
same run:

```
absolute output accuracy:          0.84375   (a regression score over the output distribution)
within-query separation accuracy:  0.2305    (does the model rank a satisfier above a violator?)
```

This is the specific error `TACOSM-C5-001` made, now measured rather than
inferred. Its calibration reported 0.84375, the run completed, and the
capability table looked plausible while `verification_rate` was 0.0000 in all
75 cells. C5-002 was registered to detect exactly this, and it did: a model can
be 84% accurate in absolute terms and still rank candidates arbitrarily within
a query — here, it ranks a one-bit-flipped violator at or above the satisfier
in **197 of 256** cases.

That is the entire reason the gate tests *separation* (`hi >= threshold > lo`)
rather than absolute output level:

```python
hi = executions[2 * i].result.output      # the relation's satisfier
lo = executions[2 * i + 1].result.output  # the same descriptor with one marked bit flipped
if hi >= GATE_THRESHOLD > lo:
    n_separated += 1
```

Absolute accuracy asks "is each output near its target in isolation?"
Separation asks "does this model's output support a decision *within one
query*?" C5-001's endpoint depended on the second and measured the first. A
gate that cannot distinguish the two will always let an unrepresentative model
through, and C5-002's contribution is that the failure was caught before a
capability number existed to be misread.

## What the gate measured, and what it did not

**It measured the instrument.** The gate asks whether this frozen model, on
this graph family, under this threshold, along the exact execution path the
task stream uses, produces an output that supports the downstream decision. It
does not, so the *measurement instrument* cannot measure CASM-S execution
capability on this family. That is a finding about the bridge, the training
target, the threshold and the soft-op chain — the measurement — not about the
substrate.

**It did not measure CASM-S.** The four candidate causes of the
`verification_rate = 0.0000` result in C5-001 were left unresolved by that run,
and this gate does not separate them either:

1. **soft-op `alpha` scaling** — CASM-S's ops are `alpha`-scaled soft Boolean
   functions whose products do not reach 1 unless the whole `alpha` chain is
   1, so a perfect-match candidate can still score below the 0.5 threshold. On
   this evidence the most likely candidate, and the one a fix should address
   first, but not established here.
2. **the bridge training target** — the bridge is trained on each candidate's
   Boolean output in isolation, which is the absolute criterion. Nothing in the
   objective rewards within-query ranking, so a model trained to this target
   optimizing 0.84375 absolute accuracy is *behaving as trained*; the
   separation failure is a specification error in the target, not a learning
   failure.
3. **the 0.5 threshold** — fixed at the verifier's own decision boundary, and
   pre-registered, so it is not tunable here. But it interacts with (1): if the
   `alpha` chain cannot reach the threshold, the threshold is asking the
   substrate for an output its dynamics do not produce.
4. **unlearned gate routing** — the CASM-S internal gates are not trained on
   the relation, and their routing may be a constraint on the output the bridge
   then reads.

These four are listed in the order they should be investigated, not in the
order this run ranks them. **No run to date separates them.** A diagnosis that
does would be the genuinely new contribution; another measurement attempt
without one is the same instrument with a new experiment ID.

## The separation contract held under the failure

The endpoints were pre-registered to be independently interpretable so that an
execution failure could not contaminate an addressing number. The gate meant
the question never arose, but the design property is still what made the
outcome clean:

| quantity | endpoint | reads CASM-S output? | status |
|---|---|---|---|
| retention / coverage | `coverage_rate` | **no** — retained indices ∩ hidden acceptable set | not measured (gate failed) |
| execution correctness | `execution_accuracy_rate` | yes, via selection and verification | not measured (gate failed) |
| work scaling | `structures_executed_per_query` etc. | no — the serialized graph | not measured (gate failed) |

`TACOSM-C5-001` merged these into one `success_rate` and the merged number was
uninterpretable. C5-002 separated them, and because the gate fired first the
result is an unambiguous "the instrument cannot measure this" rather than
another ambiguous table.

## What this result does and does not license

It licenses:

- a statement that the C5-002 instrument, as pre-registered, is **invalid for
  measuring CASM-S execution capability on this graph family**, evidenced by a
  gate that failed at 0.2305 against a 0.5 minimum;
- the *methodological* claim that the gate belongs before the task stream and
  not after it — this run is the direct evidence, and it is the difference
  between a terminated run and another void;
- the observation that bridge calibration accuracy and within-query separation
  are distinct quantities that can move independently, measured here on one
  model at 0.84375 vs 0.2305;
- continued use of the C5-001 work accounting, which does not depend on the
  output being semantically correct (see `docs/TACOSM-C5-001-RESULT.md`).

It does **not** license:

- **any capability claim.** No C5 capability number exists. `C5` in
  `docs/CLAIMS.md` remains `UNTESTED`.
- **any negative capability claim.** This is not a finding that CASM-S cannot
  execute the relevance circuit, that selective execution cannot work, or that
  addressing cannot scale. The four candidate causes above are causes of the
  *measurement*, not of the substrate, and none is separated by this run.
- **a threshold change.** `GATE_MIN_ACCURACY = 0.5` is pre-registered and was
  fixed before the run. Relaxing it after seeing 0.2305 is exactly the
  post-result threshold adjustment the gate exists to prevent, and would
  convert the gate from a termination condition into a tuning knob. The
  threshold stays at 0.5 for the successor, whatever that successor is.
- **an amendment to `contracts/TACOSM-C5-002.json`.** A pre-registration
  revised to fit a result is not a pre-registration. The contract stands as
  registered and this document is its outcome.

## The route back

The user's escalation chain, recorded so it is not re-derived:

```
gate failure
  -> instrument diagnosis (separate experiment, not a C5 re-run)
       resolve which of the four candidate causes is binding
       the leading candidate is the alpha-scaling / threshold interaction
  -> CASM-S bridge experiment
       a bridge whose training target is within-query separation, not absolute
       output — i.e. train on pairs, not on single candidates
  -> new frozen checkpoint
  -> new pre-registration (a NEW contract, not an amended TACOSM-C5-002.json)
  -> C5-002 successor run
```

The two solid design decisions in this experiment carry forward unchanged and
should not be re-litigated:

1. **The gate runs before the task stream.** Its position is the reason this
   outcome is a clean termination rather than a third voided run.
2. **The separation contract.** `coverage_rate` reads no CASM-S output, so an
   execution failure cannot depress an addressing number. The gate meant this
   was not exercised, but the property is what allows the successor to be
   confident that a *passing* gate implies an interpretable coverage
   measurement.

The failure also makes one requirement on the successor explicit: the bridge
must be trained on **pairs** — one satisfier and one violator against one
reference — with a loss on their separation, rather than on single candidates
with a loss on each output's absolute level. The current target cannot express
the ranking the downstream decision depends on, and no amount of epochs on it
would have fixed 0.2305. That is the specific, falsifiable design change this
result licenses.

## Status of `TACOSM-C5-001`

Unchanged, and permanently void. See `docs/TACOSM-C5-001-RESULT.md`. Its 128×
reduction in submitted CASM-S work remains an instrument-level observation
about the accounting, citable without being promotable, and is **not** C5
progress. This run does not revisit it and does not affect it.

## Reproduction

The artifact retains the frozen checkpoint (`casm-s-c5-002.pt`, sha256
`4343c870d4e00b0913553e95b046765d294e948ccc4998d1bdd15cb93d00ce99`) and the
run record. The gate is reproducible from the registered contract:

```
python scripts/measure_c5_casm_002.py \
  --casm-root /path/to/cdl-attention-experiment \
  --train-bridge \
  --checkpoint-out /path/casm-s-c5-002.pt
```

The script terminates as `INSTRUMENT_INVALID` and writes its summary before any
arm is executed. A `--smoke` run declares its deviation and is not a
measurement.

The successor is not a re-run of this script. It requires a new bridge, a new
checkpoint and a new pre-registration.
