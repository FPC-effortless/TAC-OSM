# ME-056 boundary-consolidated TAC execution result

Status: **architecture implemented; real comparative training remains blocked**.

ME-056 changes the organization of TAC computation rather than relaxing the
ME-055 resource gate. Fast evidence state updates every frame. Slow state and
the relevance trace consolidate at fixed 128-frame observed boundaries. The
released training compute is used by a dense physical-transition objective on
an outcome-independent half of the 256 authentic FIT frames.

This is a substantive architecture/training change:

- persistent world-state mutation is separated from prediction-time work;
- slow memory is updated at explicit anchors rather than every frame;
- 128 physical transitions provide direct dense supervision per optimizer
  update, compared with only the 16 open-loop targets in the primary loss;
- the 16-step bounded action-conditioned rollout and factual intervention
  censoring remain intact;
- all 38,266 native parameters remain active.

## Strict resource result

| Measure | ME-056 TAC | GRU control | Relative gap | Gate |
|---|---:|---:|---:|---:|
| Trainable parameters | 38,266 | 38,295 | 0.076% | pass |
| Profiled FLOPs/update | 42,257,505 | 42,546,619 | 0.680% | pass |
| Median paired CPU/update | 0.038523 s | 0.035837 s | 8.007% | **fail** |

The CPU estimate uses 41 paired measurements, alternating execution order, one
pinned CPU, one Torch thread, five warmups per arm and `process_time_ns`. The
paired median bootstrap interval is **6.353% to 9.999%**, entirely above the 5%
limit. Training therefore remains unauthorized.

ME-056 reduced the robust gap from ME-055's remeasured 16.17% (95% interval
15.08–17.33%) to 8.01%. It did not clear the registered gate. Batch 8 worsened
the gap to 10.20%, and prototype tensor scan worsened the original layout to
45.45%. Whole-history compilation was stopped after more than 90 CPU seconds
without a first result. These execution variants are rejected.

## Corrections and scope

The earlier claim that the shared 16-step Python rollout caused TAC's relative
gap was unsupported: GRU executes the same loop. ME-055 also uses an 8D trace,
not an 8x8 matrix. The paired evidence instead shows a persistent cost from the
native multi-state transition and its small sequential operators.

Both architectures receive identical normalized physical state, action,
control-authority, servo-prior parameters and causal histories. TAC's episode
initializer processes the same first observation that GRU receives in its
first recurrent observation; it does not add source information.

The evaluation estimand remains factual open-loop prediction before takeover.
Post-takeover physical steps are censored because logged human actions are not
an autonomous counterfactual plan. No policy value or control-capability claim
is permitted; IPW would be required only for a separately registered policy
estimand with defensible action propensities and overlap.

The block split groups four adjacent episode IDs before assigning blocks by a
revision-pinned hash. Publisher metadata does not expose operator/session IDs,
so independence beyond four-episode blocks is unverified and will remain a
limitation of any future result. TEST remains unopened.

## Test-count reconciliation

ME-049's 1,062 count came from the independent
`research/me049-supported-action-state-2026-10-21` branch. That branch is not
an ancestor of the ME-052–ME-056 robot chain. ME-055 reported 951 tests on its
own stacked branch. The difference is branch divergence, not removal of 111
tests from a common lineage.

## Variant ledger

The reserved task has not been fitted by any variant. Execution-only candidates
inspected on the single registered FIT profiler episode are:

1. ME-055 fused relevance trace: blocked at 13.18% point-estimate overhead.
2. ME-055 paired remeasurement: blocked at 16.17% median overhead.
3. Prototype scan: rejected at 45.45% overhead.
4. Prediction-refinement boundary layout: rejected at 22.76% overhead.
5. Boundary plus dense supervision, batch 8: rejected at 10.20% overhead.
6. Final ME-056, batch 2: blocked at 8.01% overhead.

No DEV or TEST outcomes were used in these choices. Any future candidate gets a
new hash, repeats strict preflight, and enters this ledger before training.

