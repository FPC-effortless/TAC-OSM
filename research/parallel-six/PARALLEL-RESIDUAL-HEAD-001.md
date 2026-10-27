# PAR-FFN-001 — parallel TAC post-scan residual head (NO FIT)

**Status: experimental neural source + constructed tests; not trained, not real-world qualified, not merged.**

## Scope

Reuses the existing v1.7 ParallelTACWorld and WorldStreamState from the six-family
research branch. All scan gates are computed before the ordered affine prefix
scan, independent of the old matrix read. The optional FFN runs AFTER the scan
and does not feed back into memory writes. Observed actions are observational
conditioning, not counterfactual interventions or executable controls.

The model is in src/tac_osm/variants/parallel_residual_world.py and exposed
through instantiate_untrained_postscan_core in parallel_six.py.

For source-legal observation x_t and recorded action a_t, with previous
associative read r_t = M_(t-1) k_t:

    z_t = concat(f_t, s_t, r_t, tanh(encoder([x_t, a_t])))
    u_t = tanh(W2 * SiLU(W1 * z_t + b1) + b2)
    gate = sigmoid(learned_residual_logit)
    forecast_t = base_forecast_t + TRAIN_delta_scale * gate * u_t

At the default state/action widths 14/14, hidden state width 88, baseline
decoder width 132 and FFN width 12, there are 96,303 active weights:
93,324 original parallel model + 2,978 FFN + 1 learned gate.

Streaming state is UNCHANGED: 88 running input summary values + 8x8
associative memory = 152 float32 elements (608 nominal bytes), plus an
integer count. Model parameters, runtime allocations, source/entity metadata
and any later caches are excluded from this state number.

The correction is bounded per coordinate by abs(TRAIN_delta_scale) in exact
arithmetic because sigmoid gate lies in [0,1] and tanh lies in [-1,1].
This does NOT prove accuracy, useful associative memory, gradient stability,
calibration or a whole-model formal/IEEE/AD correspondence.

## Mandatory no-fit gates

A new CPU-only test checks all active parameter counts, FFN gradients,
parallel-batch versus stream outputs and memory, causal prefix invariance,
stream input nonmutation, and gate-zero parent-output behavior.
That test is a constructed fixture, NEVER real-world performance evidence.
Do not assert it passed before the actual GitHub workflow result.

## Registered next genuine-source study: PROPOSAL, NOT EXECUTED

Main candidate: 96,303 active parameters. An unmodified parallel TAC
decoder=144 comparator would have 96,288 active parameters (0.0156% difference),
but must also meet complete forward/backward/optimizer FLOP budget parity
within 1% at the exact registered shapes; parameter parity alone is not enough.

Use independently source-verified real temporal sensor or physical robot
observations, legal source-time and entity IDs, identical chronological
train exposure, optimizer budgets, preprocessing, >=5 seeds, preregistered
independent withheld cohort, and full checkpoint+optimizer+RNG+sample-order
cold replay. Include GRU/SSM/attention where task-appropriate plus genuine
strong ridge/persistence/constant-velocity controls. CPU time is an outcome,
not a forced equality.

Test original forecasts plus true memory-read incremental value, no-read
ablation as a DIAGNOSTIC rather than an equal-param training comparator, and
point+uncertainty metrics. Never promote exposed ALOHA/UCI501 development data
to new independent TEST. No data, fitting, GPU or protected outcome access
is performed by this PR.

## Linked mathematical obligations

PNDS counterpart: PNDS/ParallelResidualHead.lean and
foundation/v1/parallel_residual_head_math.py (exact-real scalar bounded
post-scan head and independent reference, NOT full neural tensor/FP/AD proof).
Six existing families, 17 historical source records, negative results,
original checkpoints and parent graph remain unchanged. CED is an optional
future POST-SCAN study, not an included untrained promise.
