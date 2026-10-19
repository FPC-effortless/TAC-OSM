# DEV065-SEMANTIC-STABILITY — preregistered gradient/exposure factorial

Status at registration: UNFITTED. Parent: DEV063, closed NO PROMOTION.
This is the training-stability lane proposed in the portfolio; DEV064's separate
chained-task CASM deadline is unchanged. No CASM novelty or commercial claim.

## Question and controlled changes

Does removing decision-loss gradients from a learned semantic posterior improve
calibration, and does exposing the actor to exact training semantics prevent an
encoder/actor private code? DEV063's curve deterioration and 35.3% exact-belief
action changes motivate this question; neither establishes causality.

Four TAC arms cross coupled/isolated decision gradients with predicted-only/
mixed actor inputs. Mixed means a shared deterministic Bernoulli(.5) row mask
selects exact PUBLIC posterior targets during training. Isolation detaches actor
inputs from the encoder; proper posterior KL remains trainable at every update.
No arm freezes proper belief learning. All arms have the same semantic-only actor,
initializations, batches, update counts, targets, EMA and validation search budget.
Retrained parameter-matched GRU and no-CASM use the isolated/mixed recipe.
Primary arm: `tac_isolated_mixed`, named before any fitting.

Unlike DEV063, training includes both public sensor-error laws .1 and .2 for ALL
arms. Thus historical cross-study improvement cannot be attributed solely to
gradient isolation. Within-study factorial effects isolate the two interventions.
Supervision is explicitly posterior imitation, not autonomous physics discovery.

## Data, seeds and exposure

Machine-readable constants: protocol.json. Twenty fresh initialization seeds
96501–96520; pilot seed 96500, LR {.001,.003}, all six arms, all retained.
Fresh common train 2048 snapshots namespace 6651001; validation512 6651002.
Training histories32/128, missing50% initial readings, sensor laws .1/.2,
inspection prices .05/.1/.2. Teacher observes only public history and uses the
known-law depth-two policy with a certainty STOP guard. Teacher never exported.

Each model gets2048 updates, first1024 belief-only, AdamW1e-4 decay, clip1,
posterior KL weight4, policy1, futureKL.25, costHuber.1, EMA.995. DAgger at1280,
1536,1792, horizons2/5/10,64 rollouts per arm/round; union shared across all six
arms, half base/half aggregated minibatches. DAgger histories32, error.1; mixed
sensor-law examples remain in the base corpus. Validation ranks LR by mean KL,
then policy NLL, then LR, without test access. Deployment is final EMA, not
best-on-test. Temperature {.5,1,2,4} chosen on validation only; it does not alter
argmax. No dropping failed seeds, extra fitting, adaptive stopping or test tuning.

Main fits/pilots freeze before test generation. Fixed test256 namespace6651005
contains both sensor laws. Primary64 physical starts6651003 reused at histories
32/128/512/1024. Secondary64 fresh starts6651004 at(error.2,inspection.1) and
(.1,.4), H32. Shared worlds across arms/seeds are paired replications, not
independent physical trials. Fresh episode identities are random16-vectors.
Smoke adds10000000 to data/initialization namespaces, 8updates,2worlds; excluded.

Budget: fit including corpus creation, pilots and DAgger ≤7200 CPU seconds,
420 CPU seconds per six-arm seed group, two single-thread CPU workers. All
groups must complete; a cap/nonfinite failure is preserved and disqualifies
promotion, never silently discarded. Test evaluation/replay CPU accounted
separately. No paid compute. No opening the real forecast's July B holdout.

## Registered decision gates

All are conjunctive for primary-recipe promotion:

1. At least18/20 seeds reach ≥95% completion in EACH primary history stratum.
2. Zero excess final physical-fault bits against the public reference per world,
   across ALL20 seeds, both primary and EACH secondary condition. Also ≥18/20
   seeds ≥95% completion in each secondary condition. Zero faults does not mean
   zero transient harm; report collateral transitions and healthy-target repairs.
3. Median across seeds of fixed-query p95KL ≤.05 and p99KL ≤.1 (unchanged from
   DEV063). Also report each seed, p95/p99 on rollouts, Brier, TV, calibration,
   and fraction of individual seeds passing both fixed tail thresholds.
4. Primary minus GRU AND minus no-CASM mean cost, paired-seed bootstrap95 upper
   bound ≤+.02 synthetic units/world (registered noninferiority tolerance).
   This is recipe competitiveness, NOT superiority or equivalence. Report exact
   sign-flip p and Holm over those two comparisons, without promoting a p>.05
   as equivalence. At least10% median p99 improvement vs coupled/predicted with
   its paired bootstrap upper difference <0 is required for mechanism gain.
5. Full saved-checkpoint reconstruction, exact action matching, semantic-only
   deployment and relative event-clock invariants. Float64/thread1, numeric
   tolerance1e-8, deterministic epsilon1e-6 choice rule; remote remains UNKNOWN
   until executed. No universal hardware-determinism claim.

Primary two-factor effects on per-seed fixed p99KL and cost are paired contrasts:
isolation average versus coupling, mixed average versus predicted, and interaction.
Holm correction over these six contrasts. Report intervals, per-seed paired CSV,
learning curves, costs/actions/faults/stopping errors, params and training CPU.
No equivalence claim from an inconclusive contrast; no CASM necessity claim from
this saturated two-entity task. No outcome-based selection of another arm.

## Closed-loop semantic interventions (evaluation only)

Fresh namespace6651006,64 starts at H128/H1024 and errors.1/.2, price.1. Execute
predicted, exact public, and blurred(.5q+.5/12) actor beliefs in independent
reconstructions for every main checkpoint. Each mode maintains its own public
posterior along its OWN visited history. Shared seed/step/action randomization
does not make diverging trajectories identical counterfactual physical draws.
Report cost, completion, faults, action counts and paired-seed exact-minus-
predicted regret. Raw action-flip rate is not a target: different actions can be
equally useful. Exact substitution can expose semantic misalignment but cannot
establish universal causality or commercial reliability. No deployed oracle,
analytical action selector, STOP override or future outcome enters the model.
Budget1200 CPU seconds; preserve partial results on cap, no promotion without
completed audit. Additional same-history shuffle/clock probes may be reported
as diagnostics, not used to select weights or change the registered gates.

## Supplied structure, artifacts and limits

Keep DEV058–DEV063 exports/evidence untouched. New source/modules, strict hashes,
all raw/final EMA weights, DAgger data, pilots, reports and tests remain available.
Inference is Torch-only and public-input-only. Fixed two-entity manifest, cosine
identity binding, typed trained admission, FIFO64,12-state health×graph schema,
dense three-MLP CASM and time-independent synthetic laws remain explicit priors.
No scalable indexed retrieval, general symbolic discovery, physical memory-traffic
saving, actual maintenance economics or autonomous belief discovery is claimed.
The root commercial model goal stays ACTIVE even if this registration passes.
