# DEV064-CHAINED-CASM — independent attribution pre-registration

**Status:** REGISTERED DESIGN, not fitted and not promoted. This branch is based on the
DEV067 research head and does not modify DEV068, REAL-MAINT-004, past weights or their
training/test data. Date: 2026-10-19. Benchmark implementation may be revised during
smoke/preflight only with versioned changes made *before test exposure*. Retain every
failed attempt and protocol amendment; amendments require a new experiment identity
when they change the scientific decision.

## Research question and boundary

Does learned **multi-step compositional action evaluation** confer lower verified
decision cost than independently retrained no-CASM, GRU, MLP and attention
controllers under the *same* fully observed public information and supervised
known-law action/outcome targets? This is a synthetic **mechanism** experiment, not
posterior-free world learning, industrial competence, or a real-repair benchmark.

Do not reuse DEV067's saturated two-entity task as evidence of composition.
The current TAC agent uses dense learned MLP operator mixtures; do not claim
general symbolic CASM or programmed-optimal planning in the deployed model.

## Benchmark and information contract

Implement an independently reproducible, fully observable discrete maintenance
world with 3 causally linked components in randomized role/order; repair action
outcomes depend on parent conditions; healthy-target interventions can create
secondary damage; inspection/repair costs and stochastic effects are explicit.
A STOP action terminates with a terminal cost for residual faults, and action
budgets constrain chaining. Observations to ALL neural arms contain exactly
the same current statuses, relation graph, publicly known action costs/rates,
and remaining action budget. No oracle action or future effect enters a model
at inference. Exact finite-state dynamic programming is EVALUATION / training-
teacher only, never an exported learned-policy fallback.

Before training, verify nontrivial exact-public-oracle headroom over one-step
greedy on a **separate validation generator**. The precommitted minimum is
mean exact-oracle regret of the one-step greedy choice >=0.05 cost units AND
at least 5% of validation situations with strictly positive regret (>1e-6).
If absent, record NO_HEADROOM and do not fit pilots or interpret completion
parity as CASM superiority. This is a *necessary*, not sufficient, task
discrimination requirement. Record the exact gap and fixed data hashes.

The initial runnable implementation learns oracle action-Q values, not a
separately supervised transition distribution. Consequently, that early
value-regression preflight cannot establish autonomous consequence-model
learning or full DEV064 architectural attribution. It must pass its own
stronger closed-loop and consequence-prediction gates before any promotion.
The five model arms are automatically enforced to be within 10% of one
another in trainable parameter count.

## Comparators and training equality

Arm 1: learnable multi-step compositional TAC operator, with *trained* shared
transition/outcome operators and decision head, no oracle/planning rule in inference.
Arm 2: independently retrained no-CASM network, not a severed trained TAC.
Arm 3: independently retrained GRU recurrent controller.
Arm 4: independently retrained attention controller.
Arm 5: tuned dense MLP baseline, with learned parameters and matching inputs.

Require within 10% active trainable parameters where practical, or disclose exact
mismatches and prohibit parameter-equivalent claims. Keep identical training
examples, optimizer/update budget, loss terms, validation-selection policy and
test starts. Predict legal candidate-action costs and consequences; optimize
supervised action-value regression plus cross-entropy to public oracle policy.
No hidden physical-state labels beyond fully public states. No trained policy
may call the exact dynamic-programming evaluator at inference.

## Data selection, sealing and budgets

Do NOT import DEV068/REAL004 or their holdouts. Source benchmark seeds are:
TRAIN 640641, VALIDATION 640642, TEST 640643. Protocol pilot
init seeds 640650 and 640651 are validation-only; confirmatory init seeds
640701..640720 (20 per arm). Generate training 4,096 situations, validation
1,024, test 2,048, each from disjoint deterministic generator namespaces.
The TEST generator may not be invoked until training, validation-only
hyperparameter selection, *all* twenty-seed fits and a permanent exposure
marker are complete. Use 1,024 optimizer steps/arm/seed, AdamW, batch 128,
weight decay 1e-4, learning rate 0.001 (no test-based retuning).
Preserve the first seed, not the best test seed, as portable example.

Pilot proceeds only to full matrix if implementation correctness, exact-DP
reference consistency, oracle headroom, training fit finiteness and independent
validation replay pass. If compute is unavailable, mark INCOMPLETE; no imagined
scores or declared failures from missing runs. An implementation smoke test is
not a confirmatory pilot.

## Exact evaluation and decision

At fresh test, compare oracle regret, terminal residual-fault rate,
unnecessary interventions, newly created faults, action cost, completion/STOP
and end-to-end inference CPU for each arm on identical paired worlds. The
same physical world RNG streams must be used for each arm via common random
numbers, without allowing any model to inspect outcome seeds.

Primary claim requires BOTH:
1. No per-world increase of residual faults against the best retrained control
   (or against registered safety/reference guard), and at least 18 of 20 seeds
   reaching >=95% completion on each registered stratum.
2. Paired-seed bootstrap 95% upper bound on total decision cost for CASM minus
   each trained GRU, MLP, no-CASM and attention control is < 0 (strict
   superiority), together with a precommitted practical reduction >= 0.02
   cost units/world and no >10% inference-cost regression over the best
   parameter-matched practical control.

If the exact-public reference headroom is absent or outcomes are saturated,
report inconclusive. If CASM fails after the complete budget, close
NO_SUPERIORITY; narrow/retire its superiority claim. No adaptive seeds,
stop thresholds, altered prices, post-test selection, or merging on an
incomplete result. Report uncertainty with shared-world clustering; 20
model seeds on common test worlds are not independent factories.

## Reproducibility and publication

Commit the runnable benchmark, protocol, source checks, tests, per-seed losses
and raw outcomes, negative cells, environment versions, and checkpoint hashes
to GitHub; keep the research PR DRAFT and unmerged until independent replay
passes. Public known-law teacher and exact planning reference must be
labeled as scaffolding, not autonomous discovery. Existing DEV067/REAL003
claims remain unchanged.
