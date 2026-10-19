# SEQ-TRANSFER-008 — randomized causal state updates, validation-only

**Registration date:** 2026-10-19. **Status:** source committed BEFORE SEQ008 validation fits; no SEQ008 held-out TEST is exposed.

**Parent:** SEQ006C draft PR #233, including original SEQ006 task runner.
SEQ007 draft PR #234 motivates final-label-only objectives but its source is neither overwritten nor required.
This study does not reopen the reserved SEQ006C TEST, real maintenance holdouts, DEV068, or original checkpoints.

## Research question

SEQ006C achieved 100% on late overwrite, toggle, and pair-XOR across 20 seeds
and L512/E6 for both TAC-inspired entity-local recurrence and simple
entity-routed GRU. The source generator used public IDs, constrained informative-event
locations, and cancelling XOR toggles. Can the result survive randomized informative
event times, *noncancelling* joint changes, and long distractor tails when both
architectures train ONLY from final-outcome labels?

This is a falsifiable stress test of entity-local updating, NOT a test
of discovered entities, partial observability, Hebbian plasticity,
CASM attribution, or industrial prediction.

## Registered event semantics

Events are public (entity_id, operation) with SET0, SET1, TOGGLE, NOOP.
Every sample has a randomized query entity (or query pair), initial
state zero, and a new randomized cutoff between 50% and 80% of the
sequence. All causally relevant query-entity events occur at randomly
selected, nonfixed positions before that cutoff; the remaining stream
is independent distractor traffic.

- Late overwrite: 3–7 query updates with initially contradictory SET values,
  followed by randomly chosen SET/TOGGLE events.
- Repeated toggle: a random initial SET and 3–7 toggles on that entity.
- Pair-XOR: two independent initial SET values and two changing entities;
  their number of toggles has opposite parity, preventing joint cancellation.

Labels are derived by replay of actual ordered operations. The generator
draws candidate sequences BEFORE knowing their label, then selects equal
numbers per task and label. No desired answer is inserted into a positional
operation. Record total attempted sequences; selection-conditioned
correlations remain possible and should be probed subsequently.
Intermediate reference states exist ONLY for generator/audit assertions;
they are never supplied to the training loss or model at inference.

## Model and training contract

Use existing scratch StreamingAgent arms from SEQ006:
tac_gated, routed_gru, global_gru. All fit independently from scratch;
original DEV061.5/DEV062 checkpoints are never used. Architectures
are not parameter-equal, so cross-arm superiority is not attributable
uniquely to TAC.

- Train seed 780801: 3,000 examples, L32/L64, 4 public entities.
- Validation seed 780802: 600 examples, L64/E4; seed 780812:
  600 examples, L128/E4.
- Equal 100 examples in every task × binary-answer cell.
- Model seeds 780820 and 780821: 400 AdamW updates/arm,
  batch 48, LR .003, grad norm 1, weight decay 1e-4.
- One end-state binary BCE supervision term; NO intermediate teacher.
- Six fits retained with optimizer, checkpoint hash, measured validation
  strata, host wall times and loss curves; no best-seed choice.
- CPU deterministic algorithms, single thread. Smoke is not a result.
- No pretrained representations, CASM, verifier, SSM, executable memory
  library or continuous plasticity in this experiment.

## Source and exposure boundaries

Reserved future TEST namespaces 780803, 780813, 780823 are listed but
deliberately NOT callable. This pilot has no TEST generator or promotion
mechanism.

Preflight must verify:
1. Every label equals independent replay of all event operations.
2. Input-label balance and exact train/validation determinism.
3. A last relevant update causally changes the final label.
4. Randomized timing and noncancelling two-entity XOR.
5. TEST sources inaccessible and no intermediate teacher consumed.

Only source integrity and paired validation accuracy are reported here.
Even if every pilot reaches 100%, results remain exploratory.

## Prospective decision rule

If BOTH entity-local arms score at least 95% on each L128 task in
both fixed seeds, treat this as encouraging validation only; separately
preregister new held-out TESTs, 20 fresh seeds, a trainable alias-binding
comparator, and explicit representation and role baselines. If an arm is
weaker, preserve that result and diagnose failed operations/time lags.

DO NOT promote a model based on this exploratory protocol.
Commercial cost, hidden-state forecasting and irreversible action safety
remain out of scope.

## SEQ009 direction (not claims from SEQ008)

Progress to aliases and switches, nonpublic identities, noisy/incomplete
receipts, learned role–filler binding and executable update operators,
each compared to entity-routed GRU before experimenting with plastic SSM
or BDH-inspired dynamic memory and counterfactual verification.
Partially observed conditions require an independently measured
information-theoretic ceiling.
