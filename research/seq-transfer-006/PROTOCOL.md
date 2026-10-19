# SEQ-TRANSFER-006 — learning updates to persistent entity state

**Status: registered preflight, not fitted or promoted.**
**Date:** 2026-10-19. Based on DEV067 source, independent of the
SEQ-TRANSFER-005 report branch and independent of DEV068/REAL-MAINT-004.
No historical checkpoint is overwritten or claimed to be used by this study.

## Question
Can a recurrent neural model learn correct **overwrites, repeated toggles,
and two-entity parity** across histories containing hundreds of intervening
events? SEQ-TRANSFER-005 showed stable-fact retention but ~chance toggles
and weak late overwrites at 512 events. This study isolates acquisition of
update rules versus retention and relation composition. It is **not**
an experiment on the frozen or fine-tuned DEV061.5 checkpoint; its
TAC-inspired variant is scratch-trained and cannot be called original TAC
architecture superiority.

## Dataset and information
Each world supplies time-ordered observable (public entity identifier,
SET-0/SET-1/TOGGLE/NOOP) records; final query addresses one or two entity
identifiers, provided *only after* the stream. Labels are exact final
entity bit or pair XOR, computed from all events by an independent
deterministic replay. Episode identifiers are randomly permuted.
No future label, hidden result, future event or queried identity is in
the event encoder. Training optionally uses exact intermediate states as
supervision; name this **symbolic-teacher learning** and do not claim
posterior-free or autonomous state discovery.

Task families: late_overwrite, repeated_toggle, pair_xor, with held-out
labels balanced within each family; matched random irrelevant-entity
distractors. All informative queried writes end well before the
sequence end. Control for accidental correlations with label-shuffle
and iid-unpredictable negative checks.

## Locked namespaces and budget
- TRAIN: 690601, 3,000 examples/task mixture, lengths 32/64, 4 entities.
- VALIDATION: 690602 for L64/E4 and 690612 for L128/E4, 600 examples each.
  Both validation namespaces are disjoint from reserved TEST 690603; the same
  registered examples are used across all model arms. Earlier preflight CI
  was blocked by the test-seed guard before evaluation; no test material was opened.
- RESERVED TEST: 690603, 600 each at L256/E4, L512/E4 and L512/E6.
  **The initial script must not contain a way to open TEST**. No test
  generation until pilots, preregistration and full-seed training complete.
- Validation pilot seeds: 690610 and 690611, 400 updates/model,
  batch 48, AdamW LR .003, clipping at norm 1.
- Candidate controls: scratch TAC-inspired entity-local gated recurrence,
  scratch entity-routed GRU, and global recurrent GRU;
  plus exact deterministic state machine as a non-neural reference.
  Include trainable parameter counts and measured compute, not claims of
  parameter equality. Distinct core architectures and role priors are disclosed.
- Frozen label/readout convention and loss:
  final binary cross-entropy weight 1 plus intermediate entity-state
  supervision weight 0.25, across eight uniformly selected checkpoints.
  No teacher state at inference. Model trains jointly across all three tasks.
- Track per-task initial and validation accuracies, train/validation loss,
  parameter count, fact-age statistics and per-seed failures. No cherry-picking.

## Pilot entry gate and stopping
Before any confirmatory TEST exposure:
1. Exact replay of input event streams reproduces all deterministic states,
   label balance is adequate, no test seed accessed, and all weights reload.
2. At least one **learned** model must reach >=85% accuracy at L128/E4
   separately on late_overwrite and repeated_toggle, and >=70% on pair_xor,
   for both pilot seeds. A different study with new design is required if
   this fails; do not tune on the reserved TEST.
3. Full promotion experiments require >=20 fresh training seeds per model
   and a new source-frozen evaluation protocol. Pilot results cannot be
   called confirmed length generalization or architectural superiority.

## Causal interpretation
A scripted entity-local state machine is an upper reference, not a learned
architecture. Public identity routing is a programmatic prior and not entity
discovery. A successful entity-routed GRU without CASM undercuts claims that
specialized TAC processing is essential. Even if the TAC-inspired model wins,
this is a synthetic, supervised, fixed-vocabulary setting, not industrial
cost advantage or a general world model. Keep negative results and audit
failures; do not merge without independent replay and scientific decision.

## Reproduce
`python -m unittest discover -s tests -p test_seq_transfer_006.py -v`
`python scripts/run_seq_transfer_006.py --stage smoke --output results/seq006`
`python scripts/run_seq_transfer_006.py --stage pilot --output results/seq006`

No unattended CI repetitions of completed pilots on every PR synchronization.
