# SEQ-TRANSFER-008 — randomized update validation results

**Status: COMPLETE EXPLORATORY VALIDATION PILOT, NOT held-out qualification.**
Original successful GitHub Actions preflight and six-fit training:
https://github.com/FPC-effortless/TAC-OSM/actions/runs/39309620973

Original pre-result training source HEAD:
7ffa779a2c0f2e853d84ca62a3a97f59655434f8

The first workflow run (#1) was cancelled deliberately after discovering
that full-repository checkout was unnecessary; fixed sparse-checkout
workflow run #2 passed both jobs. The source/generator was committed
before the six full validation fits. No held-out SEQ008 TEST examples
were constructed, used for validation selection or trained upon.
Original checkpoint/artifact data remain preserved.

## Fitted controls and results

Three independent scratch neural architectures, trained for 400
AdamW updates per seed using **final binary outcome labels only**,
on 3000 L32/L64 public-entity examples. The two fresh fixed fitting
seeds were 780820 and 780821. Identical 600-example held-in
validation samples (200 balanced examples/task) were evaluated at
L64/E4 and L128/E4, with seed 780802 for L64 and 780812 for L128.
These are two trained model repetitions on the **same validation worlds**,
not independent test datasets.

| Arm | Seed | L128 late overwrite | L128 repeated toggle | L128 noncancelling XOR | L64 across all three |
|---|---:|---:|---:|---:|---|
| TAC-inspired gated recurrence (4750 parameters) | 780820 | 100% | 100% | 100% | 100% |
| TAC-inspired gated recurrence (4750 parameters) | 780821 | 100% | 100% | 100% | 100% |
| Entity-routed GRU (6706 parameters) | 780820 | 100% | 100% | 100% | 100% |
| Entity-routed GRU (6706 parameters) | 780821 | 100% | 100% | 100% | 100% |
| Global GRU (10930 parameters) | 780820 | 49.5% | 50.5% | 47.0% | 50.17% |
| Global GRU (10930 parameters) | 780821 | 53.5% | 50.0% | 51.5% | 46.67% |

Mean global GRU L128 across tasks and training seeds: 50.33%.
Mean global GRU L64: 48.42%. Both entity-local arms reached
100% in all six task/length cells at both seeds, including L128.
Only two optimizer seeds and public entity IDs were tested.

## Evidence preserved

- Preflight compilation, deterministic source generation and label
  replay, sealed future TEST seeds, randomized last-target write
  positions, causal counterfactual mutation check, noncancelling
  two-entity XOR contract: PASSED.
- Single-seed two-update smoke: PASSED; this was NOT measured as an
  accuracy experiment.
- Six full pilots and checkpoint/optimizer upload: PASSED.
- Original artifact: seq008-original-pilot-evidence,
  GitHub artifact ID 12215367412 (six .pt checkpoints + pilot.json).
- Separate preflight artifact: seq008-preflight, ID 12214448339.
- Source reports per-seed final losses, checkpoint SHA-256 values,
  per-cell accuracies, model parameter counts and fit host wall time.
  These are original training records; an *independent* restore/replay
  across every checkpoint has NOT yet been carried out.
- Original SOURCE/TEST boundary: no SEQ008 TEST source callable, no
  test evaluations, and no previous reserved SEQ006C or
  REAL-MAINT/DEV068 artifacts were changed.

Original fit SHA-256 digests:

| Arm / optimizer seed | Checkpoint SHA-256 |
|---|---|
| TAC / 780820 | 047d29a2506339da2774db28c3437f2973922c405685c45b8a9f36af1675c831 |
| Routed GRU / 780820 | 03728c8dd9dc6a394c362cd1ce5b682d475557c30a55f545bf2dedb590f18d0a |
| Global GRU / 780820 | 7047c3b99cefd3dc232a509259e2c0892ab5c3a8ae225b7b5dba134ad058f38c |
| TAC / 780821 | d4b6e9ea3238bd725e3e1ca300df6d687a658539b07dc0f118d892fe81582ece |
| Routed GRU / 780821 | 1000abeb62cfe511f3808d20ebd87043369e7d4f3d0ef7ae28483d5a15ea4951 |
| Global GRU / 780821 | 73b714bdd33a93a07f6dd33e787e36edea4f6a03b783f4bca3cd2f64c7f49549 |

## Interpretation and failures of attribution

*Randomizing update positions, delaying the last relevant update and
making pairwise XOR toggles noncancelling did not damage validation
accuracy for either entity-local learner.* The published source
generator still uses task-specific templates and **public true entity
IDs**, an exact four-operation vocabulary, supervised final labels and
a synthetic deterministic oracle. Informative positions are random
within an explicitly bounded prefix, not an adversarial or fully
unpredictable world.

The simpler independently trained routed GRU exactly matched the
TAC-inspired gated recurrence yet again. Nothing here attributes
performance uniquely to TAC, neural CASM, CDL, BDH, Mamba,
executable memory, continual plasticity, or a learned verifier.
Cross-model parameter counts and learning/optimization baselines
are not yet fully matched. The globally recurrent control failing
does not establish general architectural state of the art.

This study also does not verify actual cost-sensitive action or
inference-time learned belief under incomplete and noisy observations.
As a result, **no agent, model family, implementation or commercial
capability is promoted**.

## Next preregistered direction

The proposed threshold for advancing the research question
(both entity-local models reaching >=95% per L128 task in both seeds)
was met on validation. The next unexposed experiment should target
**identity alias switches, role reassignment and incomplete observations**,
with trainable binding and strong controls:

- Oracle true-identity routing as an upper bound, not counted as learned.
- Surface-ID GRU which cannot silently access stable hidden identity.
- Trainable role–filler association/memory, plus routed GRU/global GRU.
- Mechanism-specific identity-swap, noisy-receipt, missing-receipt,
  and noncancelling relation probes.
- Bayes/observation-only information ceiling in partially observed
  settings, so irreducible uncertainty is not mistaken for failure.
- A fresh *separate* >=20-seed held-out reliability protocol only after
  exploratory validation and frozen source/controls.

Future executable-memory, sparse plastic SSM/BDH and counterfactual
verification studies should be compared to this simple routed baseline,
not assumed beneficial.
