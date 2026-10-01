# TACOSM-C5-LEARNED-STATE-BUDGET-001

Status: PREREGISTERED — result pending.

## Purpose

C5-LEARNED-STATE-001 and its continuous-vs-binary diagnostic show that the
current learned encoder has a real held-out semantic signal but only modest
Top-1 state retrieval. This experiment tests whether that boundary is mainly
caused by insufficient training budget.

The binary index is excluded. The measured path is:

\`noisy query -> learned continuous semantic encoder -> exhaustive state score\`.

All task generation and the train/evaluation code split remain fixed.

## Registered protocol

- M=64 persistent state items.
- H=64 task anchor.
- 5 seeds (0..4).
- training budgets: 32, 128, 512 epochs.
- 100 held-out noisy queries per seed/budget.
- input width 10; latent width 8.
- learning rate 0.02; margin 0.25.
- training codes = CODEBOOK[16:64].
- evaluation codes = CODEBOOK[:16].
- one-bit noisy query.
- same A1 H-invariant query/state generation as C5-LEARNED-STATE-001.

## Arms

### No-learning continuous

Random frozen dual encoder. State embeddings are cached once and all 64
states are continuously scored for each held-out query.

### Learned continuous

The same architecture is trained for one registered budget (32, 128, or 512
epochs), then evaluated with the same continuous exhaustive state scoring.

No budget is selected post hoc. Every registered budget is reported.

## Primary endpoint

Target-state Top-1 recall: whether the correct opaque persistent state address
is selected by the continuous semantic scorer.

Secondary:

- mean target rank;
- number of training pairs/updates;
- query arithmetic;
- one-time state-embedding arithmetic.

## Decision rule

The experiment is diagnostic, not a broad C5 test.

- a >=0.20 absolute gain from 32 to 512 epochs indicates material
  training-budget limitation;
- a <0.20 gain with 512-epoch recall below 0.50 indicates that additional
  budget does not remove the current low-recall boundary under this
  architecture/objective;
- a 512-epoch recall >=0.50 with <0.20 budget gain indicates a usable
  continuous retrieval level that is relatively insensitive to more budget.

The rules do not establish optimality or generalization.

## Cost accounting

For the 8-dimensional encoder over 64 cached state embeddings:

- query encoding = 80 MACs/query;
- continuous state scoring = 64 x 8 = 512 MACs/query;
- total query arithmetic = 592 MACs/query, excluding the one-time state build;
- state embedding build = 64 x 80 = 5,120 MACs/build.

## Interpretation boundary

A positive budget response would identify optimization budget as one limiting
factor. A plateau would shift attention toward the representation/objective,
including negative coverage and the structure of the semantic embedding.

Neither outcome establishes sublinear retrieval, learned indexing, long-horizon
memory, natural-language reasoning, or broad C5.

## Reproduction

Measurement command:

\`python scripts/run_c5_learned_state_budget_001.py\`

Contract:

\`contracts/TACOSM-C5-LEARNED-STATE-BUDGET-001.json\`
