# TACOSM-C5-SPARSE-FUNNEL-001

Status: PREREGISTERED — full registered measurement triggered.

## Purpose

The prior cosine-selective experiment isolated a specific failure boundary:
the full-candidate cosine representation was useful, while the cheap binary
proposal discarded too much relevant state before reranking.

This experiment adds a continuous teacher-student proposal. The teacher sees
the complete persistent-state population during offline distillation. The
student learns a query-to-prototype distribution by KL divergence. At runtime
the student proposes a bounded prototype union, and the fixed cosine teacher
reranks that shortlist.

## Registered measurements

- M=64 persistent states.
- H=64, 128, 256.
- Five seeds.
- Width-16 raw-trained representation held fixed.
- 512 representation-training epochs.
- 256 proposal-distillation epochs.
- Eight mean-gradient negatives and one positive view.
- 48 training codes and held-out query codes. Distillation state supervision is restricted to the persistent states whose values are in the training-code set; evaluation states are excluded from the distillation target.
- 16 continuous prototypes with capacity four.
- K in 4, 8, 16, using prototype beams 1, 2, 4.
- Fixed cosine reranking.
- Logical packed-state layout diagnostics.
- Exact downstream candidate index and executor.

## Objective

The teacher scores every training-state item during distillation:

p_T(i) = softmax(s_T(q,c_i) / tau_T). Runtime reranking scores only the bounded
shortlist, while the exhaustive reference scores the complete 64-state population.

Teacher probability mass is aggregated to the prototype buckets:

P_T(j) = sum_{i in B_j} p_T(i).

The student predicts a prototype distribution:

p_S(j) = softmax(s_S(q,B_j) / tau_S).

and minimizes:

L_KD = D_KL(P_T || p_S).

The teacher is fixed during proposal training. Gold target addresses and
downstream outputs are not used by the distillation optimizer. Runtime reranking
then uses the full persistent-state pool.

## Funnel

S_t -> packed persistent state -> student proposal -> over-fetch ->
teacher cosine rerank -> executable program -> exact CASM/executor ->
verification -> S_{t+1}.

CASM remains downstream of selection. It is not used as the semantic relevance
oracle for the proposal. The registered primary endpoint is actual target-state
recall; proposal retention remains a separate coarse-boundary diagnostic.

## Budget diagnostic

K=4 is the existing aggressive budget. K=8 and K=16 test whether the current
loss is primarily a proposal-budget problem or a proposal-mechanism problem.
The same student is evaluated at all three budgets.

## Negative filtering

TeacherNegativeFilter provides optional teacher-based screening for contrastive
experiments. Negatives whose teacher probability is too close to the positive
are rejected as potential false negatives. Accepted and rejected candidates are
both accounted for; rejected candidates are never silently dropped.

## Addressing and hardware

product_key_index.py implements a two-factor product-key addressing experiment.
It is intentionally separate from the teacher reranker and does not claim
universal O(sqrt(M)) behavior or hardware speedup.

packed_state.py records logical row-major layout, contiguous row runs, and
packing density. Arithmetic reduction is not treated as wall-clock speedup.

## Reproduction

Smoke: python scripts/run_c5_sparse_funnel_001.py --smoke

Full: python scripts/run_c5_sparse_funnel_001.py

Artifact: artifacts/TACOSM-C5-SPARSE-FUNNEL-001.json

No scientific conclusion is licensed by implementation or unit-test success
alone. The measurement artifact and preconditions are required.
