# TACOSM-GRAPH-CASM-BEHAVIORAL-COMPATIBILITY-012

## Scientific status

PREREGISTERED — implementation under audit; no confirmatory result exists.

This experiment is a new boundary after G-CASM-010 and G-CASM-011. It does not
modify either experiment or reuse their results as tunable outcomes.

## Scientific question

Can a query-conditioned behavioral compatibility model recover the relevant
executable candidates more reliably than the G-CASM-010 exact-identity router,
and can that recovery translate into capability-constrained selective execution?

The distinction is intentional:

- G-CASM-010 trains an exact candidate/query identity-style dual encoder.
- G-CASM-012 trains candidate/support behavioral compatibility.
- G-CASM-011 supplies an exact, non-learned public-support behavior index only as
  an information/identifiability control.

## Proposed intervention

For candidate graph G_i and observed support row (x,y), the learner predicts
compatibility(G_i, x, y).

A support-set score is formed from the row-level scores using a differentiable
soft-min, so a candidate must explain the weakest observed row rather than
accumulating unrelated positive evidence.

The training loss contains only:

1. row-level compatibility labels derived from public training programs; and
2. a support-set compatibility label defined as the conjunction of those public
   row-level labels.

Target candidate identity is not a training label.

## Benchmark validity audit

The benchmark is inherited from G-CASM-010 but the primary interpretation is
changed.

The four-row primary condition can be ambiguous at larger candidate
populations. Therefore exact target Top-1 is retained as a diagnostic and is
not the sole capability definition.

The executable target is always present and has a unique complete truth table
within each candidate pool. Exact execution over verifier-only rows therefore
provides the capability-resolution step after routing.

Support sizes are restricted to 4, 8 and 12. A 16-row routing condition is not
used because it would leave no verifier-only rows and therefore make the
execution-capability endpoint vacuous.

The primary endpoint is evaluated only at support size 4, preserving the
original difficult information boundary. Support sizes 8 and 12 are
generalization/diagnostic conditions.

## Leakage audit

| Channel | Allowed? | Boundary |
|---|---|---|
| candidate executable graph | yes | public intervention |
| support input/output rows | yes | router-visible |
| target candidate index | no | evaluator only |
| verifier-only rows | no | executor/verifier only |
| complete evaluation truth table | no | evaluator only |
| evaluation target structure | no | absent from training |
| evaluation target truth signature | no | absent from training |
| candidate order | no causal information | shuffled independently |
| training target identity | no | never encoded in compatibility loss |
| test outcome | no | unavailable until after route |
| representation fitting on evaluation results | prohibited | no post-result fitting |

The training objective receives candidate structure and support rows, but never
receives the evaluator's target identity. Positive compatibility is a property
of each candidate against the support, not a privileged label saying which
candidate is the answer.

## Protocol audit

Contract:
contracts/TACOSM-GRAPH-CASM-BEHAVIORAL-COMPATIBILITY-012.json

Pinned parent:
06c36f4cda96b8b0ae4411013c7fdf8939b1d8f8

Pinned external generator:
FPC-effortless/cdl-attention-experiment@c31554413301e3c9d3e6b3f8c8c6be572a74a748

Registered:

- M = 32, 64, 128, 256, 512;
- seeds = 0..4;
- 256 training programs per seed;
- 1,200 router steps;
- 32 evaluation tasks per seed/M;
- budgets = 1, 2, 4, 8;
- routing support sizes = 4, 8, 12;
- primary support size = 4.

No hyperparameter selection from confirmatory outcomes is allowed.

## Implementation audit

The required execution chain is:

contract -> split -> public graph features + support rows -> behavioral router
-> ranked candidates -> exact graph executor -> verifier -> work accounting
-> artifact.

The exact executor is unchanged from G-CASM-010.

The behavioral router is implemented independently in
src/tac_osm/behavioral_compatibility_router.py.

The runner uses the G-CASM-010 benchmark generator and exact executor as a
frozen parent dependency rather than copying their semantics into a second
generator.

## Instrument controls

Before a confirmatory result may be promoted, the test suite must establish:

- exact exhaustive graph execution succeeds;
- training/evaluation structure and truth-table signatures are disjoint;
- candidate-order shuffling preserves candidate identity/score correspondence;
- changing evaluator-only target metadata cannot affect router output;
- constant and random rankings remain near their expected negative-control
  behavior;
- support-label permutation degrades compatibility selection;
- fixed-budget work executes every selected candidate;
- adaptive work stops only after a verifier-passing candidate;
- an exhaustive run executes the entire candidate population rather than
  stopping after the first success.

## Statistical audit

The unit of analysis is the seed/M/support-size cell.

Raw trial-level and raw per-seed values are retained.

Because evaluation tasks generated under a seed are correlated by the generator,
the confirmatory uncertainty analysis uses seed-level bootstrap rather than
treating individual tasks as independent replicates.

The primary bootstrap repeats the complete "eligible budget -> minimum work"
selection rule for every resample.

No asymptotic exponent is a primary endpoint.

## Scientific interpretation

A positive result may establish only:

On the registered synthetic executable-graph workload, compatibility-conditioned
routing can reduce verified execution work at a specified capability level.

It does not establish:

- semantic language competence;
- autonomous program discovery;
- hidden-wiring inference when wiring is unavailable;
- asymptotic sublinear complexity;
- hardware wall-clock superiority;
- universal routing superiority;
- C5 outside the registered workload.

A result where support-set recall improves but the capability-constrained work
criterion fails is a routing/representation result, not a C5 result.

## Relationship to G-CASM-011

G-CASM-011 asks whether public support rows uniquely identify the target among
the candidate pool.

G-CASM-012 accepts that the support may identify a set rather than a unique
candidate and asks whether learned compatibility can recover that relevant set
well enough for exact execution and verification to resolve the remaining
ambiguity.

Thus the experiments are complementary:

011: information available / identifiable?

012: can the learner recover relevant candidates?

010: can learned routing then reduce actual execution work?

## Confirmatory-run rule

The smoke path is implementation-only.

A confirmatory artifact is admissible only when:

- all preflight tests pass;
- the registered full grid is executed;
- no protocol field is changed;
- raw trial data and provenance are retained;
- the exact artifact is frozen before claim disposition.
