# COLLECT015 — residual-tiered memory: proposal and falsification

User proposal, 2 November 2026: combine (1) exact local mixing over a bounded
recent window, (2) phase-addressed associative delta state, (3) fast/slow decay,
(4) a gate using current association residual, and (5) a bounded routed exact
store populated only by gate admissions. The intended advantage is spending
exact storage on compression failures rather than every input. The proposed
falsification compares against admission of every item and a random gate at the
same admission count, sweeping key load beyond compressed-state capacity.

Status: recorded hypothesis plus a generated component diagnostic. This is not
an implemented five-part language architecture, learned gate, or neural model
training result. It does not change the frozen COLLECT014 training arms.

## Primary-source review and novelty

[Titans](https://arxiv.org/abs/2501.00663) combines neural historical memory with
attention for current context. [Memorizing Transformers](https://research.google/pubs/memorizing-transformers/)
uses an approximate kNN read of stored recent key/value pairs. These are related
mechanisms, not validations of this particular gate.

A much closer antecedent is [SR-TTT v2](https://arxiv.org/abs/2603.06642v2), revised
22 July 2026. It studies surprise-gated exact-attention residual storage. Its
author withdrew the initial retrieval gains after correcting answer-position
and future-access leakage, reporting zero exact-match performance in the
corrected conditions and identifying storage-position and addressing problems.
These are that author's reported results, not an independent reproduction here.
They do not prove this phase-state variant must fail, but they rule out treating
the broad residual-to-exact-cache arrangement as established novelty or success.
Older search abstracts can still display the withdrawn positive claims.

## Corrections to the proposed interpretation

Pre-write residual is e=||v-S*k||. With a unit key and unit-gain plain delta,
S'=S+(v-S*k)*k^T gives S'*k=v immediately. Therefore a large pre-write error
does not prove that the updated compressed state cannot store the item. It may
just indicate novelty. Later writes, decay and future query demands determine
whether the item remains retrievable.

Conversely, an item predicted perfectly now may become important after later
interfering writes. Surprise is not future relevance. The executed four-event
counterexample has pre-write errors [1,1,0,1.1728]. The repeated zero-error fact
is not cached, and the final correlated write makes its class read incorrect:
[0.0975,0.6534] instead of [1,0].

Use v from the currently observed representation, not an unavailable future
ground-truth answer. Retained key/value tensors are exact only while resident;
address matching, eviction, versions, learned queries and readout can still fail.
A bounded buffer remains bounded; its occupancy may saturate and cumulative
admissions can grow linearly with surprising input length. A growing overflow
archive gives up fixed-state memory. Every observed overwrite must invalidate
or refresh a cached older value, even when the new event is not admitted, or the
cache can override a correct compressed read with stale data.

## Executed best-case component test

PROTOCOL.json was fixed before generated evaluation. probe.py uses supplied
unit phase addresses (16 dimensions), eight one-hot value classes, two delta
states with decays .98/.999, a common last-eight-record exact buffer, and a
FIFO exact store of capacity 8 or 16. Reads use the local buffer, then an
identical-address cache match, then compressed state. This is idealized exact
address matching; learned routing and paraphrase matching are not tested.

Threshold is fixed at .75. For each episode, the random control admits exactly
as many events as the residual policy, sampled without query labels. This is
an explicitly retrospective count-matched evaluation control, not a deployable
causal admission rule with access to the future admission count. All other
state updates and capacities are identical. Repeated-address invalidation
prevents stale cache values; coalescing and FIFO eviction costs are counted.

Three RNG seeds, 16 histories per seed, key loads 8/16/32/64, two capacities,
three regimes and four policies give 4,608 policy evaluations. There are 576
unique stream histories; capacities/policies reuse them. All active fact keys
are queried after their context; their answers do not enter read/admission.
Uniform facts, repeated predictable distractors and novel distractors distinguish
compressibility from relevance. Numeric retained state includes both matrices,
the local window and resident cache vectors, excluding Python/search/workspace.

The complete 96-cell summary is in SUMMARY.json. All raw predictions, targets,
routes, residuals, admissions and evictions are retained in RESULTS.json in the
companion evidence archive. Lower capacities and negative results are retained.

At capacity 16:

| Fact keys | Stream | Residual class accuracy | Matched random | Residual complete class maps | Matched random maps |
| --- | --- | ---: | ---: | ---: | ---: |
| 16 | Uniform facts | 99.35% | 98.70% | 93.75% | 81.25% |
| 16 | Repeated distractor | 99.35% | 77.86% | 91.67% | 0% |
| 16 | Novel distractors | 39.45% | 39.58% | 0% | 0% |
| 32 | Uniform facts | 64.52% | 64.58% | 0% | 0% |
| 32 | Repeated distractor | 62.11% | 60.22% | 0% | 0% |
| 32 | Novel distractors | 24.54% | 24.61% | 0% | 0% |
| 64 | Uniform facts | 39.26% | 39.26% | 0% | 0% |
| 64 | Repeated distractor | 38.41% | 45.41% | 0% | 0% |
| 64 | Novel distractors | 19.37% | 19.30% | 0% | 0% |

These are descriptive means over 48 histories per cell, not significance or
architecture-superiority claims. At load 8 with a local window of 8 facts,
uniform-fact retrieval is already covered by the exact local path, so that
cell cannot establish a residual-cache contribution.

The gate substantially helps one favorable repeated-distractor regime at
moderate load. It does not reliably beat matched random across load/regime,
and complete-map reliability remains zero at larger loads. At 64 facts with
repeated distractors, its class accuracy is worse than random despite very
similar exact-vector recall (25.03% vs 24.45%). Classification and exact-vector
recall are different metrics; neither is substituted for the other.

The broad claim that current residual identifies exactly the items needed for
future retrieval fails this test. The narrower conditional benefit remains a
candidate. No general global-retrieval or fixed-state capacity barrier is solved.

## Next discriminating repairs

Candidate future controls are an age/position-stratified protected quota, a
small uniform reservoir alongside the residual cache, and a query-relevance
predictor trained on delayed use. Each changes the hypothesis and needs a new
preregistered comparison with a matched storage/compute budget. The reservoir
is especially useful as a hedge against currently predictable but later useful
facts. Its benefit is not measured here. Stale-value invalidation is a necessary
engineering safeguard, not a learned retrieval breakthrough.

Before any language integration, require all-seed whole-map performance,
query-answer exclusion, future-token poison tests, oracle-storage/read controls,
per-position admission/containment, overwritten-value rejection and complete
storage/search/eviction cost. Keep separate the events “not admitted”, “evicted”,
“stored but misaddressed”, and “correct vector read but wrong decoded answer”.
