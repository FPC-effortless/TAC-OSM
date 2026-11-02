# COLLECT016: carried credit and task-specific memory

This protocol follows completed COLLECT014–015, not a claimed architecture win.
The user authorizes supervised multi-window training and a trained current-value
control. The previous training/evaluation mismatch was identified, not all other
confounders isolated. One pilot cannot deliver a final architecture verdict.

## What is established and what remains ambiguous

- COLLECT014's 13.89% normal versus 14.58% wrong-history scores do not establish
  a real reverse effect or a particular drifting mechanism. They show no useful
  retrieval advantage in that small untrained assay. Histories are shared across
  seeds/conditions; counting them as fresh independent examples is incorrect.
- Full carried-state forward computation does not ensure cross-window learning.
  Explicit detach removes earlier-state credit. Short-horizon training can still
  learn a useful stationary update rule; detachment is not proof that all long
  retention learning is impossible. We compare it directly instead of assuming.
- Connected BPTT provides a path, not a guarantee of effective gradients. We
  test an actual earlier activation and numerical reference, keep gradient norms,
  and report the small finite training budget.
- The current-only condition retains native fast/slow/rotary/SSA history. It
  tests the incremental historical phase lane, not all recurrent memory. Removing
  unused address weights and reallocating them to active FFNs also changes the
  distribution of capacity. All compatible initial tensors/FFN slices are shared.
- Same examples, targets and optimizer updates are not full FLOPs parity. The
  detached graph and current-only computation have different training work.

## The running-summary proposals: valid uses and limits

| Representation | Appropriate question | Limit relevant to this research |
| --- | --- | --- |
| Coverage bitset / interval set | Which known locations were visited? | Size depends on resolution/domain; it cannot reconstruct arbitrary observed text. Bloom filters introduce false positives. |
| Bayesian belief | Which modeled hidden states remain plausible? | Sufficient under the assumed Markov model and known distributions; exact belief can itself be large and the model can be wrong. |
| Sparse landmarks / relative poses | Where am I relative to persistent anchors? | Depends on observability, association and map size; cannot recover every discarded observation. |
| Cardinality / frequency sketch | How many distinct items or approximate counts? | Answers restricted aggregate queries with algorithm-specific guarantees, not arbitrary fact lookup. |
| Learned recurrent state | Flexible task-specific prediction | Fixed storage and learned compression do not imply sufficiency for unknown future queries. |

Keeping position plus a summary is useful when the task has a compact sufficient
statistic. It does not prove that arbitrary language histories have one. For N
independent facts each with C possible values, lossless recovery of every possible
map needs at least N log2(C) bits: C^N maps require that many distinct state codes.
This is a counting argument under finite-state/exact-recall assumptions, not a
claim that neural states cannot generalize or that physical capacity caused the
COLLECT014 failure. Approximation, restricted queries and external archives change
the requirement. A particular hard-slot implementation is not logically forced.

Active sensing can help when the agent controls observations and pays an explicit
query/movement cost. This passive byte-stream experiment has no such policy. An
external archive could support selective revisits, but archive bytes and retrieval
cost must be counted; they do not disappear into a constant-size state claim.

## Biological suggestions: inspiration with corrections

Human fMRI evidence links learning-related hippocampal representational
differentiation to reduced interference [1,2]. This motivates testing write/read
separation or contrastive addresses, not claiming perfect orthogonalization or
unlimited noninterfering storage. Finite-dimensional orthogonal keys are bounded.

The hippocampus is not an exact, interference-free cache. Direct human recordings
also distinguish correct and contextually misattributed false recall [3]. The
literal "exact first, lossy later" description is therefore too strong.

A complementary-systems consolidation model proposes transferring memories when
this improves generalization and warns that indiscriminate transfer can overfit
noise [4]. This is a specific computational theory with supporting context, not
a settled circuit-level implementation of our proposal. Neural Episodic Control
already combines neural representations with rapidly updated episodic buffers
[5]; hybridization itself is not a novelty claim. Biological learning is plastic;
the absence of literal backpropagation does not show the routing task untrainable.
No evidence here supports the numerical assertion that 95% of tokens should
never use exact storage. That fraction must be measured for a task and budget.

## Candidate repairs, after measuring the current failure

1. Train future query utility, not just pre-write surprise. The present copy loss
   sends credit from later queries through the actual carried graph. If this
   fails, separate local key parsing/binding, query-address alignment, retention
   and output decoding with interventions before adding another memory tier.
2. Compare stable write addresses with a separately learned query projection.
   The current phase lane reuses one contextual key for reads and writes; its LM
   frontend is not COLLECT010's validated small key encoder. Contrasting similar
   identifiers may improve address separation, but can harm useful similarity.
3. A relevance-trained exact store could retain a small uniform/age-stratified
   reserve alongside learned admissions. This hedges against predictable but
   later useful facts; it is an untested proposal with explicit byte/search costs.
4. Consolidate an archived association only after delayed rehearsal verifies its
   compressed recall. A finite audit sample cannot certify every forgotten fact.
   Version invalidation and overwrite protection remain mandatory.
5. Use typed compact state for quantities with known update rules, and preserve
   exact evidence for queries requiring it. Measure both selection mistakes and
   reconstruction failures. Neither the types nor the router are implemented by
   merely describing this design.

COLLECT016 implements the first training/interface test only. It does not add an
external store, novelty claim, biological circuit or active-sensing controller.
Source-grounded copy accuracy, ordinary-text DEV, overwrite behavior, absence,
query dependence and efficiency remain separate scorecards.

## Primary sources checked

1. Experience-dependent pattern differentiation (2016):
   https://www.nature.com/articles/ncomms11066
2. Abrupt remapping and interference resolution (2021):
   https://www.nature.com/articles/s41467-021-25126-0
3. Hippocampal activity and contextual false memories (2023):
   https://pubmed.ncbi.nlm.nih.gov/37751551/
4. Generalization-optimized consolidation (2023):
   https://www.nature.com/articles/s41593-023-01382-9
5. Pritzel et al. (ICML 2017), Neural Episodic Control:
   https://proceedings.mlr.press/v70/pritzel17a.html
6. Tallec & Ollivier (2017), truncation bias / ARTBP:
   https://arxiv.org/abs/1705.08209
7. Dai et al. (ACL 2019), Transformer-XL segment recurrence:
   https://arxiv.org/abs/1901.02860
8. Kaelbling, Littman & Cassandra (1998), planning under partial observability:
   https://doi.org/10.1016/S0004-3702(98)00023-X

Some publisher full pages were blocked by login/anti-bot checks; primary indexed
abstracts and author/publisher metadata were used where available. Findings are
limited to the inspected passages; no full-text replication is claimed.
