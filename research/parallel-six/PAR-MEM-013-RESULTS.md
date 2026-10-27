# PAR-MEM-013 — trained real-source event-specific historical content discrimination

**Completed original exploratory genuine-source experiment, 27 October 2026.
Ten original checkpoint+optimizer+RNG bundles committed before formerly exposed
development scoring; independent clean checkout replay PASSED.**

**Result:** source-aligned associative TAC beat an identically-sized,
independently trained **aligned-associative-matrix-blind, source-context-only**
TAC on genuine old-value pair prediction at **16 and 64 source-observed
events**. The difference is small, and retrieval of exact requested
long-history events is **NOT independently qualified**. At 128 events
improvement in historical MAE is marginal/inconsistent and actual
correct-event ordering is only 53.1%.

## Historical reasons this is not just another key-space test

The canonical master plan documents DIAGKEY004 identity-aligned shared
read/write + tied values, ME100 source-authorized observed-only admission,
LATFUSE004 source-balanced associative heads and LATFUSE018 genuine old-event
content supervision. PAR-MEM-012 showed the integrated 410-float model
could exploit frozen reads at 16/64 yet failed exact old-address controls.
PAR-MEM-013 therefore trains a *specific pairwise discrimination* objective:
for each source owner at source clock t, observed past events a and b have
different legitimate prior ordinals (b is the **next actually observed
old event** after a). Both old PM2.5 readings are present in the legitimate
earlier observation stream but neither value is attached to a query key
or passed to a retrieval decoder as oracle target input.

A trained source-history model should reconstruct both and assign each to
its correct recorded event time. It cannot claim success merely from a
strongly ranked time-key embedding or high key entropy.

## Genuine physical source and source permissions

Publisher original UCI PRSA2017 Beijing sensor archive SHA256
`d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8`,
12 independently named monitoring stations. The authentic original data
adapter stops at original 2014 source rows; original 2015–17 physical
numbers remain untouched. Source normalization is from pre-March 2014.
Training: original March–April 2014 FIT records, 129 legitimate historical
source-observed events per source case, physically observed channels and
source hour-ordinal metadata. Evaluation: original May–June 2014
**previously examined genuine development** records. This is neither
sequestered TEST nor an independently validated model capability. Source
sites/anchors/pair selection depend only on known owner/event clocks,
not on old values or large target differences.

## Exact original neural controls

Five sources seeds 271028–271032. Two *fully fitted neural models per
seed*, each **98,326 genuinely learned parameters** and exactly **410
retained FP32 state values** plus owner/clock/count metadata; 36 original
FP32 AdamW updates, batch4 independent site histories, equal original
source case order and original physical pretrained 96,639 backbone per
seed. Both architectures also compute the same differentiable parallel
affine historical write graph. They differ only by a registered fixed
selector in the historical READ feature used by the learned decoder:

- **aligned_associative_read:** original read from source-known query
  address and the two 16x8 original source-written associative matrices.
- **trained_context_only_read:** a meaningfully trained 16D source-context
  feature from the current source embedding and source-known historical
  temporal code. **It never uses the extra two source-aligned matrices
  for its history decoder or next physical residual**. The relevant
  source-key, content, retention, write/router modules still have
  active gradient roles, not dummy counted weights.

**Critical:** the context-only architecture *retains the original
TimeAwareParallelResidualWorld's native fast prefix and native 8x8
associative memory*, and both sides still execute/retain the new matrix
write graph for computation parity. This is NOT a true stateless model,
a conventional GRU/SSM baseline, nor proof that ALL TAC persistent memory
is beneficial. Nor does an actively allocated matrix imply decoded value.

Original objective: two equal-weight SmoothL1 losses for the paired
genuine observed old values, plus a **0.35 × label-gap-weighted signed
historical pair-order softplus** (registered fixed margin 0.1 and
temperature 0.25), plus **0.15 × real next-hour source physical
prediction loss**. Each legitimate source event pair is included without
choosing examples because the readings differ.

[Original immutable source protocol](PAR-MEM-013-PROTOCOL.json) |
[Original strictly prefit source-case and all-seed optimizer order](PAR-MEM-013-FROZEN-PLAN.json) |
[Original ten trained checkpoint and full optimizer/RNG receipts](../replay-bundles/PAR-MEM-013/TRAIN_RECEIPT.json) |
[Full real source per-seed metrics](../replay-bundles/PAR-MEM-013/DEV_RESULTS.json) |
[Original successfully completed fit and clean-checkout replay](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40540826413)

### Source training-compute comparability

An original REAL source-input CPU training step (129 observed events ×
4 source stations) observed **2,113 distinct ATen operation/input-shape
signatures**, with **zero differences between aligned and context-only
arms** across forward, both old-value decoders, true physical loss,
backward, gradient norm clipping and AdamW. Profiler-reported subset
of operator FLOPs was exactly **828,338,312 per arm**, relative gap 0%.
Each learned parameter and genuine 410-float numerical source state
budget matches exactly, and train data/site order and update count match.

**Mathematically complete scalar FLOPs of every unsupported ATen nonlinear,
grad/optimizer and probability-law refinement were NOT certified.**
A one-example shape trace does not prove total five-seed compute parity
at arbitrary source backend paths. Therefore label all comparisons
exploratory, and do not infer qualified fair architectural superiority.
CPU wall-clock times are not a parity requirement.

## Original real-source five-seed DEV outcomes

Each delay covers exactly **120 genuine source-station/anchor pairs
per seed and arm**; the same 120 events are repeated across five
seeds so 600 source seed-evaluations are *NOT 600 independent test cases*.
The paired adjacent old PM2.5 values differ in 114 (lag16) or
116 (lag64/128) of the 120 source case pairs.

Pair MAE = mean normalized absolute error of BOTH correctly labeled
historical PM2.5 values, lower is better. Pair order accuracy only counts
genuinely different past observations. A perfect randomly keyed decoder
cannot be inferred from these results.

| Genuine event delay | Associative TAC pair MAE | Trained context-only pair MAE | Associative paired improvement | Associative correct-event order | Context-only order |
| --- | ---: | ---: | ---: | ---: | ---: |
| **16 observed events** | **0.583906** | 0.612037 | **4.596%** | **60.18%** | 49.65% |
| **64 observed events** | **0.521955** | 0.543315 | **3.931%** | **57.07%** | 47.24% |
| **128 observed events** | **0.582239** | 0.584347 | 0.361% | **53.10%** | 40.52% |

The associative control beats trained context-only pair MAE in **5/5
source seeds at lag16, 4/5 at lag64, and only 1/5 at lag128**.
It wins within-seed source-history order accuracy in 5/5 at lag16,
4/5 lag64, and 4/5 lag128, but original 128-event absolute order
accuracy is close to chance.

The average **true-labeled vs swapped-labeled absolute-error margin**
for the associative model is **+0.000666** at 16 and **+0.000392** at
64, then **−0.000124** at 128 normalized PM2.5 units.
These margins are TINY relative to the real average difference between
the two actual old source values (~0.190/0.199/0.236), and this
confirms weak *actual event-specific content discrimination* even where
paired MAE and ordering improve. Always report this small magnitude,
not accuracy alone.

The physically observed next-hour PM2.5 MAE stays near
**0.163159** for associative and **0.163161** for context-only across
these source anchors and lags. The added associative read does not
produce a meaningful future physical forecasting gain in this cohort.
No new persistence/seasonal baseline or independently trained conventional
GRU/SSM is available within this experiment.

## Engineering outcome, scientific interpretation and stop gates

- Demonstrated: the integrated six-family-compatible, genuinely trained
  source-known associative read helps an old-value PREDICTION objective
  relative to this independently trained context-only reader at shorter
  delays, consistently at 16 and substantially at 64.
- Not established: faithful numerical lookup of the specified old event,
  stable 128/256-step recovery, causal operational memory advantage,
  learned abstention/calibration, robotics, novel general-purpose language
  ability, or independent real-world generalization.
- No source target was fed to a query as an oracle. Old numeric targets
  DID naturally occur in legitimately observed earlier source history.
  The task is constructed on real physical data to test memory, not a
  recorded externally demanded workflow or agent's action decision.
- The source time queries may exploit atmospheric clock correlation and
  the original TAC fast/slow/native memory; the paired discrimination
  margin is too small to assert correct value addressing has been solved.
- Short 36-update training is not evidence of converged network
  optimization. Do not blindly increase training using exposed DEV
  results and then claim a new heldout PASS.

**Next qualified step:** owner-authorized new *independent* physical
longitudinal source with stable real monitor/entity IDs and actual
observed clocks, source original timestamp/quality/code and source
availability audit, then preregister untouched cohort and matched
~100k trainable stateless/GRU/SSM/matrix-blind controls. Keep exact
source exposure, all original checkpoint+optimizer+RNG+case schedule,
same full forward/back/optimizer mathematical FLOP budgets, and
independent replay. The repeatedly examined 2014 DEV cohort must
remain a diagnosis, not newly relabeled TEST.

Original source model checkpoints and all previous positive/negative
evidence unchanged; draft and unmerged.
