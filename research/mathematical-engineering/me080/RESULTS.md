# ME080 — trained native reusable-state mixture world model

**Five original real-data fits complete. Full registered gate FAIL.
Evidence tier MIXED_TRADEOFF. Draft PR[#371](https://github.com/FPC-effortless/TAC-OSM/pull/371), unmerged.**

## Whole-model capability result

The new99,816parameter native model retains fast32/slow32/8×8associative memory.
It learns quiet/movement coordinate mixtures, with medians and intervals from
the same distribution, and explicit reusable observed/recurrent state.
No new GRU/Transformer fits were needed for this single-architecture development
study; no architecture-winner, component-attribution or matched-compute claim.

Same licensed source,120FIT/40previously exposedREPORT episodes. Same960H1,
956H4and928H16actual future pairs. Five original seeds, not independent new data.
All scores below are means of **individual** models, not a five-model deployment.

| Physical MAE, lower better | Frozen ME079 native | ME080 native | Persistence | Clipped velocity |
| --- | ---: | ---: | ---: | ---: |
| 1frame | .060595 | .059393 | .068668 | .058998 |
| 4frames | .185953 | .219619 | .302400 | .311101 |
| 16frames | .947424 | .883994 | .975966 | 1.564791 |

**What improved:** H1mean error falls1.98%and H16falls6.69%relative to ME079.
H1stationary error drops43.79%, .024795to.013938. H16proper interval score falls
12.62%,12.01622to10.49987. Compared with persistence, mean errors are13.51%,
27.37%and9.42%lower at1/4/16frames respectively.

**What regressed:** H4MAE rises18.10%. H1actual-moving error increases from
ME079.433934to.533420 (+22.93%); ME078remains stronger at.421182. H1motion
AP/AUC fall fromME079.60775/.91971to.53443/.89689. H1/H4interval scores worsen.
H1stationary error remains8.19times the ME077specialist's.001701, and3.50times
persistence's.003983. Native H1all also remains slightly worse than clipped
velocity; no blanket best-baseline claim.

The H1new-minus-ME079paired40episode bootstrap95%interval is
**[-.009942,+.006327]**, including zero. Four of five individual new fits improve
H1relative to corresponding old fits, but the registered5%mean threshold fails.
These are fixed-model intervals on already exposed data, not retraining or
new-source confidence. No architecture family superiority is established.

## Uncertainty and hard gates

| Horizon | Coordinate90%coverage | Mean physical width | Proper interval score | ME079 interval score |
| --- | ---: | ---: | ---: | ---: |
| 1 |93.46%|.321503|.709378|.397724|
| 4 |91.96%|.852057|2.490101|2.105854|
|16 |91.91%|4.850747|10.499870|12.016222|

Coverage meets the registered85%floor; conservative coverage alone does NOT
qualify sharp, calibrated uncertainty. Full machine gates and individual-fit
metrics are in[CAPABILITY_RESULTS.json](CAPABILITY_RESULTS.json). Primary moving
and stationary preservation gates FAIL, H4accuracy guardrail FAILS, and H1/H4
interval-score guardrails FAIL. One-step practical benefit is not established.
**This registration authorizes no more fits.** Keep all negative results.

## Reuse is working execution, not a long-memory intelligence result

The model computes encoder/address/observed fast projections once per training
chunk and carries explicit state for inference. Trained-model verification on
all960authentic eight-frame query inputs per seed confirms full-chunk,
split3+5and one-observation step execution agree within the frozen tolerance.
Caller state is not mutated. Entity interleaving/isolation and gradient paths
also pass software tests. Cache payload157float32values=628bytes/entity excludes
allocator/container overhead, activations and model weights.

There is no global hidden memoization, fused GPU kernel, JIT-throughput result,
associative scan or category-theoretic novelty. The nonlinear recurrence is
sequential; observation-only work is batched. Actual prediction training/scoring
still uses eight-frame reset histories. Forty-step execution-equivalence tests
DO NOT establish useful long-history prediction. Inference caches must be
invalidated on weight/preprocessing changes or stream restart. Caller supplies
known commands/entity routing; the model does not discover identity or act.

## Integrity, repairs and boundaries

Prefit source/registration committed4c7309ffBEFORE all fitting. All five original
model/optimizer/RNG/live receipts frozene37a4b88BEFORE REPORT scoring. Original
forecasts/targets/scores committedea6cdd5e. Actual whole models each99,816
trainable/deployed parameters, all2048guarded AdamW steps. Each seed sees49,152
windows/393,216prefix presentations and1,172,102–1,172,868eligible future target
presentations (repeats, not that many independent observations). CPU34s/seed on
this host; not a device guarantee or fair-compute victory.

The frozen runner's first `finalize` failed because its final JSON omitted the
root original-registration reference, despite original model receipts and
checkpoint hashes correctly binding that registration. Scores were computed
after this metadata failure; every original was already Git-frozen and the
individual receipt/model checks had passed. A separate no-fit final-delivery
helper now validates original hashes, actual tensors and all2048updates, adds
the missing reference to the NEW final document and passes final validation.
**Frozen runner, original registration, model weights and forecasts unchanged.**
See[POSTFIT_METADATA_REPAIR.json](POSTFIT_METADATA_REPAIR.json). A separate
summary-only NumPy boolean serialization error was corrected without changing
predictions, gates or numerical metrics. Neither incident authorizes retraining.

CDF inversion and scalar scoring use prospectively specified float64; neural
parameters stayfloat32. Original raw replay limits rtol1e-5/atol1e-6remain fixed.
ME078/079strict external interval-score failures stay visible and unchanged.
Independent local clean-clone replay PASS. The complete external census
(run40006593903) restores all five models and14,220forecasts: three lower
quantiles and one interval score FAIL the frozen tolerance; all other recorded
fields PASS. Full strict external replay remains FAIL. See ARTIFACT.md; a
source/test/audit-process pass is not a full original-score replay PASS.

Physical forecasting only. Source CAL/TEST/protected tissue-box stayed closed.
Future takeover/source endings mask target availability, not input features.
No identified causal action consequence, autonomous control, safety, enterprise
transfer, independent source, customer benefit or commercial qualification.
See[PROTOCOL.md](PROTOCOL.md)and[MATHEMATICS.md](MATHEMATICS.md).

## Next capability direction

Retain the genuinely useful trained candidates and this mixed result. The
remaining requirement is one model that preserves quiet precision AND strong
changing-state/mid-horizon forecasts, not a new claim based only on state
caching or mean error. Choose the next substantial integrated objective under
a new registration, with all development attempts visible and no protected
TEST tuning. Independent-source qualification and operator usefulness remain
separate outstanding gates. No more fits under ME080.

Final local repository suite: **1,161 tests PASS**. Static live-training audit: **198 supported Python files PASS**.
