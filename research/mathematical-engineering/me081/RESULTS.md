# ME081 — stronger native physical forecasts, with unresolved quiet-state failures

**Five original models trained. Full registered gate: FAIL. Evidence tier:
MIXED_TRADEOFF. [Draft PR #375](https://github.com/FPC-effortless/TAC-OSM/pull/375), unmerged.**

The new complete model has 100,165 trainable/deployed parameters. It retains
native fast32/slow32/8×8 associative memory, trains through 32 real observation
frames, shares a horizon-query decoder across 1/4/16-frame forecasts, and learns
quiet/moving coordinate distributions using CRPS + a small log score + factual
motion supervision. This is actual new training, not an execution-only result.

## Same-task primary forecasts

Same original 40 exposed REPORT episodes and eight observed frames per query.
Five-seed means of individual model errors, not an ensemble deployment.

| Physical MAE, lower is better | ME080 native | ME081 native | Persistence |
| --- | ---: | ---: | ---: |
| 1 frame, 960 pairs | 0.059393 | **0.055968** | 0.068668 |
| 4 frames, 956 pairs | 0.219619 | **0.166113** | 0.302400 |
| 16 frames, 928 pairs | 0.883994 | **0.737952** | 0.975966 |

Mean errors decrease **5.77%, 24.36% and 16.52%** versus ME080, and **18.49%,
45.07% and 24.39%** versus persistence. At four frames, the model also improves
10.67% over the stronger older ME079 native mean of 0.185953.

Actual-moving one-frame MAE decreases from ME080 **0.533420 to 0.408457**,
23.43% lower. This is also below the older ME078 native 0.421182 and the fixed
clipped-velocity reference 0.482509 on these same 84 factual moving queries.
Overall one-frame error is below clipped velocity's 0.058998. These are exposed
population mean comparisons, not independent superiority qualification.

All five fits improve one-frame mean MAE over the corresponding ME080 fit.
Nevertheless, the paired 40-episode bootstrap 95% interval for new-minus-ME080
one-frame MAE is **[−0.012166, +0.005730]**, including zero. It conditions on the
fixed fitted models and this repeatedly exposed population, not retraining or
independent source variation. Operator/session/site independence is unverified.

## Important regressions: quiet precision and density scoring

| Near-stationary physical MAE | ME080 | ME081 | Change |
| --- | ---: | ---: | ---: |
| 1 frame | 0.013938 | 0.022168 | +59.04% |
| 4 frames | 0.005226 | 0.008134 | +55.65% |
| 16 frames | 0.013141 | 0.034465 | +162.28% |

The ME077 stationary specialist remains much better at one frame: 0.001701.
ME081 is about 13.0 times worse on that subset and does not meet the registered
quiet-preservation condition. No task's overall one-frame mean regresses more
than 10% versus persistence, but that does not erase quiet-state failures.

Physical marginal NLL also worsens at every horizon: **−5.64984/−4.14339/−2.47950**
versus ME080 **−5.94993/−4.30164/−2.59987**, where lower is better. Thus this is
not an across-the-board probabilistic calibration improvement.

Mean individual-fit one-frame motion AP/AUC improves versus ME080 from
0.53443/0.89689 to **0.54992/0.91466**. It remains below the stronger historical
ME078/079 motion ranking. Event probability is a separate factual forecast,
not latent mixture-component identity or a safe action decision.

## Proper interval scores improve

| Horizon | ME081 nominal90% coordinate coverage | Mean width | ME081 interval score | ME080 interval score |
| --- | ---: | ---: | ---: | ---: |
| 1 | 92.92% | 0.214704 | **0.343562** | 0.709378 |
| 4 | 92.88% | 0.818263 | **1.541268** | 2.490101 |
| 16 | 91.88% | 3.706394 | **5.197868** | 10.499870 |

Proper interval scores fall **51.57%, 38.10% and 50.50%**, with coverage above
the registered 85% floor. Coverage refers to marginal coordinates, not a joint
trajectory or whole-episode 90% guarantee. The NLL regressions still matter.

## Actual longer-context readout

The registered secondary condition contains **904 authentic 32-frame query
prefixes**, with 904/900/872 genuine future targets at 1/4/16 frames, from the
same original exposed episodes. No synthetic events, padding or future inputs.

| Same eligible queries: physical MAE | ME081 with 8 frames | ME081 with 32 frames |
| --- | ---: | ---: |
| 1 frame | 0.055775 | 0.055557 |
| 4 frames | 0.159718 | 0.157857 |
| 16 frames | 0.748874 | 0.740098 |

These small descriptive differences compare exactly the same query IDs and
actual targets. They do not establish a material memory advantage, architecture
superiority, arbitrary-history retention or independent transfer. The 32-frame
model beats persistence at all three horizons and clipped velocity on its
moving one-frame subset. All registered rows are preserved, not selected winners.

## Acceptance, compute and preservation

**Full gate FAIL:** quiet preservation and the primary paired-episode CI condition
fail. Other registered conditions pass, including the primary 5% mean reduction,
movement guardrail, stricter four-frame improvement over ME079, interval-score
and coverage conditions, all five seed wins, and secondary physical references.
See the complete unedited [CAPABILITY_RESULTS.json](CAPABILITY_RESULTS.json).
**No additional fits are authorized under ME081.** No threshold was relaxed.

Original source: BodenAI/RW-RL-HIL revision
9290f5a96dc91ea58256e81d9ce2dd9621890b51, CC-BY-4.0; 120 FIT / 40 previously
exposed REPORT episodes. Source CAL/TEST and protected tissue-box remain closed.
Future human takeover/source endings affect target eligibility, never features.
This is factual conditional forecasting, not identified causal intervention value.

Five cold starts each receive 2,048 guarded AdamW updates, 49,152 windows and
1,572,864 observed prefix presentations. Valid dense future target presentations
are approximately 4.71 million per fit: repetitions, not independent measurements.
Observed-prefix exposure is **four times ME080**, with different loss/context/
sampling eligibility. No matched-compute, sample-efficiency, architectural winner
or training-speed superiority claim. Whole-model budget and actual 2,048-step
receipt verification pass; CPU is about 123 seconds per fit on this host only.

Prefit implementation/registration frozen at **b5698c86**. All five original
weights, optimizer/RNG states and live receipts frozen at **dd71fee2 BEFORE**
either report readout. Original 8/32-frame forecasts and scores committed at
**bf23f52b**. There are **27,600 model forecast records** across seeds/history
conditions, still only the original 40 episode clusters. Independent replay
status is recorded separately in ARTIFACT.md; no software test alone qualifies it.
Original historical ME078/079/080 numerical exceptions remain unchanged.

The local full suite passes **1,169 tests**. Mathematical correctness of CRPS,
finite active gradients and causal state carry are software facts. They are not
new real-world capability evidence by themselves.

## What is and is not established

This candidate has stronger average moving-state and multi-horizon physical
forecasting, and better interval scores, on an exposed real development cohort.
Stationary precision and density scoring remain serious weaknesses. The model
is not a general intelligence breakthrough, long-memory winner, causal controller,
safe physical actor, enterprise-qualified system or verified customer benefit.
The next substantive capability registration must preserve movement/mid-horizon
skills while recovering quiet precision, then seek independent real qualification.
No attribution campaign, hidden retry, protected TEST use, promotion or deployment.

## Final replay qualification

Independent fresh remote checkout: **PASS**, all five originals, both histories,
27,600raw forecasts and JSON-exact full summary locally. Sparse checkout was used
after a disposable full clone exhausted disk; original research bytes remained
intact. **Strict external replay FAIL** on a small number of point/uncertainty
fields; complete all-model census is committed and summarized in ARTIFACT.md.
Targets, coverage, risk and NLL fields meet the external contract; that does not
qualify the full replay. No tolerance widening or original output changes.
