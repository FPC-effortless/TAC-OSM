# ME078 — learned real motion capability, with a stationary/architecture trade-off

**15 original real-data models trained and Git-frozen. MIXED_TRADEOFF.** Native
TAC learned substantially better movement prediction and movement discrimination;
it did NOT pass the registered learned-control superiority gate. All observations
are historically exposed development, not independently unseen generalization.

| Horizon, genuine query-target pairs/seed | Native TAC MAE | GRU MAE | Causal Transformer MAE | Persistence MAE |
| --- | ---: | ---: | ---: | ---: |
| 1 frame,960 | **0.061221** | 0.061701 | 0.062945 | 0.068668 |
| 4 frames,956 | 0.195231 | **0.190473** | 0.194741 | 0.302400 |
| 16 frames,928 | 0.950255 | **0.944978** | 0.946789 | 0.975966 |

Mean original recorded-joint absolute error across five independent fitted seeds.
Native improves persistence by10.84%/35.44%/2.63%, and H1ME077 by7.91%. Original
ME077 and new H1targets/queries are exactly the same960origins in40episodes.

**Actual moving H1(84queries):** native0.421182 versus frozenME0770.741981,
persistence0.743241, clipped velocity0.482509, and newly coordinate-corrected
FIT servo0.709437. Native moving error is43.24%lower than ME077 and12.71%lower
than the strongest tested physical reference, clipped velocity. The servo now
converts action-normalized commands into state coordinates; its values differ
from the older formula. It is not claimed to establish actual causal actuation.

**Stationary H1(876queries):** native0.026704 versus ME0770.001701 and persistence
0.003983. This is a serious regression, about15.7times the old model's error.
Good movement forecasting must not be described as preserving stationary gains.

**Actual movement discrimination:** mean of five individual native100k-model
scores gives AP**0.61717** and ROC AUC**0.92603**, versus frozenME077AP0.16222 and
AUC0.63493. The same84positive/876negative true cases are scored. Native per-seed
AP0.59157–0.63115; GRU meanAP0.60129, Transformer0.61498. These small new-model
risk differences do NOT establish architectural superiority. H1Brier0.04478.
The primary JSON additionally includes ranking of mean probabilities across five
fits; that is an ensemble diagnostic, NOT a deployable100k-model score.
[RISK_INDIVIDUAL_SEED_RESULTS.json](RISK_INDIVIDUAL_SEED_RESULTS.json) keeps the
proper individual-fit mean definitions and every seed visible.

## Registered decisions

All three declared ME078 physical utility gates PASS: >=5%H1gain vs persistence
and ME077, >=5%moving gain vs best fixed physical reference, no task>10%regression
vs persistence. Those gates did not include a separate stationary guard; its
observed regression remains compulsory to report and prevents a clean gain claim.

**Learned-rival gates FAIL.** Aggregate horizon error relative to persistence:
native0.836938, GRU0.832223, Transformer0.843583. Native is0.57%worse than GRU and
0.79%better than Transformer; neither reaches the registered2%advantage.
Paired40episode adjusted97.5%bootstrap intervals for native-minus-control:
GRU[-0.006101,+0.014880], Transformer[-0.016668,+0.003401]. Both include zero;
seed-win conditions also fail. Full registered gate FAIL. No general architecture
winner, safety intelligence, independent transfer, action policy or customer ROI.

## What was built and controlled

The complete native model has32Dfast/32Dslow and8×8corrective associative state,
causal physical features, SwiGLU kinematic residual heads, ordered5/50/95%physical
quantiles and a separately supervised motion event. GRU56 and two-layer32Dcausal
RMSNorm Transformer receive the SAME observed inputs, prediction interface and
original targets. No privileged future actions or intervention flags as inputs.
Eight actual causal prefixes per window,1/4/16future targets with explicit
source availability/takeover masks, importance-weighted motion-enriched FIT
sampling. All learned modules are active and counted; no frozen learned encoder,
dummy padding or unused work to manufacture matching.

Parameters: native100158; GRU100494; Transformer99722, maximum gap0.774%.
Algorithmic charged update estimates117984772/117431798/118200482, gap0.655%,
with ZERO unsupported floating ATen operators. Independent torch.profiler agrees
exactly with the dominant matrix-operation subtotal (~96%of each update). This
is a **documented algorithmic scalar-operation estimate**, not hardware FLOP,
energy, wall-clock or foundation-scale parity. The original final model graphs
and live whole-model guard/steps were verified for all15fits. CPU is efficiency
metadata only. Profiling/source preparation overhead is separate from optimizer
training charges.

Five paired seeds78001–78005,2048AdamW updates×24windows, eight dense prefixes,
final-only checkpoint, no sweeps/restarts/DEVselection. Source and sample order,
importance weights and valid-target exposure match across each seed's three arms.
Dense counts are repeated **presentations**, not new independently collected data.
Finite task/component stratification differs by at most one proposal draw/task;
see the explicit nominal-importance clarification in[MATHEMATICS.md](MATHEMATICS.md).

Original licensed source and120FIT/40exposedREPORT split remain unchanged.
Future source disappearance or takeover is excluded per horizon and denominators
are reported, not counted as successful predictions. No CAL/TEST/protected
physical target or customer dataset opened. Only **eight-event window** physical
prediction is trained; persistent lifetime memory and causal control are unproven.

## Original evidence and replay

Prefit source/implementation/registration commit**dd824ffe**. All15original whole
model/optimizer/RNG/live-receipt checkpoints Git-frozen**0352faba**, BEFORE scoring.
Original all-horizon raw forecasts/targets/probabilities/intervals and scores
committed**8b1dfb8c**. All historical ME076/077weights and results unchanged.

Fresh remote clean Git clone restored and reproduced all15original checkpoints
and42,660forecast records within original rtol1e-5/atol1e-6, with no fitting;
original metric JSON also matched. [Local independent receipt](../../replay-bundles/ME078/INDEPENDENT_REPLAY.json).
**External GitHub strict replay FAILED**, first on native78001interval_score in
[run39999894647](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39999894647).
No tolerance, original model, original scores or training/scoring source changed.
The [complete external audit](../../replay-bundles/ME078/GITHUB_CROSS_CPU_AUDIT.json)
restored all15original models and checked every field of42,660forecasts. All
predictions, targets, MAE, risk probabilities/Brier, interval endpoints/widths
and coverage meet the original tolerance. **54of42,660derived interval scores**
violate it, maximum absolute interval-score drift4.57764e-5. Full strict replay
stays FAIL. Audit completion is not a successful strict contract; no threshold
was widened, original byte changed or exception hidden.

1152local regression tests PASS; scientific preflight/final parameter/data/
exposure/estimated-compute gates PASS. Reproducibility is distinct from utility,
architecture attribution and independent new-source predictive improvement.

## Next bounded integrated improvement

ME079 is separately preregistered to learn risk-conditioned smooth shrinkage in
ALL three families, cold starts and equal training opportunities, with an added
hard stationary-preservation gate. It does not erase this negative superiority
result or grant another hidden fit in ME078. Customer/operator value remains
UNKNOWN/NOT_MEASURED and all models remain research drafts without deployment.
