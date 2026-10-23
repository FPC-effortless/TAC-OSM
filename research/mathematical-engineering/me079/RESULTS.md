# ME079 — better native forecasts versus persistence, no learned-rival advantage

Second bounded integrated architecture round;15new original real-data fits,
all cold starts. **MIXED_TRADEOFF; full registered gate FAILED.** Native forecast
accuracy improves, but stationary preservation fails and both matched learned
controls score better in aggregate. This does not establish TAC architectural
superiority, independent generalization, control or general intelligence.

| Genuine forecast task | Native TAC MAE | GRU MAE | Causal Transformer MAE | Persistence MAE |
| --- | ---: | ---: | ---: | ---: |
| Next frame,960query-target pairs | 0.060595 | **0.058553** | 0.059470 | 0.068668 |
| Four frames,956pairs | 0.185953 | **0.181330** | 0.188688 | 0.302400 |
| Sixteen frames,928pairs | 0.947424 | 0.937354 | **0.923996** | 0.975966 |

Means across five original fitted models, original recorded14-joint units.
Native improves persistence by**11.76% /38.51% /2.92%**, respectively. Next-frame
MAE is**8.85%below frozenME077** and1.02%below ME078. Each model is approximately
100kparameters; figures are averages of original model errors, not a deployed
ensemble of five models. Same40exposed episodes; horizon-specific future validity
is reported, never counted as a correct prediction when censored.

## What improved and what failed

- **Actual motion,84H1queries:** native MAE0.433934 vs persistence0.743241,
  clipped velocity0.482509, FIT coordinate-corrected servo0.709437 and ME077
  0.741981. That is41.52%below ME077 and10.07%below the best fixed physical
  reference, clipped velocity. It is3.03%worse than ME078's0.421182.
- **Near-stationary,876H1queries:** native0.024795 vs ME0780.026704, ME0770.001701,
  persistence0.003983. This modestly improves ME078 but remains14.6times ME077.
  The hard requirement to retain stationary MAE within10%of ME077 **FAILS**.
- **Real movement discrimination:** mean individual-fit AP0.60775/AUC0.91971,
  versus ME0770.16222/0.63493. ME078 remains stronger at0.61717/0.92603. New
  GRU meanAP0.62278; Transformer0.60767. No new native risk superiority.
- Three earlier physical utility conditions PASS; stationary preservation FAILS.
- **Learned-rival conditions FAIL.** Aggregate horizon-relative-to-persistence
  errors: native0.822702, GRU0.804255, Transformer0.812258. Native is2.29%worse
  than GRU and1.29%worse than Transformer. Native wins1/5paired seeds vsGRU,
  2/5vsTransformer. Paired episode-adjusted97.5%intervals for native-minus-
  control are[+0.007971,+0.029801] and[+0.001129,+0.020130]. They favor the
  controls on this exposed cohort; no native2%advantage was demonstrated.

All original gates, adverse strata and per-seed results are preserved in
[CAPABILITY_RESULTS.json](CAPABILITY_RESULTS.json). Uncertainty uses5000paired
bootstrap resamples of40episode clusters, seed79079, conditional on the fitted
models and exposed data. Session/operator/site independence is not established.
The within-study adjustment covers two learned controls; it does not turn the
second DEV-informed architecture recipe into independent confirmation.

## Integrated native architecture and fairness

ME079 retains native32Dfast/32Dslow/8×8associative dynamics, observed-motion and
causal command-error features, dense8prefix supervision and1/4/16frame heads.
It adds learned risk-conditioned smooth residual shrinkage; lower/median/upper
are shifted together to retain quantile order. Global risk is a separately
supervised event, not a promised joint physical distribution. Fixed-weight
shrinkage bounds do not guarantee performance after end-to-end retraining.
[Mathematics](MATHEMATICS.md).

Actual active/deployed parameters: native**100200**, GRU**100536**, Transformer
**99764**; maximum pairwise gap0.774%. All are within the binding95k–105kband.
Audited algorithmic scalar-operation gap0.652%; final profiles match the frozen
preflight graphs with zero unsupported floating operators. This is qualified
under the documented arithmetic estimation convention, not measured hardware
FLOP/instruction/time/energy equality or foundation-model-scale competition.

Each architecture received2048AdamW updates×24windows, identical real sample
order/importance weights, actual dense-target masks, loss and selection, five
paired seeds79001–79005, final-only originals. All learned modules and42new
scales trained. No warm starts, extra hidden encoder, sweeps, restarts or TEST
feedback. Both ME078 and ME079 recipe attempts were offered to all three families;
all30original fits and the earlier failures remain in Git. No additional fit in
this protocol is authorized, regardless of score.

Source: licensed pinned original BodenAI RW-RL-HIL160episode hash,120FIT and
40already exposedREPORT, same960H1origins as ME077/078. Eight past states and
lag1known commands; no future action or authority label as feature. Future
positions are factual logged outcomes, not identified do(a)effects. Input
missingness, long persistent memory, task success, permission and safety are
unqualified. Protected tissue-box/CAL/TEST stays closed; no customer or production
integration. Operator utility UNKNOWN/NOT_MEASURED.

## Preservation and reproducibility

Prefit registration/source/implementation freeze**5fe54ce4**, BEFORE optimization.
All15original model/optimizer/RNG/live-receipt checkpoints frozen**aabb05b4**,
BEFORE scoring. Original forecasts/targets/probabilities/quantiles and scores
frozen**d1c8e2ed**. Parent ME078 models and all older originals are unchanged.
Final live parameter/data/exposure/estimated-compute validation PASS; full local
regression suite**1156tests PASS**. Separate clean-clone and external numerical
replay records are maintained below; no blanket cross-CPU pass is assumed.

The original evidence supports stronger actual motion skills than ME077 and
persistence, while GRU/Transformer remain competitive or better. Neither recipe
is a fully qualified unified native winner: ME077 is still much better on
stationary periods. Keep the original capability portfolio separate; do not
silently combine multiple100kmodels and claim a single100kdeployment.

**Fresh remote Git clone replay PASS:** all15original checkpoints and42,660raw
forecast records restored and reproduced under original rtol1e-5/atol1e-6,
without refitting. The full original metric JSON matched as well.
[Independent receipt](../../replay-bundles/ME079/INDEPENDENT_REPLAY.json).
External strict CPU replay/audit is tracked separately; no passing status is
assumed from local success.

**External strict replay FAIL, preserved:** [run40001617818](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40001617818)
restored all15original models. Its [complete read-only audit](../../replay-bundles/ME079/GITHUB_CROSS_CPU_AUDIT.json)
confirms all prediction arrays, actual targets, MAE, probabilities/Brier, quantile
endpoints/widths and coverage meet the frozen contract. **29of42,660derived
interval scores** exceed it, maximum absolute difference6.10352e-5. The full
strict gate remains FAIL; no tolerance, model byte or original score was changed.
