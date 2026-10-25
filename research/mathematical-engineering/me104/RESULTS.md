# ME104 — asymmetric distribution implemented and trained; qualification FAIL

Fifteen warm-started complete models finished1,536new updates each. New architecture
has99,731trainable/deployed weights: the unchanged native memory/point backbone
plus a1,236parameter source-conditioned two-sided width head. All5ME103strong
reconstruction parents are used in every recipe. New run seeds control sampling,
not new independent root initializations. Fresh AdamW .0003/.001; old weights have
4,608lineage updates, new width weights1,536. No architecture-superiority claim.

Source/registration89d756d5, reciprocal links6021a05f and PNDS bindingd81bad75 were
frozen before optimization. All15original model/optimizer/RNG/data-plan/live-guard
artifacts were Git-frozen ata163708e BEFORE any REPORT. All original failures,
parent weights and scores remain unchanged. Exactly six active model families.

## Complete original exposed-development results

Five-seed means on unchanged143natural-gap queries, six-hour forecasts,
256hour histories. Lower MAE/NLL/interval score is better. Nominal90%coverage;
registered90–98%per seed. Each new row uses all5models, not an ensemble forecast.

| Model/recipe | Gap PM2.5 MAE | Gap PM2.5 coverage | Gap six-channel NLL | Gap PM2.5 interval score | Ordinary PM2.5 MAE | Full gates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original ME103 strong parents |37.5242|95.66%|5.1510|379.6232|33.4031|1/5 old gates|
| likelihood |36.9680|76.36%|5.1439|214.7926|35.4530|0/5|
| balanced |36.2121|74.41%|5.1359|222.5394|35.4438|0/5|
| point_guard |36.1055|74.27%|5.1253|223.8236|34.6475|0/5|

Mean gap point error improves1.48%,3.50%,3.78%; proper gap interval scores improve
43.42%,41.38%,41.04%; mean gap NLL improves slightly. However **ALL15models fail
the unchanged coverage gate**. Ordinary point error worsens6.14%,6.11%,3.73%
on average; only2of5seeds in each recipe pass the daily-point guardrail. No recipe
passes all5seeds, and0of15individual models pass every gate. This is not calibrated
predictive intelligence, overall capability qualification, safe uncertainty or a
reason to select one favorable seed. Mathematical coherence is not calibration.

The new protocol additionally requires at least10%per-seed interval-score gain
and stricter own-parent point/NLL guards. The full-gate count is therefore not a
matched pass-rate comparison with the parent's old gate, but every new model also
fails the original unchanged90%coverage floor. All raw per-seed gates are in
RESULTS.json and every channel/width tail/paired conditional interval is preserved
in PANEL_SUMMARY.json. Do not replace the planned decision with the improved mean
point or interval score. No threshold changes, retry or extra ME104fits.

## What the architecture establishes

Each channel has one proper censored distribution: equal probability halves of a
Gaussian quantile transform with separate positive scales, common median and
explicit zero atom. CDF, positive density/Jacobian and requested quantiles agree.
The new head reuses the existing source/evidence/memory read after the scan;
no new write/state or same-scan feedback. It does not discover actions, learn a
counterfactual policy, prove useful long memory or supply language intelligence.
These are per-channel laws at one horizon, not a coherent multi-horizon joint
trajectory or identified Bayesian posterior.

Zero head initialization reproduced every parent forecast exactly on real FIT
checks. Changing only the new head preserves the median at fixed backbone weights;
it does NOT preserve arithmetic means or trained point accuracy. All31parameter
tensors received finite nonzero real-FIT gradients. Joint training can change the
shared backbone. Learned side scales can be as small as one quarter of the old
scale: the inherited variance-floor amplitude is reduced to a conditional
old_floor/16lower bound. This is disclosed in the frozen protocol, not an old
floor-preservation or coverage guarantee. Source age remains capped256hours.

## Data, resource and numerical evidence

Authentic UCI501/CC-BY-4.0 archive SHA256
 d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8.
FIT2013–2014 only, real observed outcomes; training masks never invent labels for
naturally missing readings. REPORT2015was already exposed:143gap queries in21
calendar-week resampling clusters and4,288ordinary queries in53overlapping
clusters. Repeated seeds do not add independent real cases. Protected2016–2017
numeric outcomes remain unparsed/unscored. Source-row origins/labels/masks and
same parent query arrays were independently recomputed by the summary runner.

Per model:24,576new sampled examples,2,686,976history-event exposures; records
repeat. Same-seed initial weights/order/augmented-input hashes and postfit profiled
ATen graphs/dense forward-backward FLOPs match across the3recipes. Unsupported
pointwise/optimizer/hardware FLOPs remain unquantified; CPU is an outcome, not a
parity gate.897.529aggregate training CPU seconds; max2workers/one torch thread.
Old/new differ by1,236parameters, fresh optimizer, objective and training exposure;
this practical development comparison cannot establish architectural superiority.

All997local tests pass. Source-pinned PNDS source tests7/7 and six Lean scalar
proofs pass, with only standard core axioms. PNDS PR17source-math/build CI passes.
All15fitted networks pass30FP32/FP64 source/reference checks at unchanged original
tolerances: forward, state, read/key, width features, input/parameter gradients,
CDF zero atoms, censored NLL, default median quantiles and carried chunks. Largest
raw median discrepancy0.0008545original units reflects unit scaling and remains
inside the registered relative/absolute contract; standardized/state/gradient
errors are separate, not an absolute-error-below2e-6claim. No full-neural/IEEE,
Gaussian normalization Lean, statistical calibration, device-latency, customer or
safe-control proof is established. Original numerical evidence remains immutable.

A scoring-only runner edit raced the first no-step preflight; the source-hash
check blocked source freeze/fitting. Its artifacts are preserved in prefit-drafts,
and all profiles/gradients were repeated against final frozen source before any
optimizer update. No discarded trained model or report access occurred.

## Next integrated capability target

This fixed experiment is closed. Do not run another budget extension or choose
favorable seeds. A next separately registered candidate should limit uncertainty
updates from disrupting point learning, and learn/retain source-gap calibration
without allowing width flexibility to erase the missing-evidence floor. Differential
optimization/gradient roles plus an explicit source uncertainty floor are future
hypotheses, not proven explanations or fixes. Continue integrated capability
training, preserving negative evidence and deferring attribution studies.

## Original restoration

A separate clean detached checkout of6d684f07reloaded all15original models and
recomputed original forecasts/raw-density scores/gates, all30fitted numerical
audits, and the complete paired summary/publisher-row audit. No retraining or
altered originals. All67payload SHA256checks pass. Independent GitHub original
replay40195130742passed. The capability scorecard declares TRAINED_REPLAY_VERIFIED,
MIXED_TRADEOFF outcome and full-gate FAIL; its structure validator is not proof
of a predictive gain. Latest metadata-only CI is separate from this source replay.
