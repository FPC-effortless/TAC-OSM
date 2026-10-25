# ME099 — improved probabilistic forecasting, full gate still failed

**Full registered gate FAIL. Four of five seeds pass; seed99001 four-frame coverage is84.192562%, below the unchanged85%minimum. No further fits in this bounded correction cycle.**

| Recorded horizon | Persistence MSE | Candidate mean MSE | Reduction | Nominal90%coverage |
| --- | ---: | ---: | ---: | ---: |
| 1frames | 0.000493550 | 0.000299312 | 39.36% | 89.42% |
| 4frames | 0.007526495 | 0.004465788 | 40.67% | 85.77% |
| 16frames | 0.094477579 | 0.054845821 | 41.95% | 87.44% |

All five models pass every forecast, same-task parent-preservation and interval-score criterion. Coverage fails for the one seed/horizon above. Do not round it into a pass, select only favorable seeds, change the threshold or classify mean coverage as the all-seed decision.

Compared with the complete frozen ME098parents on exactly the same source windows, mean MSE falls1.14%/1.52%/2.08%at1/4/16frames. Proper interval scores improve20.75%/25.32%/21.71%. Mean coverage rises from81.01/75.86/76.20%to89.42/85.77/87.44%. These are descriptive adaptively studied DEV results, not independent or matched-training-compute architecture superiority.

**Trade-offs and unresolved weaknesses:** mean intervals widen44.97%/47.41%/52.60%versusME098. This repairs some underdispersion; it is not sharper intervals. Width audit is posthoc descriptive, no selection/gate change. Mean joint NLL improves from approximately-6.61269to-6.78332percoordinate but remains worse than the FIT-moment reference(-6.89786). Nominal90%coverage is still under target at most horizons, and the full gate fails. Long-history memory benefit, new-site transfer, action/control utility and otherfivefamilyheads remain unqualified.

## Actual architecture and mathematical engineering

103,380trainable/deployed weights: unchanged coherent trajectory core+58uncertainty-regime values. One conditioning pass computes native input-scheduled associative evidence and read-refined fast/slow features. Stable mean law composes under fixed coefficients; one response-Gram scale couples future horizons. One shared uncertainty regime combines narrow/broad joint paths; Cholesky is reused. Analytic Student4CDF, bracketedFP64quantile and independently checked implicit first-order derivatives. No extra persistent observed state, future neural unroll or seventh active model.

Both exact-real kernel proofs and executable full-source neural references are in PNDSdraftPR11. StandardLean axioms are audited; no sorry. FP32/FP64 source values/state/gradients/masks and scalar rootgradcheckPASS. All5fitted originals match independent serial equations on first2realFITwindows64frames under unchangedrtol1e-5/atol1e-6: maxpointdifference7.153e-7, maxquantile difference1.424e-6, maxgradientdifference1.193e-7. Quantile differences can exceed the absolute component while satisfying the combined relative/absolute contract. No wholeNNLean/IEEEproof or arbitrary hardware throughput claim.

## Original experiment provenance

Licensed source revision7a752b39f7e69de7e38aee485a6bab07528a061a, parquetSHAa3409d3282b2cf9a94d32b301756fbd15dca5862cc1c32973c7316dd8853c57a.59FITepisodes,13previously exposedDEVepisodes,104windows,64historypositions,1/4/16frame future targets.19,968window-targetpositions permodel; overlaps and5seedrepeats do not increase13independentepisodes. Last15%TESTwas not scored. Inputs x[t], recorded action[t-1], unchanged FIT-only normalization. Units are standardized recorded state, not certified SI or exact elapsed seconds.

Five originalseeds99001–5 use fixed corresponding ME098parents.512newupdates perseed,batch8. Full core lineage1,280updates, trajectoryhead1,024updates,new58values512updates. Two distinct five-fit studies were completed in this continuation(ME098andME099); no hidden retries. All10original models and both sets of originalscores/failures are retained. EarlierME096archive-order failure and ME095interruption are unchanged.

Originalmodel freeze12d72fde precedes scoresa2e58687. Complete weights/optimizer/RNG/schedules/preprocessing/live parameter receipts and rawmean/scale/regime/per-episode scores are inGit. Independent clean local replay and GitHubrun40166817604 PASS; no retraining in replay. Dedicated proof/source checks and PNDSfullbuildPASS.962localtestsPASS. Software/replayPASS never turns the predictive gate intoPASS.

## Next ideas; not implemented or qualified

1. A learned nonnegative future-noise envelope in the response-Gram factor: allow middle-horizon uncertainty to differ from constant-noise dynamics while preserving valid joint distributions. This is a hypothesis, not a proved explanation of the remaining failure. Qualify on genuinely new real data rather than repeatedly tune these13episodes.
2. Observed elapsed-time and entity-aware corrections: use authentic irregular timestamps and stable entity histories, with explicit observed-world versus imagined-computation separation. Current units remain recordedframes; padding tests are not real missing-observation evidence.
3. Reusable trajectory-noise sampling and square-root probability kernels: draw coherent alternative paths from one conditioned model without changing evidence, and evaluate QR/factor-space arithmetic plus chunked hardware kernels under the original numerical contracts. These are proposed engineering/capability extensions, not implemented autonomy or measuredGPUwins.

Capability-to-value: read-only physical future prediction. Enterprise integration, real operator outcomes, external pilots, money saved, action causal effects and general intelligence NOT_MEASURED. All PRs remain draft/unmerged.
