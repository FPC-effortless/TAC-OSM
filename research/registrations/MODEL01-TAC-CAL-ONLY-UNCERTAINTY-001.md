# MODEL01-TAC-CAL-ONLY-UNCERTAINTY-001 — frozen 103,252p native checkpoint reliability

**Temporal capability track • retrospective/exposed development diagnosis • no training or protected TEST.** Registered before this study's first calibration/DEV computation. Builds on original [trained and separately clean-replayed #502](https://github.com/FPC-effortless/TAC-OSM/pull/502). All three original source-checkpoint SHA256 digests are pinned in the implementation. Any corrupted source or original parameter tampering blocks the study. New study uses only publisher UCI374 original ZIP `2fccf354445d886e7917620b0195db1f3e3e34d5a067a93b844694a4c561255a` and original first-80% FIT/CAL/DEV, not the protected future 20%. Git-owned original neural weights/optimizer/RNG/sampler never change.

## Exact claim and candidates

**Question:** Can a fixed CAL-only uncertainty construction reduce the 90% forecast interval score of each already trained original native TAC model without worsened observed DEV coverage or worse point predictions? This is not independent model qualification: the DEV source and prior forecast/coverage information were already exposed, and all observations come from the same house. No formal marginal/exchangeability coverage guarantee under temporal correlation.

First retain original full-CAL symmetric empirical absolute-residual q for honest source-matched original checkpoint output. New alternatives are **only** time-independent CAL-derived post-processing offsets to the predicted center; no new learned weights, new training data, no future features, DEV-driven selection or individual household leaks. Original point predictions and MAE/RMSE must remain exactly unchanged.

Frozen **chronological CAL split:** first floor(0.6×3941) points estimate candidate offsets, next to floor(0.8×3941) select exactly once, last ≈20% is an audit that cannot influence selection. This produces nonindependent adjacent time fragments, so audit is development evidence only.

**Three predeclared schemes:** (1) original-form global symmetric offset ±90%-empirical absolute error from first CAL block; (2) global signed residual offsets at 5%- and 95%-order statistics, allowing asymmetric extreme tails; (3) four fixed 36-index daily phase groups of signed 5%- and 95%-empirical residual offsets. Phase is only deterministic source observation index modulo 144; no source future label. Every phase must contain at least 200 CAL-fitting examples.

**Fixed selection:** choose candidate 2 or 3 only when its middle-CAL **interval score improves by at least 1%** over first-block symmetric reference and middle-CAL empirical coverage is **at least 88%**. Among candidates passing both select smallest middle-CAL interval score, breaking ties by name. Otherwise use reference scheme 1. There is no DEV tuning or post-hoc threshold adjustment.

**Audit:** after candidate locked, report untouched final CAL 20% interval score, observed 90%-nominal coverage, width, below/above counts, alongside the first-block symmetric reference. **Exposed DEV:** after selection, evaluate the chosen source candidate and original **full-CAL symmetric** original checkpoint on *exactly same 1,968 source observations* for each of seeds 1701/1702/1703; report interval score, coverage, mean width and tail misses. An interval score improvement on DEV is accompanied by **coverage >=90% and nonworse point MAE/RMSE** to describe *development reliable improvement*. If scores improve but coverage falls under 90% classify MIXED_TRADEOFF; if no improvement, record FAIL and retain original frozen interval. The original q comparator gets more CAL data than the split candidate; disclose this asymmetry.

## Proper interval score

For 90% nominal alpha=.1, interval [L,U] and real observed y: `S=(U-L)+20*max(L-y,0)+20*max(y-U,0)`. Width and both miss directions are reported. Calibrated coverage cannot be guaranteed without exchangeability under time dependence. The experiment is not an off-policy control/causal calculation, transformer superiority result or source-available-at guarantee.

## Mandatory frozen boundaries and outcomes

Original model 103,252 trainable/deployed parameters stays unchanged; original checkpoint SHA256 by seed is retained; **no training**. All original future UCI TEST after index 15,787 unparsed, so do not claim independent skill or new site generalization. Original classical ridge interval score DEV 412.5287 remains unchanged; original native three interval scores DEV 447.2049, 440.3443, 422.9428. Any new alternative should not be called superior until actually scored. Same source has already been reviewed repeatedly and is development-only. Original predictions may be re-evaluated, but these cannot serve as blind evidence.

## Capability-to-Value Card

Capability tested: uncertainty postprocessing for real-data 60-minute appliance-energy forecasts while leaving point forecast unchanged. Practical reference: original frozen same-model 90% interval plus original same-source ridge; success metric 90% interval score with coverage floor. Utility hypothesis: fewer wide or missed consumption intervals could improve advisory energy operations; **no actual business value, bills, avoided costs, automated actions or paid customer evidence**. Clean-checkout replay uses same unchanged Git neural checkpoints, and complete original result should be preserved in Git with CI. Evidence tier: `EXPOSED_DEV_CAL_ONLY_CANDIDATE_DIAGNOSIS` regardless of sign.
