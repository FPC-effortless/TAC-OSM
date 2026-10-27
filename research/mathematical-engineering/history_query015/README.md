# Frozen content-model historical query specificity

Question: after real historical-content training, does the native predictor
distinguish the requested old observation from an episode's general pollution
level? This follows the preserved negative classifier audit in PR #483 and
source-trained continuous-content models in PR #484. The latter already ran
the content-loss training; this study does not repeat that training.

## Frozen protocol

Original LATFUSE018 code `fd201207cb76b30470f64e1fc1baecf9de44560d`, four
original 99,390-parameter checkpoints, seeds18018/18019, and exactly its 117
original May–June2014 exposed FIT windows. Source is UCI501, CC BY 4.0; reuse
the recorded publisher/license receipt and original archive SHA. Original
checkpoints, optimizer/RNG archives and every prior result are unchanged.

At each original query time `anchor+64`, replay the actual owner observations
from `anchor-23` through query time. Both original architectures receive the
same `[batch,88,13]` feature-only histories, masks, ages, prior values and
source IDs. No raw targets or future feature rows enter the predictor.
Query eight actual past clocks `anchor+[0,8,...,56]` at this **same frozen
state**, without privileged writer keys. Each result is the unchanged native
continuous content head, not the old binary classifier's probability.

Primary H64 comparison: correct original clock error versus (a) mean error of
the seven other legal clocks and (b) the error of the target-free average
prediction across clocks. Require >=5% gain with positive station-bootstrap
95% error-gain bounds for both original associative seeds. Also require
positive within-window centered skill and >=5% centered-MSE improvement over
the original fast/slow readout control, with positive grouped error-gain bound.
Four thousand resamples, frozen seed15015, station units retaining all windows;
regional station dependence remains a limitation.

Report raw H64 error, original prior, zero-read and current-last controls,
wrong-query and collapsed-query errors, all observed query-target errors,
prediction/target range, missingness, and all negative results. Score only
actually observed original PM2.5; never fill missing targets with zero.
Compare scalar N=1 H64 inference to the first N=8 output within2e-6 and
reproduce original saved H64 MAE within2e-6 before interpreting new metrics.

The scoring-only centered metric removes each window's predicted/true mean:

`skill = 1 - mean_window(mean_observed(((p-p_mean)-(y-y_mean))**2)) /
                 mean_window(mean_observed((y-y_mean)²))`.

No true mean enters inference or a deployed baseline. Constant predictions
have zero centered skill even when they capture average episode pollution.
Missing/zero-variance cases are INCONCLUSIVE, not automatic PASS.

## Interpretation boundaries

Only H64 was supervised during original fitting. Queries at lags8..56 and
deliberately mismatched clock/target pairs are diagnostic distribution changes.
Failure cannot prove no stored information or incapacity to learn variable
lags. A positive result on this exposed FIT cohort cannot independently
qualify recall or architectural superiority. The practical question is whether
the *current checkpoints* show time-address-specific native content access.

This reuses LATFUSE014's same-window controls rather than claiming a new
retrieval method. LATFUSE018's data, active parameters, training order and
original registered-shape training-compute receipts remain as published.
The model graph/weights do not change. Full PNDS Lean/Python/IEEE/AD source
equivalence remains OPEN; local query-shape checks do not close it.
Full source-file-reload future poisoning also remains INCONCLUSIVE.

## Execution and evidence

```sh
python scripts/phone_lab.py plan research/phone_queue/history-query015.json
python scripts/validate_research_parity.py --manifest research/mathematical-engineering/history_query015/FAIR_COMPARISON.json --phase preflight
python scripts/audit_content_query_specificity.py --execute \
  --implementation-root /pinned/latfuse018-original \
  --source /authorized/PRSA2017-original.zip --output /new/path/RESULTS.json
```

Internal research, GitHub CPU, 20-minute bound, zero GPU budget. No training,
TEST parsing, independent certification or promotion. The dedicated read-only
workflow runs constructed software falsifiers separately from original-source
evaluation, then requires byte-identical replay in a fresh Python process.
All source/code/checkpoint/cohort hashes, cases and failed gates are retained.
