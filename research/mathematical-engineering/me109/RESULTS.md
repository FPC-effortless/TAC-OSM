# ME109 — parent-preserving residual native TAC: completed real-data study

Full registered development gate **FAIL**; complete seed passes **0/5**. All originals preserved; no promotion, merge, gate relaxation or extra fits.

Five104663-parameter jointly trained models,4096updates each, native recurrence and six families unchanged. Initial parental forecast/state preservation is verified, but is not a post-training improvement guarantee.

## Original evidence and scope

UCI501 Beijing Multi-Site, CC-BY-4.0, original archive SHA256 d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8.2013–2014FIT only; exposed2015REPORT,143natural-gap and4288general queries,6h-ahead,256h history. Protected2016–2017 numerical rows remain closed. Primary gap queries require current PM2.5age8–255h and an available future target; this is conditional offline forecasting, not all-outage safety. Seeds share the same labels and do not multiply independent observations.

Source452a1a977275823d21b3f04265a26ff836518ce6; frozen prefit6cbcd6c3. All5 original model/optimizer/RNG/plan/receipts were Git-frozen65e2f81791f0af048cc70a720374d8a7b89e0b45 before REPORT. All10original fitted FP32/64 graph/state/45parameter/input-gradient and read-only optimizer-proposal audits passed and were committed c8aed5ec before REPORT. No audit optimizer step.

Pre-training setup failures are preserved in PREFIT_STATUS.json. After fitting, an initial checkpoint staging attempt was refused by sparse-checkout rules; the fitted auditor then refused unfrozen files. The sparse scope was expanded, and identical original bytes were committed before any REPORT scoring; no fit was retried or changed.

## Five-seed measurements

| Metric | ME103 parents | ME108 prior candidate | ME109 | Change vs ME103 |
| --- | ---: | ---: | ---: | ---: |
| Gap PM2.5 MAE ↓ | 37.524168 | 41.132003 | 44.437162 | +18.4228% |
| General PM2.5 MAE ↓ | 33.403120 | 31.572055 | 32.014751 | -4.1564% |
| Gap six-channel marginal NLL ↓ | 5.151004 | 5.138193 | 5.127208 | -0.4620% |
| General six-channel marginal NLL ↓ | 4.944393 | 4.889856 | 4.891830 | -1.0631% |
| Gap PM90 interval score ↓ | 379.623199 | 241.566815 | 222.612326 | -41.3597% |
| Gap PM90 mean width ↓ | 369.909332 | 229.904205 | 197.248291 | -46.6766% |
| Gap PM90 coverage | 0.956643 | 0.966434 | 0.937063 | -2.0468% |
| General PM90 coverage | 0.898647 | 0.886707 | 0.891931 | -0.7474% |

Coverage rows above are fractions; point/width/interval score use original concentration units. Historical references are unmatched: ME10398495parameters, ME108102713, ME109104663 and different additional fits/seeds/plans. These are developmental system results, not attribution or architectural superiority.

| Seed | Gap PM MAE | Max allowed | Gap PM90 coverage | Full gate |
| --- | ---: | ---: | ---: | --- |
| 109001 | 42.419655 | 36.353154 | 95.1049% | FAIL |
| 109002 | 44.094875 | 38.074168 | 93.7063% | FAIL |
| 109003 | 46.436878 | 35.684479 | 91.6084% | FAIL |
| 109004 | 44.726963 | 41.476278 | 94.4056% | FAIL |
| 109005 | 44.507439 | 39.849239 | 93.7063% | FAIL |

Gate pass counts: `{"coverage": 5, "daily_point": 5, "gap_nll": 5, "gap_point": 0, "interval_score": 5}`. All5 must pass all original gates. No averaging into an individual pass.

## All-channel calibration, not only the primary gate

| Pollutant | General 90% coverage | Gap 90% coverage |
| --- | ---: | ---: |
| PM2.5 | 89.1931% | 93.7063% |
| PM10 | 91.6266% | 85.8741% |
| SO2 | 93.0778% | 82.6950% |
| NO2 | 91.5307% | 75.3623% |
| CO | 89.9087% | 81.8571% |
| O3 | 88.5206% | 90.0000% |

Only gap PM2.5 has a registered coverage acceptance gate. The complete table must still be assessed before claiming useful multi-channel uncertainty. Neither a PM-only pass nor a positive scale floor proves all-channel calibration.

## Update and verification boundaries

Actual updates20480; block checks266240; maximum computed applied point dot0.0; projected blocks94810; actual parameter-rounding rejected blocks24443. New-update CPU seconds1396.580641, excluding historical parent training and preprocessing; profiler FLOPs are partial and exclude optimizer/unsupported operators.

The guard protects the current-minibatch six-channel average, not each held-out channel/missingness regime or finite-step population loss. Seven limited new Lean identities/role/bound/count contracts and independent executable equations pass. Full neural/AD/IEEE/censored-law source refinement, useful associative retrieval, transfer, intervention intelligence, control and product safety remain OPEN.

Independent clean restoration and final replay evidence are recorded separately in the original bundle when completed. No further ME109 fits or changed thresholds are permitted.

Clean detached restoration at a68121d8 PASS: all5 original models,10 fitted graph/input/proposal checks,398790 saved quantile scalars, original scores/gates and summary replayed; no retraining, clean worktree. Full1087local tests PASS after scoring. GitHub original-score replay is pending at this receipt.

## Post-hoc FIT-only loss and regime trace; no new fit

A separate deterministic read-only audit used three original-augmentation B32,
256h FIT batches (16natural-gap+16general origins) for each of the five frozen
models. All candidate proposals used the original4096-step AdamW moments, original
objective/clip and original average-point guard; no weight or moment was applied.
All15average-point directional dots and195block dots stayed nonpositive. However,
gap-PM2.5 directional dots were POSITIVE in10/15whole proposals and75/195blocks;
general-PM2.5 dots were positive in0/15whole proposals and12/195blocks.

This is a concrete limitation of protecting an averaged task: it permits local
opposition to the gap-PM task. It is not proof of the cause of the held-out
regression, nor a measurement of realized finite-step losses. No REPORT/protected
rows were accessed by this audit. All weights, checkpoint hashes, model versions,
optimizer moment tensors and4096step counters were unchanged. The script and
original full gradient norms/pairwise cosines live beside the immutable results.
A task/regime-specific optimization contract is a proposed NEXT protocol, not
a changed ME109 loss, threshold or additional fit.

## Merged upstream anti-cheat and observatory tools adopted after fitting

PR417/420master tooling was restored byte-for-byte from
ada32393651614eeb60cf4f3a69f1aad44449e55; UPSTREAM_TOOLS.json records all15file
identities. Original training/model source and scores are unchanged. These tools
were integrated AFTER fitting/scoring, not claimed as retrospective preflight.
The native CLI targets PredictiveStateAgent, so a real ME109 adapter uses the
generic APIs rather than reformatting sensor data as fictitious native inputs.

All5 original final checkpoints are examined on genuine FIT only: architecture
inventory, actual module activation trace, weighted loss gradient atlas and
gradient health; one actual FIT origin per model has only its raw six-hour future
PM target poisoned before the original augmentation producer. Input-boundary,
that specific future-target invariance, deterministic forecast invariance, traced
noninterference and gradient-health findings PASS5/5; replay also PASS.

The overall anti-cheat ledger deliberately remains **INCONCLUSIVE**5/5: source
CSV/joins, full partition/preprocessing provenance, nuisance-ID/shuffle controls
and incompatible native-state tomography/dependency/entity/runtime adapters
are not verified by these probes. The poison holds normalization caches fixed;
it does not certify their upstream fitting. Runtime call order is not a tensor
DAG. These are no-fit engineering observations, not real capability evidence
or proof of no possible leakage. No optimizer was created or stepped; no
REPORT/protected dataset was loaded by this adapter.

GitHub original-score replay40280842012 and scientific preflight40280842069 PASS.
After adopting the15upstream tool files,64new tool/automation tests and full1151
local tests PASS. Additional adapter replay compares input/checkpoint identities
and finding statuses exactly, and requires exact clean-vs-poison output hash
equality within each run. Only four runtime prediction digest fields are
normalized across runs; all numerical summaries retain the existing1e-5/1e-6
contract. Saved original JSON/model/score bytes are not overwritten.
