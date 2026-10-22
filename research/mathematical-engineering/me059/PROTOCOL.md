# ME-059 — independent real robot training, prospectively registered

**Preregistered 2026-10-22 before original ME-059 fitted model updates.**
No claim of real-world predictive superiority unless actual original outcomes, paired budgets, sealed cohort access and independently valid FLOPs qualify. This program uses a new real task, not the exposed earlier ME-054 task.

## Real source and historical source-only evidence
- Publisher: BodenAI/RW-RL-HIL-Dataset, pinned `9290f5a96dc91ea58256e81d9ce2dd9621890b51`, task `clean_stains_off_tabletop`.
- Metadata-only original split: 228 authentic FIT episodes, 76 DEV and 80 TEST grouped by four adjacent episode IDs before hash allocation; publisher provides no verifiable operator/session IDs, so split independence beyond block separation is unproven.
- Independent genuine original FIT receipt: sha256 `c4064651e91cced7aafc11337c32a27266f470ea10388bb0dd2562ae955f88fd`. Full 228 FIT files already audited in ME-057 and independently rechecked in ME-058.
- ME-058 GitHub Actions successful [run 39840368795](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39840368795), machine-readable original artifact ID 12452514575. **Corrected pre-feedback** authentic real two-arm optimizer profiler: TAC 38,266 parameters and 42,257,730 reported operator FLOPs/update; GRU 38,295 parameters and 42,546,844 reported FLOPs/update. Gaps 0.076% and 0.680%, both below registered 1% budget. CPU times 0.873 s versus 0.648 s for one profiling update are recorded only as efficiency telemetry. Those are profiler-instrumented first updates, not production throughput comparisons.
- Exact pinned FIT-only physical prior sha256: `3d66d00ba01f345d7fcf3c23149f808986da795b1dea07ead4520b8bafd8af96`.
- Study registration: `FAIR_COMPARISON.json`, v3. Six immutable feature/target/optimizer/selection/evaluation source texts and their corresponding SHA256 live in `CONTRACTS.json`.

## Controlled training
- Arms: anchored 128-frame slow-consolidation TAC; fully trained GRU world model. Both share original physical state/action/authority streams and FIT-only servo prior; no dummy/dead trainable parameters.
- TAC additionally trains 128 historical physical-transition auxiliary outputs from **prefeedback** recurrent memory and the same original sensor/action windows. GRU observes those source windows under burn-in but does not optimize that dense auxiliary. Hence the comparison tests the full TAC architecture **plus its declared auxiliary training objective** against GRU; no isolated causal attribution to memory alone is permitted.
- Exactly five seeds `58001, 58002, 58003, 58004, 58005`. Separate fixed FIT sampled-window-order SHA per seed is frozen in the source plan and independently reverified at runtime.
- Each model gets 600 AdamW steps, LR 0.001, weight decay 0.0001, global gradient clip 1.0, batch size 2; 256 real observed burn-in transitions per update plus a 16-step real logged-action-conditioned forecasting objective. No DEV/TEST pre-fit access, no tuning trials, no early stopping; identical checkpoint selection at step 600.
- Per seed/arm: 1,200 real original sampled windows and 326,400 physical state-time exposures, including observations and supervised forecasting. Same source fields/order and budget across arms; the treatment differs explicitly in native auxiliary targets.
- Profiled operator compute/update from genuine original FIT differs by only 0.680%; approximate planned 600-update operator counts: TAC 25,354,638,000 vs GRU 25,528,106,400 under common ceiling 25,528,106,400. These are **partial profiled operator estimates** and do not independently certify complete true training FLOP consumption. The postfit v3 validator must return `PASS_ACCOUNTING_ONLY_UNCERTIFIED_FLOPS` and `eligible_for_architecture_superiority=false` unless fully supported future receipts establish otherwise.
- CPU process time is reported, **not** equalized and never used as a fair-comparison gate.

## Factual heldout evaluation, only after freezing 10 authentic model checkpoints
- Score genuine physically measured 14D future joint-state conditional on the *logged* actual 16-step action sequence, from preregistered origin frames 256, 512 and 768 when robot autonomy is initially observed. Censor all physical scores from the first human takeover onward. This supports conditional physical forecasting only, **not** model-planned action effectiveness or off-policy causal effects.
- Score original physical-unit pooled and horizon-wise MAE for TAC, GRU, zero-trained last-state persistence `x_t`, and the common FIT-only servo prior without a learned residual.
- Report original real DEV separately from original TEST, all five original seeds, episode counts/valid horizon steps, actual takeover-onset support and seed variability. Hold every original TEST label sealed until all ten model weights/hashes and postfit resource receipts are frozen.
- Claims: real predictive outcomes are genuine but full computational parity remains uncertified; results are **developmental**, never an independently certified architecture winner. Do not promote if persistence or servo prior wins. Operator/session split ambiguity and logged-action conditioning must remain explicit.

## Prior research reuse
- R2D2: recurrent burn-in, https://openreview.net/forum?id=r1lyTjAqYX
- DreamerV3: learned physical world dynamics, https://doi.org/10.1038/s41586-025-08744-2
- TD-MPC2: action-conditioned multi-step prediction, https://arxiv.org/abs/2310.16828

The unresolved question is TAC-OSM's actual incremental predictive value on authentic independent physical sequences, **not** the already-known existence of recurrent dynamics.

## Fail-closed obligations
1. Check actual frozen code paths against ME-058 git-blob SHA before first fit.
2. Verify original source metadata, all 228 original FIT physical-source bytes and FIT-only scaling SHA.
3. Check original full v3 manifest and exact frozen sample schedule hash for each seed; stop on mismatch.
4. Do not open DEV/TEST before 10 real models' checkpoints are hash-frozen and postfit budget receipts checked.
5. Publish every negative finding and original checkpoint hash. No simulated performance, imaginary labels, fabricated cost or policy value.
