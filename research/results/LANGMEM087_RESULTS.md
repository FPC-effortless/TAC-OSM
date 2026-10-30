# LANGMEM087 stage A: more training improves language loss, not consistent slow-state usefulness

All six original continuations, nine first-new-document evaluations and strict aggregate passed. Internally measured, not independently qualified. No training equations or original checkpoints changed. Additional training alone did **not** establish useful old slow-memory content on this cohort. Bias, longer-context credit, stronger writes, dropout, auxiliary supervision and read-before-write retraining were designed as later stages and were **not run**.

## Primary results: one fixed fresh cohort

Three original100521p LANGMEM086 seeds, exact160→640→1280optimizer/data-order continuation. Same3584distinct genuine next-byte positions per checkpoint, two new Rust chapters, t256..2047 perowner; repeated across3seeds/3durations (32256predictions, not32256independent source targets). NLL in nats/byte, lower is better. The160baseline here is newly measured on087documents/windows; it is not directly comparable with086’s different-cohort2.6879NLL.

| Updates | Sampled FIT targets/seed | NLL | Top1 accuracy | ΔNLL removing old slow content |
| --- | --- | --- | --- | --- |
| 160 | 163,840 | 2.823832 | 25.04% | -0.001585 |
| 640 | 655,360 | 2.696131 | 30.11% | -0.001454 |
| 1280 | 1,310,720 | 2.717111 | 32.22% | -0.001769 |

Δ=intervenedNLL−fullNLL. Positive would support the original read; negative means removal helped. Sampled FIT targets include repeated source exposures, not unique new text. All durations were registered in advance;640was not selected as an early-stopped winner.

All three seeds improve NLL from160to640, then worsen from640to1280 (+0.022490,+0.015348,+0.025102 respectively). Accuracy continues to increase. This divergence does not identify a calibration or overfitting mechanism by itself.

## Attribution and measured pathway statistics

| Updates | Δfast_zero | Δslow_zero | Δboth_zero | Δold_slow_permuted | Δcurrent_slow_write_zero | Mean slow mixture | Mean slow write gain |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 160 | +0.050507 | -0.001571 | +0.057156 | +0.000106 | +0.000011 | 16.06% | 0.005363 |
| 640 | +0.177994 | -0.001470 | +0.185915 | -0.000090 | -0.000020 | 9.71% | 0.004468 |
| 1280 | +0.290594 | -0.001803 | +0.308855 | -0.001818 | -0.000063 | 14.35% | 0.003646 |

For fixed original priorstate, key, observed current value, query and gates, S_new=H+C, H=rho*S_old(I−eta*kkᵀ), C=eta*vkᵀ. old_slow_zero reads C only, retaining currentwrite; current_slow_write_zero reads H only (this is **not** S_old read-before-write). old_slow_permuted uses a cyclic value-row permutation of S_old in H and preserves original C. Every original matrixwrite/nonupperstate is bitwise unchanged. Counterfactual upperstate is discarded; next query uses original trajectory. NLL effects are nonlinear and need not add.

Fast-read dependence grows strongly, but those lesions include the current fastwrite and therefore do not prove historical fast-byte retrieval. Mean slow gains decrease; mean slow mixture remains roughly10–16%, not a zeroed branch. These are measured postfit DEV mixtures/gains, not equal-logit initial softmax illustrations.

## All seed results and positive exceptions

| Seed | Updates | Full NLL | Δold_slow_zero | Δold_slow_permuted |
| --- | --- | --- | --- | --- |
| 1701 | 160 | 2.845761 | -0.003343 | +0.000344 |
| 1701 | 640 | 2.756813 | -0.002136 | +0.001091 |
| 1701 | 1280 | 2.779303 | -0.001377 | +0.000982 |
| 1702 | 160 | 2.815620 | -0.000513 | +0.000162 |
| 1702 | 640 | 2.664074 | +0.000198 | +0.000522 |
| 1702 | 1280 | 2.679422 | -0.000595 | -0.000763 |
| 1703 | 160 | 2.810115 | -0.000900 | -0.000187 |
| 1703 | 640 | 2.667507 | -0.002422 | -0.001883 |
| 1703 | 1280 | 2.692609 | -0.003335 | -0.005671 |

At1280all3seed-average oldslow removal deltas are negative. At640seed1702 has a small positive +0.000198delta, while the other two are negative. Individualowner oldslow benefits exist:640seed1701/ch07 +0.000382,640seed1702/ch09 +0.000980,1280seed1701/ch07 +0.004365; other owner/seed cells do not provide a consistent benefit. Original full by-owner values are retained in JSON.

At1280, mean oldslow removal delta is +0.002053 over streamclocks256..511, +0.000467 over512..1023, and −0.003843 over1024..2047. These are **streampositions, not historical content ages**. This preserves small favorable strata without representing them as long-term recall. Two source owners and repeated targets are insufficient for broad iid significance claims.

## Autonomous querying and simple references

| Updates | FIT-only prior NLL |1928B ring NLL | Query-context top1 hits when a match exists |
| --- | --- | --- | --- |
| 160 | 3.408396 | 2.952908 |0/52 per seed |
| 640 | 3.413822 | 2.957271 |0/52 per seed |
| 1280 | 3.415586 | 2.959052 |0/52 per seed |

224logged query diagnostics/checkpoint,52matchingcontextcases; same sourcepositions repeated acrossseeds. No historical address/key/target enters model forward. Nearest original-writer-key cosine is only a4bytecontext proxy: sign, transformed keys and distributed coding limit negative interpretation.0hits does not prove no information. No auxiliary historical-byte decoder or factual-retrieval success was established.

No equally trained fast-only neural control was fitted. Consequently this stage does not meet all three proposed criteria for convincing slow-memory usefulness, and cannot establish clipping/architecture superiority. The ring is a storage reference, not a matched trained architecture.

## Source, training and cost

Original14PSF CPython FIT owners, original frozen sourceSHA911981cc63fd5436e0495cb067784e53475971131c811b7026e07d97a886d2d4. Exact sampled-window hashes in FIT_PLAN.json; original160checkpointSHA/optimizer lineage in PARENTS.json. AdamWlr.001,wd.01,globalgradclip1,4x256; no other change/tuning. Model resets at each training window, so extra updates do not add a >256byte training credit horizon.

Rust Project book Git2771df165f092a4fbb9f46b5b2e7450f19010adb, MIT OR Apache2.0. Gitblob/size and license metadata frozen before fitting, raw content acquired only after last1280seal. RawSHA256recorded afterward:

- `src/ch07-01-packages-and-crates.md`: 3806bytes, SHA256`7141297fcc8bd0d0cb4e850b9f853574ddbb15b48a8b56c615539979c8a9e27e`.
- `src/ch09-01-unrecoverable-errors-with-panic.md`: 8520bytes, SHA256`edce46b44eee961106ca49be8b1ce306e400c767304e9005b8763c141a444d1d`.

Different chapters from085/086, sameRustpublisher; limited new-document transfer fromPSF FIT, not broad independent generalization. No rawSHA was falsely represented as independently frozen before fit.

Additional fit processCPU6417.143sec=106.95CPUmin across6continuations; evaluation processCPU1041.950sec=17.37CPUmin across9evaluations, below conservative180+75CPUmin budgets. Different batchlosses are not heldout-learning curves. All optimizer steps and finite connected gradients checked; all35top-level parameter groups have nonzero finite norms at recorded segment endpoints.

Persistent model tensors1920B/owner+8Bclock+UTF8ownermetadata; shared learned weights402084B, fixed codebuffer8192B. Ring1928B=16Bheader+1912bytepayload, plus2048Bsharedfloat64FITprior; transientcounts/workspace separate. Ring reaches capacity late in2048observedsteps. No total-memory Pareto claim. Peak evalRSS344510464–345436160B.

Diagnostic-wrapper fullforward1.192–1.827ms/observedbyte; includes explicit split/trace bookkeeping. Ringquery2.532–3.375ms/query, excludesappend; workunits differ and neither is a hardware deployment or equalcompute efficiency comparison. Actual timings and percheckpointcounts retained.

## Engineering, mathematical scope and original artifacts

All actual100521prefit/final gates and original parent/sampler/optimizer chains passed. All six new continuations passed cold bitwise logits/state and optimizer-step reload checks; the original160receipts preserve their original logits-replay checks. Original diagnostic default matches native086 graph bitwise for every evaluatedcheckpoint.18actualsourcefuturepoison pairs passed; maximumstream/bulk128prefixerror1.144409e−5<3e−5; old/current spliterror1.490116e−8<1e−6; weights frozen throughout. Unsupported diagnostic dimensions remain INCONCLUSIVE.

Full local suite1602passed/4skipped after scoped086import;34targeted checks after both038/086scopes passed. First import-collision sourceCI40983854617 and first proofCI40983914975 failures preserved in ENGINEERING_LEDGER.md. Corrected scoped Lean/Python CI40984197814 and bridge40984526086passed. Actual source Float64 split/read/localderivativeerrors6.78e−21/4.34e−19/1.74e−13. Full graph IEEE/AD, semantic goals, independent reward/ACL qualification and canonical registry merge remain OPEN.

Original fit + first-new-DEV + aggregate Actions: https://github.com/FPC-effortless/TAC-OSM/actions/runs/40984392435 . Registered source9d6cda3be8e200fd607bf6f8bb39fa381587bc22; explicit launch535fbfafe6202f5a8aaf53629150372e60bd7cb1. Draft TAC630, PNDS84. Aggregate artifact12948075104, ZIP SHA2569338ad3c168bd2aacad2b5f63883a9f67b0be8fe9167bb7ce6cf92226dd7e0d3. All original artifactIDs/ZIPdigests in LANGMEM087_ORIGINAL_ARTIFACTS.json, all checkpointSHAs in aggregate.

Independent workspace download verified all3evaluation ZIPs and aggregate ZIPSHA; all9actualMODEL.ptSHA checks and original parent chains passed. Read-only reaggregation is exact JSON equality to both original artifact and original Actions log; no further fit, inference or DEV text read was required. See REPRODUCE.md. Normal latercommits skip fits/DEV; no restart to improve outcome.

## Research decision

More updates alone do not resolve slow-memory usefulness in this tested256byte training task. Keep the existing architecture and1280negative evidence. Proceed prospectively to a separately matched credit-horizon and/or trainable slow-read preference study with fresh DEV and a fully trained fast-only comparator; do not tune on087chapters. No claim that larger training scales, different tasks, slow memory in general or alternate read/write alignment cannot work. Original models/data/failures preserved, no GPU quota, protected TEST, model promotion or merges.
