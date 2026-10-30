# AUTOCOMPLETE001 — actual original-checkpoint commercial-utility results (2026-10-30)

**Decision: FAIL / STOP native TAC-alone six-byte autocomplete hypothesis on this cohort.**

Original evidence: AUTOCOMPLETE001_RESULTS.json in this same directory; GitHub Actions first original experiment run 41021995664. 8/8 constructed tests passed before scoring and all original frozen checkpoint SHA/genuine Flask Git blob checks passed. No model fit, GPU, Android, protected TEST, production deployment, customer or monetary evaluation.

## Real source

Original Pallets Flask 3.1.2 at fixed Git commit 2c1b30d0503cfb064f1cb252e6614a06915a362a. Exactly three fixed original Python code files: app.py, blueprints.py and helpers.py; 512 genuine next-byte labels each (1536 unique labels). All five original neural seeds rerun across these same positions; repeated-seed measurements are not new independent source examples. Each model was offered 96 prespecified six-byte exact completions per seed, at fixed 16-character intervals. Oracle accepts correct suggestions for one Tab press, never accepts wrong ones; this is a theoretical upper bound and not user behavior.

## Observed result

| Predictor | Original learned parameters | Mean NLL down | Top-1 accuracy up | Oracle net saved keys |
|---|---:|---:|---:|---:|
| FIT-only P3 trigram | 0 | **2.616618** | **40.0391%** | 0 |
| Native TAC | 100521 | 3.115997 | 20.5729% | 0 |
| Selective diagonal SSM | 100404 | 3.155140 | 20.5990% | 0 |
| Tied head TAC | 100522 | 3.118833 | 20.2734% | 0 |
| GRU2 | 100661 | 3.279748 | 14.1797% | 0 |

Every one of 20 neural model/seed instances, and the FIT-only P3 count model, produced **zero exact-six matches** at the specified original offers. The predefined TAC advantage over zero-suggestion and P3 condition FAILS; neither native TAC nor P3 demonstrates useful six-character autocomplete as configured. No changing offer length or selecting favorable source offsets after seeing outcomes.

## Execution and resource context

The original TAC checkpoint uses 1,920 bytes of recurrent FP32 state and approximately 412,324 state_dict tensor bytes (including fixed buffers) per installed model. On the hosted CPU the mean across file/seed p95 single-observed-byte latency was ~0.659 ms (not Android); GRU ~0.066 ms; simple selective SSM ~0.071 ms; P3 table ~0.020 ms (table lookup only, in a different implementation). These descriptive process timings do NOT prove same-quality or optimized native device latency superiority.

The FIT-only P3 retains a much larger code-history count table (logical packed estimate ~255,986 bytes before Python overhead), so its memory is not equal to TAC's state. It nevertheless beats TAC's original tested next-byte NLL by ~0.499 nats/byte and top1 by ~19.47 percentage points on this independent project.

## Interpretation

A small neural model trained on CPython publisher *documentation*, then used for original Flask Python *code*, does not establish meaningful code autocomplete capability. This is a concrete negative result for an authentic code task; it does not invalidate prior prose-language comparisons or directly measure the preferred 500K graph-free LANGLAYER040 checkpoint.

Reconcile any future experiments prospectively:
- Next scientific test, separate: train/evaluate **actual code-domain 500K LANGLAYER040**, as well as strong tuned FIT-only P3 and token/code-specific baselines, with matched frozen data, budgets and checkpoint replay; no inherited superiority assumption.
- Next commercial job: choose a specific customer workflow, quantify user savings against cheapest reliable rule-based baseline, and get evidence of actual willingness to pay; the first paid pilot cannot be replaced by synthetic or oracle scores.
- Device gate: Woyengi #60 physical offline Android checkpoint export, real p95/memory/energy, privacy/no-network and saved state before any deployment claim.
- Strict research principle: DO NOT retrospectively tune six-character suggestion length or evaluate alternate source positions and call them qualification on this now-exposed Flask corpus. Register a new cohort and control selection before source access.

**No commercial value proved, no customer revenue or profitability proved. Preserve negative benchmark.**
