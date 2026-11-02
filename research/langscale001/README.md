# LANGSCALE001 — controlled TAC memory-budget reallocation

**2 November 2026. Implemented engineering candidates; no new training or DEV/TEST scoring.** Follow-up to the LANGCAP105 diagnosis in [PR 694](https://github.com/FPC-effortless/TAC-OSM/pull/694). Original checkpoints, scores, source and evaluation registrations are unchanged.

## Question and controlled designs

Does redirecting matched TAC parameters toward temporal state improve real-code fitting as model size grows? The original TAC carries 768 nonlexical floats at either scale. The new designs preserve its update/read equations and eight-layer depth; they do not introduce a delta-rule algorithm, new training objective or new attention mechanism.

| Scale | Arm | Width | c / k / v per block | Nonlexical state floats | Trainable/deployed parameters | Head width |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| 1M | Original TAC | 256 | 16 / 4 / 8 | 768 | 1,000,028 | 246 |
| 1M | Head-to-state TAC | 256 | 24 / 6 / 12 | 1,536 | 999,652 | 106 |
| 1M | Memory-scaled TAC | 232 | 32 / 8 / 16 | 2,560 | 1,000,360 | 152 |
| 10M | Original TAC | 992 | 16 / 4 / 8 | 768 | 9,999,922 | 465 |
| 10M | Head-to-state TAC | 992 | 32 / 8 / 16 | 2,560 | 10,000,888 | 116 |
| 10M | Memory-scaled TAC | 856 | 64 / 16 / 32 | 9,216 | 10,000,942 | 571 |

The head-to-state control keeps token width and depth unchanged, trading active head parameters for larger c/k/v. It tests that **allocation**, not a literally isolated state-size variable with every other tensor fixed. Memory-scaled TAC also changes width. Their contrast is informative but cannot uniquely decompose every c/k/v/head/width interaction. No unused weights pad the counts.

Fresh original-geometry GRU and Transformer arms are declared at both sizes. The largest parameter gap across all five arms is <0.3% at 1M and <0.14% at 10M. The frozen proposal is five arms x two scales x three paired seeds = 30 fits. No fit has begun. Constructor geometry selection uses counts alone: for memory-scaled TAC, the largest multiple-of-eight width with core <=90% of target; c/k/v are declared before any new language scoring. Existing exposed DEV scores do not choose the new widths.

## Implementation and mathematical correspondence

`models.py` inherits the original source's entire forward function and uses its original `AssociativeBlock`, `BudgetHead`, ownership and state-validation code. Only constructor geometry changes. Tests show that default c/k/v reproduces the original seeded initial parameters byte-for-byte at both scales. Historical files are not edited.

`reference.py` is the existing independently unrolled full mathematical Python reference from source helper commit `0b21f0feed5d6f89bb49218d8ca320cca6220850`. It already infers c/k/v from named parameter shapes. It never calls a model forward/GRU/attention module; Torch primitives are shared. The prior PNDS `LANGCAP103` exact-real graph is dimension-generic and supports specialization to these geometries. A reciprocal PNDS specialization and source registry are maintained separately; Lean compilation, bounded numerical correspondence and global IEEE equivalence are distinct statuses. No proof is silently inherited as trained capability.

`verify.py` checks FP32 and FP64, batch two, singleton/odd/longer/128-byte sequences. Checks include independent logits and every final-state tensor, all parameter gradients, nonzero carried-state gradients, parallel/serial agreement, chunk carry, nonzero-clock reference agreement, future-byte poisoning, entity isolation, wrong-owner rejection, wrong-readout rejection and unchanged parameters. Inputs are constructed fixtures. `check_probes.py` registers actual-source adapters for both new variants and tests query purity, arbitrary state views, dependencies, entity isolation and a deliberately contaminated forecast.

## Source and training/evaluation boundaries

The original Actions FIT-input ZIP was downloaded and its SHA256 independently checked (`dba50c1b32747cc898f77225f64b8c53c94bc2e6fa983c7b3937e03782bfe8a1`). `authenticate_fit.py` authenticated all 560 original FIT files against original publisher identities/hashes and reconstructed all three exact 2,560-update coordinate streams. It did not pass any corpus bytes to a new model, score labels, access DEV bytes or perform optimizer updates. The original receipt's DEV metadata is present and explicitly identified as metadata only. `evidence/SOURCE_ORDER.json` records actual FIT partition and order digests, not a new trusted signature.

`PROTOCOL.json` retains the original fixed AdamW recipe, 128-byte reset chunks and 2,560 updates to isolate allocation effects; zero tuning trials. A later tuned-recipe or carried-state campaign would require a separate registration. Every future milestone must be preserved as a distinct checkpoint with optimizer/RNG/order index rather than overwriting LATEST. Original LANGCAP105 weights are diagnostic references only, not resumed fits for the new arms.

The original exposed DEV remains archive-only. New validation and sealed-TEST source contracts remain OPEN. No arbitrary hash stands in for missing evaluator evidence. Both `FAIR_COMPARISON.json` drafts therefore contain explicit null new evaluation digests and full FLOP fields, and canonical preflight **rejects** them. `readiness.py` records the real exit/error and the separate missing independent-attestation failure. A green no-fit workflow means only its engineering/audit commands succeeded; it never authorizes or launches training.

## Compute audit and execution status

A full-batch 4x128 forward/backward profile was executed for all ten configurations, without targets or optimizer updates. Covered kernel FLOPs are similar: at 10M, original TAC ~29.860B, head-to-state ~29.873B, memory-scaled ~30.030B and GRU ~30.108B. These are **partial profiler totals**, not full training-step FLOPs, compute parity, speed or capability. The profile's objective is squared logits, not byte CE/AdamW. Uncounted operations and optimizer work remain explicit. Do not replace null full-FLOP registration fields with these totals.

The initial profiler command failed because the original script filename `profile.py` shadowed Python's standard-library module. It was renamed `profile_compute.py`; the retry completed. The failed invocation is retained in the evidence ledger. This tooling defect did not modify a historical model or explain its language loss.

The phone compute planner routes the proposed long campaign to **two physical NVIDIA T4 GPUs**, with an upper proposal budget of 240 GPU-minutes. This is a ceiling/proposal, not consumed quota or an approved run. Private offline source staging, a reviewed dual-GPU workload, an authorized dataset reference, full compute receipts, new evaluator contracts and independently signed source/reward/ACL preflight evidence remain absent. The protected execution environment has not been exercised here. No Kaggle training job was submitted; no alternate accelerator was substituted.

[AGENTS.md](../../AGENTS.md) requires: **“Missing receipts or failed gates block fitting.”** The study remains draft, with no trained-improvement or scientific-superiority claim.

## Reproduction and next executable boundary

From repository root, with the original pinned Torch CPU version:

```bash
python research/langscale001/verify.py --out /tmp/LANGSCALE001_SOURCE_TESTS.json
python research/langscale001/check_probes.py --out /tmp/LANGSCALE001_ADAPTER_TESTS.json
python research/langscale001/profile_compute.py --out /tmp/LANGSCALE001_PARTIAL_PROFILE.json
python research/langscale001/readiness.py --out /tmp/LANGSCALE001_READINESS.json
python scripts/phone_lab.py plan research/phone_queue/langscale001-nofit.json
```

To regenerate authentic order receipts, supply the original FIT-only artifact to `authenticate_fit.py --inputs /absolute/FIT-inputs --out /tmp/SOURCE_ORDER.json`. Preserve the original ZIP/checkpoints read-only. Registration regeneration is count-only; `register.py` never creates a source/evaluator attestation.

Before fitting: complete the independently reviewed new validation/TEST contract and full-compute instrumentation, attach authentic source/order receipts and signed evidence, implement/review the private dual-T4 training workload, run the full canonical preflight, and record actual venue/quota authorization. A fit must fail closed if any gate remains unmet. Only after all fits and independent checkpoint/replay checks should a separately registered evaluator score the fresh cohort. No old DEV rescoring is needed to debug the new models.
