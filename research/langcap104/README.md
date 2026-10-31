# LANGCAP104 — Authorized training runner, currently BLOCKED (2026-10-31)

**Purpose:** Complete reproducible 1M/10M real-CPython next-byte FIT training for
the three scaled LANGCAP103 candidates, **only after** independent prefit and
resource authorization. This branch contains runnable training code but no
authorized training job, new optimizer updates or trained weights.

## Reused immutable preparation

- [LANGCAP102 PR #680](https://github.com/FPC-effortless/TAC-OSM/pull/680)
  authorizes *preregistered model sizes* (not training without provenance).
- [LANGCAP103 PR #681](https://github.com/FPC-effortless/TAC-OSM/pull/681)
  measured 1M / 10M TAC associative stack, multilayer causal Transformer and
  stacked GRU and authenticated a CPython 3.13.0 real-source FIT/DEV split.
- The original LANGMATH101 135/138 preregistered FIT record remains unchanged.
- LANGCAP103's two `REGISTRATION_DRAFT_{1,10}M.json` files are NOT training
  authorizations. Never rename one and train: they contain no verified FLOPs,
  no T4 quota, and no independent prefit evidence.

## Training contract in `train.py`

The first operation, *before any Torch import or corpus access*, requires:

1. A **new, frozen** `research/mathematical-engineering/langcap104/FAIR_COMPARISON.json`
   (single-model-training study with nine proposed fits for one selected scale)
   bound to the authentic source SHA, all planned arms/seeds, exact model
   parameters, approved optimizer, immutable order and explicit compute/steps.
   This file does **not** exist until legitimately reviewed and registered.
2. An operator-provisioned `LANGCAP104_FROZEN_MANIFEST_SHA256`, **signed
   independent** source/outcome/ACL prefit attestation file, and separately
   provisioned `TAC_OSM_REWARD_VERIFIER_PUBLIC_KEY_HEX`; the canonical
   research parity/reward gate must validate the signed data.
3. An actual measured T4×2 training profile, complete compute-accounting
   receipt, explicit permitted GPU minutes, and protected runtime variable
   `LANGCAP104_GPU_EXECUTION_ENABLED=true`. A public PR cannot grant itself
   these privileges. CPU/GPU quotas have not been certified.
4. The original FIT-only source archive prepared by an independent source-only
   job. The model process never receives DEV or TEST data. Every input chunk
   contains 257 consecutive actual bytes from one publisher file; the model
   sees the first 256 and predicts the next 256, never crossing file edges.

After the gates pass, the worker uses measured configurations, FP32 AdamW,
fresh seeds 10301–10303, 256-byte length and 32 samples/batch, with a fully
specified fixed optimizer. It attempts the approved 2,560 / 12,800 / 25,600
step milestones (subject to separately frozen budget), retains every loss and
a weights/optimizer/RNG checkpoint, and fails closed on nonfinite gradients,
step budgets and missing data. **Final model artifacts are still unqualified**
until a separate independently signed postfit assessment succeeds. DEV scoring
and architecture superiority are explicitly excluded from this worker.

The training runner has no access to a magic signing key or evaluator. A model
that trains successfully is not, without postfit corroboration, proof of
long-memory, abstraction, transfer or superior intelligence.

## Safe current CI checks

`.github/workflows/langcap104-readiness.yml`:
- Runs signature/budget/source/frozen-manifest rejection tests **without Torch**.
- Authenticates original CPython source, checks split/exclusions and stages
  only its **560 FIT files** in a separate artifact.
- Demonstrates zero DEV files delivered and **zero optimizer updates**.
- Never requests T4 GPU or starts a model.

## Current blocker

There is no independent attestation, no operator-pinned key, no PNDS/probe
qualification for the scaled architectures, no measured full-FLOP parity /
T4 throughput & approved minutes, and no final frozen trainable registration.
Those are prerequisites, **not discretionary labels**. The shared repository
protected Kaggle workflow likewise requires independent signed prefit evidence,
operator enablement and a dedicated approval in `kaggle-research`.
An unapproved GPU run would evade the explicitly binding scientific policy.

**Current scientific result:** NO LANGCAP104 TRAINING RUN HAS STARTED OR COMPLETED.
