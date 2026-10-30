# TAO002 — genuine publisher language-model optimizer qualification

**Status: preflight-only, NO NEW FIT / NO optimizer-quality result.**

Source lineage: TAO001 draft [PR #641](https://github.com/FPC-effortless/TAC-OSM/pull/641); track [Issue #642](https://github.com/FPC-effortless/TAC-OSM/issues/642). This separate branch is stacked on PR #641 and the historical checkpoint-replay draft. Existing model versions, reference scores, protected TEST and original checkpoints are untouched.

## Research question

With an **identical 100,521-parameter native TAC language model**, does one of four explicitly routed orthogonal-update options improve real publisher-byte next-token loss, calibration, or *time to fixed quality* over AdamW when the original legal training source, initialization, token order, optimizer training/tuning budgets and evaluation definitions are held constant?

The central scientific claim is conditional and **not presently demonstrated**. External papers showing optimizers working at much larger scale do not establish a gain for 15,040 FFN-up weights inside a small stateful TAC model. The optimization algorithm does not change the inference architecture or persistent memory payload.

## Frozen source/model scope

- Native model: `research/mathematical-engineering/langmem086/compact_tac.py` (original source SHA256 `7179908fa3672879f3c2046ec4521f0b9f783d5c696aaba82c4c2cd0b6ebc9b1`); exact active and deployed parameters `100521`.
- Original **14** PSF CPython 3.13.0 publisher FIT byte documents, previously registered original SHA256 `911981cc63fd5436e0495cb067784e53475971131c811b7026e07d97a886d2d4`; original source/publisher receipt: `research/mathematical-engineering/language_source023/SOURCE_RECEIPT.json` plus UTF8 supplement receipt. Source copyright retained; no protected data.
- Seeds `1701..1705`; identical per-seed model RNG/init and original sampler; first three seed schedules must match unchanged `langmem086/FIT_PLAN.json`, last two separately frozen after actual source preflight. Provisional planned **160 updates**, B4, 256 original input bytes (163,840 next-byte labels per arm and seed); total **25 proposed fits** (5 optimizers × 5 seeds).
- Five fixed modes: `adamw` (original PyTorch control), `muon`, `adamuon` (published sign(momentum), not Nesterov sign), `normuon`, `pre_ns` (unvalidated hypothesis). Orthogonal parameter name `ffn_up.weight` only, shape `94×160=15,040` elements. All other weights (including embeddings, output head, norms, gates, both recurrent banks and associative memory) remain AdamW.
- Development only: choose a **previously exposed** frozen genuine publisher DEV subset before fitting and disclose repeated exposure. Never read sealed TEST, leak next-byte target into inputs, or tune on a fresh protected cohort. Real publisher texts are not synthetic tests.

## Phase 0: actual original-source no-fit preflight

`python scripts/tao002_prefit.py --output /tmp/TAO002_PREFIT.json`

This validates original source file hash, loads genuine FIT owners with license/hash validation, checks real frozen sampler schedules for five seeds, checks actual model count/source hash, five optimizer constructors and exact FFN route, and confirms optimizer creation does not change initial weights. It makes **zero** optimizer steps; neither reads DEV nor guarantees the future fair-comparison gate passes.

Automated no-fit push CI: `.github/workflows/tao002-prefit.yml` uses only standard GitHub Actions CPU, `permissions: contents: read`, and publishes an expiring source receipt. It has **no training workflow job**.

## Phase 1: register genuine comparative experiment BEFORE first step (not yet done)

Add `research/mathematical-engineering/tao002/FAIR_COMPARISON.json` with the original committed Phase 0 receipts and required comparison schema, exact five-seed sample hashes, frozen DEV IDs/target hashes, architecture-equal training FLOP measurement method, LR schedule, budget and identical optimizer tuning opportunity. Run `scripts/validate_research_parity.py --manifest ... --phase preflight` on actual live models; refuse fit if FAIL.

**Technical caveat:** optimizer FLOPs and wall-clock cost differ by definition between AdamW and NS5. Forward/backward model FLOPs can be fixed; report optimizer FLOPs separately and accurately. Do NOT invent equal total optimization FLOPs or claim an architectural-superiority gate if measured compute violates the preregistered tolerance. The purpose is an optimizer Pareto comparison under identical model work/tokens, with measured optimizer overhead.

The final protocol should reserve equal LR-trial count for each mode. A zero-tuning, shared-LR first pass is a useful **pilot**, not enough to choose the global best optimizer. Never cherry-pick only positive trials.

## Phase 2: genuine training/score (not yet done)

Only after a committed PASS preflight, run source-validated CPU jobs within documented free budget. Each completed arm/seed must save original in-Git full model+optimizer moments+sampler/Python/NumPy/Torch RNG and source/plan/checkpoint code hashes. Independently clean-checkout reload/score and final resource receipts are mandatory. A GitHub Actions temporary artifact alone is insufficient.

Compare matched-seed heldout DEV next-byte NLL, top1, Brier; fixed-target learning curves; full forward/backward/optimizer seconds and optimizer-only cost; peak RSS and state bytes; nonfinite/slow/failed seeds; relative paired differences with uncertainty. The only actual initial speed result so far is synthetic per-matrix CPU update timing from TAO001, NOT training benefit.

Source-time future suffix poison, source-doc partition, original optimizer AdamW parity, checkpoint restore and sample-order controls remain required. The optimizer is a learn-time module; inference / phone-memory speed and correctness must be evaluated separately.

## Mathematical engineering / cross-architecture eligibility

A preliminary Python formula is in `optimizer_tao.py`; scoped Lean+Python PNDS source-pinned equivalent and finite-precision tests remain **OPEN**. Do not claim full mathematical verification. Eligible hidden linear matrices must be named explicitly in each architecture. Conv kernels may be flattened [out, in*k*k]; recurrent state transitions, specialist gating, embeddings, LM heads, and tied output weights remain AdamW absent new evidence.

**References:** [Muon scaling](https://arxiv.org/abs/2502.16982), [AdaMuon](https://arxiv.org/abs/2507.11005), [NorMuon](https://www.microsoft.com/en-us/research/publication/normuon-making-muon-more-efficient-and-scalable/). This implementation is not claimed bitwise-identical to paper reference codes. BF16, Triton fusion, distributed sharding and asynchronous execution are not qualified.
