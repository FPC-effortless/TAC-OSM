# LANGCAP103 — scaled code learnability and resource preflight

Status: proposed study, engineering preflight only. No new model has been trained.
This is separate from historical LANGMATH101 and the LANGCAP102 policy amendment.

## Question and candidate roster

Can adequately sized, trained TAC learn authentic code prediction competitively
with credible neural competitors, before attempting long-document reasoning?

Two candidate cohorts target approximately **1M and 10M** trainable/deployed
parameters. All inputs and outputs use the same 256 raw-byte vocabulary.
The no-fit preflight selects geometry using parameter counts only, constructs
actual FP32 models, checks every parameter has a finite backward path and that
future bytes do not change earlier outputs, and verifies <=1% parameter gaps.
It never optimizes a parameter, opens a dataset or judges capability.

| Arm | Implementation | Interpretation |
| --- | --- | --- |
| TAC associative stack | Eight existing associative blocks from source-pinned LANGMATH101; residual width and genuinely used decoder capacity scale | Candidate native stack; fixed existing bank geometry is disclosed, not a claim of optimal TAC scaling |
| Causal Transformer | Four layers at ~1M, eight at ~10M; independent pre-norm attention/GELU blocks, eight heads, sinusoidal position encoding | Standard multilayer scratch reference; no pretrained frontier claim |
| Stacked GRU | Four native PyTorch GRU layers | Established recurrent comparator; not a Mamba/SSM reproduction |

The native modules in `vendor/` are source-identical copies pinned in
`SOURCE_PINS.json`; historical source and evidence remain unchanged. The
preflight uses the existing parameterized stack implementation and its real
gated readout, with at least 70% of candidate parameters outside the readout.
No dummy weights are introduced. Graph-matched PNDS and probe qualification
for these exact scaled instances remain OPEN before training.

Run `python scripts/phone_lab.py plan research/phone_queue/langcap103-prefit.json`
then `python research/langcap103/preflight.py --out ci-artifacts/langcap103/profile.json`.
The automatic no-fit workflow publishes measured configurations, exact counts,
FP32 weight bytes, recurrent state bytes, and an analytical parameter/gradient/
Adam-moment memory subtotal. Activations, workspaces and caches are excluded
and identified. These are not measured training FLOPs, trained decode speed,
GPU feasibility or a training authorization. The Transformer currently has no
KV-cache implementation; inference efficiency comparison needs that work first.

## Draft training design — not yet a frozen FAIR_COMPARISON registration

1. Authenticate a licensed real-code training corpus and freeze file/repository
   splits before optimization. CPython and separately owned repositories are
   candidates, not accepted source receipts. Do not train small scratch models
   directly on exposed LongBench/RepoQA answers or call that pretraining.
2. Use identical source bytes and document boundaries, causal next-byte targets,
   optimizer selection privileges, context access and sample orders. Select a
   manageable authentic easy slice for positive-control solvability first.
3. Candidate seeds: 10301, 10302, 10303. Draft learning-curve checkpoints:
   20M, 100M and 200M source-byte presentations per arm. These are planning
   estimates, not an adequacy guarantee or permission to spend compute. Freeze
   actual sequence/batch sizes, optimizer/LR grid, validation schedule, stopping
   rules and FLOP budgets only after original-source and resource preflight.
4. Report equal-data and equal-measured-compute curves separately. The existing
   full signed fairness contract remains binding; a two-lane observation is not
   simultaneous data-and-compute parity or architecture-superiority evidence.
   Resolve measured FLOP comparability before fitting, not after seeing scores.
5. Primary early outcome: held-out bits per byte and its learning curve, with
   source-cluster uncertainty, per-seed results and original checkpoint replay.
   A plateau or failed positive control makes capacity/training diagnosis
   necessary; it does not establish that an architecture cannot learn.
6. After basic learnability, register evidence-dependent code understanding and
   memory interventions on independently annotated real cases. Keep stateless,
   shuffled and reset controls diagnostic. No general reasoning, superiority
   or product claim follows merely from code language-model loss.

Before the first optimizer step, the authentic source, exact cohort/splits,
capability thresholds, full mathematical/probe obligations, measured resource
budget and independent signed prefit evidence must be available in a new frozen
`research/mathematical-engineering/langcap103/FAIR_COMPARISON.json`.
That file intentionally does not exist yet: nulls and draft estimates are not
an approved registration. The preflight report lists these unresolved inputs.
GPU work routes through the existing two-T4 quota-controlled workflow; this
CPU/no-fit proposal allocates **zero GPU minutes** and launches no training.

## Publisher corpus and actual-FIT resource preparation

`SOURCE_PLAN.json` fixes CPython 3.13.0 commit
`60403a5409ff2c3f3b07dd2ca91a7a3e096839c7` before corpus acquisition. The
publisher tag's GitHub API signature-verification record was checked. This
is publisher provenance, not the separate independent research attestation.

`source.py` verifies the exact clean commit/tree, each selected Git blob and
its SHA-256, retains the publisher license and additional original notices,
excludes tests/demos/bundled wheels and all exact duplicate contents, and
splits entire top-level modules/packages with a precommitted hash rule.
No file is assigned using a model result. Original raw bytes and file edges
remain intact; source code is never executed. Both partitions are public
exposed FIT/DEV within one repository. No protected TEST or unseen-repository
claim is created by this split. Near-duplicate and independent permission
reviews remain explicit limitations.

`profile_source.py` verifies the resulting archive and uses only FIT files
for six **zero-update** original-input forward/backward probes at batch 2,
sequence 256. It checks connected finite gradients and unchanged weights.
Profiler FLOPs are explicitly partial, and unreported operators are retained;
these numbers cannot pass the full measured-compute comparison gate. CPU
times include profiler overhead and exclude optimizer steps. They do not
predict T4 training or trained decode speed. Loss values and DEV scores are
not published by this resource probe.

The dedicated source workflow preserves the raw corpus, per-file manifests,
source identity, notices and original resource report as a replayable artifact.
The independent verifier, source-equivalent PNDS/probe qualification, full
compute accounting and frozen training execution contract remain required
before any actual optimizer update.

`prepare_registration.py` emits review drafts for both measured scales,
with each of the three arms and three prospective seeds (10301–10303),
the original source identity, fixed file-level FIT/DEV records, and replayable
source-order digests. Each sequence stays inside one original file; whole
FIT chunks are shuffled with a fixed seed, and state resets between unrelated
chunks. Draft milestones are 2,560 / 12,800 / 25,600 updates at batch 32,
sequence 256: 20,971,520 / 104,857,600 / 209,715,200 byte presentations.
These exposures repeat a finite corpus and do not guarantee adequate learning.

The drafts explicitly use the independent-fit learnability lane, with no
architecture-superiority claim. They are **not** the final signed
`FAIR_COMPARISON.json`, and do not bypass the full comparative-compute rules.
The real shared prefit gate is invoked without a signature and must reject
the declaration at its independent-evidence check. The emitted
`TRAINING_READINESS.json` retains that actual error and all remaining blockers.
The preparation workflow has no optimizer or GPU execution step.
