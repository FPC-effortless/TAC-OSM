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
