# LANGCAP105-EVAL01 — frozen post-training DEV evaluation

This registration is made **after** the eighteen LANGCAP105 fits and **before**
their first model scoring on the original held-out CPython DEV files. It freezes
all final checkpoint SHA256s, target windows, metrics, inference settings and
reporting rules. No optimizer updates, checkpoint selection, model repairs,
retraining, architecture promotion or sealed TEST access are authorized here.

The user instructed "Proceed" after the proposal to evaluate frozen models on
unseen code. This continues the existing owner-authorized exploratory track.
It does not manufacture independent assessor signatures or change the original
training protocol, research qualification gates or previous results.

`EVAL_PROTOCOL.json` locks all 18 checkpoints, three seeds, 128-byte chunks and
all 4,646 eligible DEV windows (594,688 target bytes from 18 files / 17 groups).
All models receive the identical DEV inputs. Each model is also compared with
its original seeded initialization on DEV, and its final weights are measured
on 4,646 actually trained FIT windows from its own frozen seed order. A byte
unigram from that same seed's original FIT target exposure is a diagnostic
lower bound; Transformer and GRU remain the primary neural references.

The primary measurement is byte-weighted DEV NLL. Report bits/byte, teacher-forced
byte accuracy, every seed and source group, initial-to-trained change, and the
fixed final-model FIT/DEV gap. Paired seed/group bootstrap intervals describe
this corpus and seed set. They do not establish architectural superiority:
training full-FLOPs/tuning qualification and independent assessment remain open.

The original source split is publicly exposed same-project DEV. It is excluded
from the original optimizer but is not an independent or sealed TEST, and exact
file deduplication is not a complete near-duplicate audit. This 128-byte task
does not test long-term memory, reasoning, executable code generation or agents.

`research/phone_queue/langcap105-eval01.json` was run through the unchanged
`scripts/phone_lab.py plan`. `PHONE_PLAN.json` records the CPU route and preserves
the generic **qualified-study** missing FAIR_COMPARISON / independent-source /
reward / ACL / protected-evaluator blockers. It is not an execution permission
or scientific certificate. Explicit user authorization covers this isolated,
frozen exploratory DEV measurement. No protected evaluator, reward promotion,
GPU quota or ordinary qualification gate is bypassed or impersonated.

The proposed worker is bounded to 20 CPU minutes, with a 25-minute Actions
timeout and at most three workers. Inference uses Torch 2.15.0+cpu, FP32 weights,
float64 score accumulation, two CPU threads and batch 16. Every final checkpoint
must retain its original SHA before and after inference. Nonfinite scores,
missing models, changed source/weights/cohorts or causal-boundary failures are
explicit failures; unfavorable valid scores remain in the results.

Status at registration: **ZERO DEV MODEL SCORES PRODUCED**.
