# CODEVALUE003: longer 500K code-language training on dual T4s

**Status: draft protocol, blocked; no Kaggle submission or new training.**
This change does not include a runnable training/evaluation bundle. The owner
requested a longer dual-T4 follow-up to CODEVALUE002 on 2026-10-30. Existing
CODEVALUE002 scores, checkpoints, code and negative commercial result are intact.

## Proposed experiment

Run the exact 499,623p graph-free upper48 TAC and original 500,165p GRU /
499,782p simple selective SSM from scratch on the original pinned CPython code.
Three paired seeds; batch8 x128 next-byte targets; 10,240 updates/model, or
10,485,760 labels each (32 times CODEVALUE002 exposure). Keep checkpoints at
320, 1,280, 2,560, 5,120 and 10,240, not just a final training loss. The constant
AdamW schedule is deliberately fixed before results, with no tuning. This first
question isolates longer training, rather than changing architecture and corpus
at the same time. The small repeated corpus may overfit: that is a result to
measure, not a reason to hide failed learning curves.

Use exactly two physically verified T4s in two worker processes. GPU0 receives
seeds1701/1703; GPU1 receives1702; each worker trains each arm sequentially.
This uses both GPUs, though the final third-seed tail will be imbalanced.
Cap allocation at120 minutes (240 aggregate GPU-minutes). This is a ceiling,
not a measured runtime prediction. Partial runs retain checkpoint/log evidence
and cannot produce a complete-cohort success claim or automatic resubmission.

## Stronger evaluation

The six Requests/HTTPX/Click files are exposed DEV. Score their full files at
each checkpoint, rather than only the first3,072 targets. Freeze the final
10,240-update endpoint in advance; intermediate checkpoints are descriptive.

TRANSFER_SOURCES.json pins ten candidate files from Django, FastAPI, Rich,
Pydantic and pytest by publisher commit, blob, size and SPDX license metadata.
Actual byte SHA256 checks, license text archiving, cross-study exposure and
exact/near-duplicate audits remain pending. GitHub contents responses may
include encoded content; no access-separated blind TEST claim is made.
Freeze all exclusions before fitting and score this transfer cohort only after
all nine final checkpoints are sealed. Never use it for tuning.

Score every eligible target once with identical causal windows/masks for all
predictors; offer512 fixed completions per file (5,120 offers on the candidate
transfer cohort, versus48 previously). Report NLL/bits-per-byte, top1/top5,
calibration, exact2/4/8/16/32-byte completions and correct-prefix lengths. Use
free-running generation; gold suffixes cannot feed later suggestion bytes.
Pair publisher/file clustered bootstrap intervals and publish each seed/group;
thousands of adjacent bytes are not thousands of independent trials.

Compare against no suggestion, unchanged FIT-only P3 and a fixed variable-order
count predictor, in addition to the neural controls. Disclose that count models
read the full FIT corpus while neural models sample windows. Synchronize CUDA
for p50/p95 generation latency and report runtime memory as well as tensor size.
Include rejection/distraction key-cost scenarios alongside optimistic oracle
keystrokes. These are simulations, not observed developer productivity.

A transfer utility pass requires positive exact4 gain over both count models
with lower paired95% confidence bound above zero, wins in at least4/5 publishers
and2/3 seeds, and positive net keys under the rejection-cost4 scenario.
A pass motivates a separately registered human pilot. It proves neither
profitability nor general English language ability. A natural-prose lane and
semantic/executable code-task assessment remain separate follow-ups.

## Executed preparation and remaining blockers

`CONSTRUCTED_MODEL_AUDIT.json` records actual inherited model counts, prefix
future-poison invariance, deterministic repeat predictions, finite logits and
connected finite gradients on random constructed bytes. No optimizer exists in
this audit. This is untrained engineering evidence, not learned language quality.

`COMPUTE_DECISION.json` is the actual phone planner receipt. It selects dual T4s
and correctly remains BLOCKED_EVIDENCE_OR_DESIGN: no registered workload,
private Kaggle data reference or complete scientific registration exists.
REGISTER.json is a **draft protocol**, not FAIR_COMPARISON approval.

The current cloud environment reports no configured Kaggle secret. GitHub auth
works, but inspecting repository/environment secrets and variables returns403.
Existing protected GitHub secrets might exist; their presence is unknown.
Never put a Kaggle key in chat, an issue, a PR or git.

Before submission, implement/review the offline dataset bundle, workload,
evaluator and watchdog; verify provenance and actual-sample compute receipts;
register FAIR_COMPARISON and integrate this owner-scoped500K exception without
weakening the global100K rule. The inherited architecture is unchanged; full
composed PNDS/IEEE/AD mathematical certification stays OPEN. No broad architecture
superiority claim is eligible with unmatched training FLOPs or without an adequate
attention control. The protected dispatch workflow currently requires signed
prefit evidence even for internal research; reconcile it with the existing
internal_research amendment through review, not a bypass.

Reproduce the completed no-fit checks from the repository root:

```sh
python research/commercial-value/codevalue003/audit_models.py
python scripts/phone_lab.py plan research/phone_queue/codevalue003-long-code-t4x2.json
```

No model promotion, protected TEST, merge, commercial certification or human
messages are authorized by this protocol. Findings must remain internally
measured, not independently qualified.
