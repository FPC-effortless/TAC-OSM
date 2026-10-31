# SENSE-TOK-006 — controlled adaptive-representation learning preparation

Status: **NO FIT. Training comparison DRAFT/BLOCKED.** No source corpus or TEST
was opened by this preparation, no optimizer was constructed, and no learned
decision, semantic-token, intelligence, efficiency or superiority result exists.

The implementation inherits S005 source commit
`045994fb432a9874d0fc4f7914e5fe45ad8fe7ec` (draft PR #674). Original S001–S005
sources, evidence, failures and model computations remain unchanged. New
controls and evaluator operations live in separate modules.

## What is implemented

`adaptive_token_controls.py` instantiates six comparison candidates, with
paired initialization for shared S004/S005 weights and unchanged caller RNG.

| Arm | Trainable weights | Deployed learned weights | Distinction |
| --- | ---: | ---: | --- |
| Dense BYTE GRU | 100,688 | 100,688 | 118-wide recurrent byte baseline, no patch readout |
| Forced TOKEN | 100,832 | 101,008 | S005, unused gate weights frozen |
| Adaptive continuous latent | 100,752 | 100,752 | Original S004 |
| Adaptive discrete codebook | 101,008 | 101,008 | Original S005 |
| Frozen gate weights | 100,832 | 101,008 | Paired gate initialization stays fixed |
| Exogenous random gate | 100,832 | 100,832 | No learned gate, explicit independent random draws |

Both maximum parameter gaps are 0.316807%, below 1%. This is an instantiated
parameter prerequisite, **not** matched training/data/tuning/FLOP evidence.
The frozen-weight gate's input representations can still learn, so its
decisions are not frozen. The random control needs real FIT-only calibration;
the readiness script's probability .5 is a count-only fixture.

Forcing S004's BYTE path would leave its 32-dimensional pending-token head
input permanently zero. The resulting 8,192 unused head weights must not be
counted as useful comparison capacity. The separate dense BYTE GRU avoids
that padding; all its encoder, recurrence and readout blocks receive actual
next-byte gradients in the constructed no-fit check.

`adaptive_token_evaluation.py` provides:

- **Fixed-key, wrong-value/zero-value interventions.** It copies the model,
  records baseline gate outputs, retains original code selection keys/IDs,
  substitutes only representation values and replays identical routing.
  Caller mode, parameters and gradients are preserved. Knockouts can be out
  of distribution; changed logits alone do not establish usefulness.
- **Joint key/value row permutation as a relabelling control.** The encoder
  and lookup share vectors; a joint permutation may rename every code while
  preserving predictions. Relevant argmax ties/numerics are explicit caveats.
  Joint row shuffling is not a valid code-destruction intervention.
- **Exact-horizon realized route regret.** Fork one observed BYTE/TOKEN
  decision, reveal later source bytes chronologically, and measure next-byte
  loss over the complete registered horizon with subsequent policy response.
  Reject truncated horizons, misaligned targets and nonfinite logits. Ties
  have no arbitrary oracle-action label. This is a teacher-forced internal
  computation diagnostic, not expected optimality or causal world action.

The local byte decoder remains in S004/S005. No code-only reconstruction,
byte-compute elimination or persistent cross-session state is established.

## What has actually run

- Complete local source snapshot suite: **1,299 passed** (Python 3.11,
  torch 2.5.1+cpu, pytest 8.4.2; no separate subtest count claimed).
- 24 new no-fit regression cases cover candidate gradients/counts/RNG pairing,
  random-policy replay/entity/prefix isolation, relabelling/value interventions,
  complete-horizon independent scoring, target alignment and fail-closed setup.
- The phone planner selects GitHub CPU and `READY_NO_FIT`; it explicitly
  denies training/TEST approval. The diagnostic parity registration passes
  `PASS_DIAGNOSTIC_ONLY`, not a training preflight.
- The preparation command's `--check-training` exits **2**, preserving the
  blocked state. No FIT/DEV scores or checkpoint weights were produced.
- The existing queue test now validates every registered proposal rather
  than assuming the queue can never grow beyond two entries.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
python scripts/sync_test_count.py --check
python scripts/phone_lab.py plan research/phone_queue/sense-tok-006-readiness.json
python scripts/validate_research_parity.py --manifest research/mathematical-engineering/sense_tok_006/FAIR_COMPARISON.json --phase preflight
python scripts/prepare_sense_tok_learning.py --output research/sensory_tokenizers/SENSE-TOK-006-READINESS.json
python scripts/prepare_sense_tok_learning.py --check-training
```

The last command intentionally exits 2. `SENSE-TOK-006-READINESS.json` records
the protocol and implementation SHA256s, measured counts and individual OPEN
gates. It is a reproducible preparation receipt, never source qualification.

## The controlled training comparison remains blocked

`SENSE-TOK-006-TRAINING-PROTOCOL.json` specifies six arms × five paired seeds,
the original-byte BPB denominator, the inherited 0.02 BPB advancement target,
DEV-only decision horizons 1/8/32, data boundaries, negative controls,
complete cost accounting and checkpoint/replay obligations. It is explicitly
unreviewed and cannot pass as an architecture-comparison training manifest.

Actual original FIT/DEV and sealed TEST custody, fixed numerical training/
tuning/calibration budgets, whole-graph FLOP parity, independently signed
source/goal/reward/ACL prefit evidence, source-specific ResearchProbeAdapter,
reciprocal complete Lean/Python correspondence and protected approved two-T4
execution are **OPEN**. No optimizer or GPU dispatch is provided by the
preparation command. Those obligations require actual evidence and integration;
source-controlled status labels cannot turn them into passes.

BLT, H-Net and VQ representation learning are prior art, with primary sources
and reuse limits in the protocol. These GRU candidates are not faithful BLT/
H-Net implementations and cannot establish superiority to them. The next
scientific result must come from qualified training and independent held-out
evaluation, rather than another untrained representation variant.

## Mathematical scope

Dense BYTE has `e_t=E[x_(t-1)]`, `h_t=GRU(e_t,h_(t-1))`,
`logits_t=W[h_t,e_t]+b`; BOS occupies the initial observed-ID slot.
S004/S005 retain their source-pinned recurrence/pooling/quantization graphs.
The exogenous schedule contains only runner-provided independent uniforms;
the frozen gate intervention changes its weights' training status only.

The selector/value intervention is `q=argmax cosine(z,c_i)` followed by
`value=c_(permutation(q))`, keeping the original `c_i` selector and baseline
routing. Jointly permuting keys and values only relabels the maximizer when
the relevant maximum is unique. Regret is selected-action horizon BPB minus
the smaller of the two fork losses; this includes realized target noise.

These equations and finite unit checks are not full source-equivalent Lean
proofs. PNDS full-graph, precision, gradient and serial/parallel obligations
remain OPEN and continue to block mathematical readiness and training.

Companion [PNDS draft PR #100](https://github.com/FPC-effortless/pnds-lean/pull/100)
adds partial selected-lookup, realized-regret and parameter-count formulations
at source commit `6887529e33c6d7379a2cacc5dc3f24b40ae458be`. Its 24 Python
tests pass locally. No local Lean build is claimed. The new prospective-study
record is byte-identical in both reciprocal manifests; all original 17
historical rows remain unchanged. The model/control source pin is the
implementation-only commit `9e2696ca0d15c1e8633777794c4fdf9a891b3890`,
before this metadata-only cross-link commit.
