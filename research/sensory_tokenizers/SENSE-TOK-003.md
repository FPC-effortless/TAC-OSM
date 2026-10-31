# SENSE-TOK-003 — patch-sparse next-byte research

**Status: UNTRAINED / NO-FIT / ENGINEERING ONLY.** Stacked successor to draft PR #669 and #666. Neither language-model skill nor real computation, memory, energy or compression savings has been established.

## Mechanism

SENSE-TOK-002's 100,400-weight dense GRU updates recurrent state at each byte regardless of its segmentation features. SENSE-TOK-003 instead updates a **100,544-weight GRUCell only when a causally available patch is committed**, while the output head still predicts *every* byte. This puts T0 fixed, T1 entropy, T2 sensory and T2b corrected sensory segmenters on one identical sparse model. It is deterministic segmentation, **not** a learned boundary detector, and implements neither the native CDL teacher nor CASM-S execution.

All weights are active: embedding of 257 x 32, recurrent GRUCell (32 input, 110 hidden), and byte head (174 input, 256 outputs). Relative size difference from dense GRU is 0.1434%; both are within the mandatory 95k–105k parameter band. Nothing is pretrained or padded.

## Prediction-time information boundary

At position t the previous input is BOS (t=0) or the actual byte at t-1. The stream first processes that previously observed byte and emits only patch events with availability time <= t. A newly complete patch has its observed byte embeddings mean-pooled; the recurrent state updates once per committed patch. The current unfinished patch has a separate local mean. A 256-logit readout depends only on the updated recurrent state, the previous observed byte embedding and local mean. No target x[t] or EOS flush enters that computation. Sample state resets before each independent sample.

The T2b variant may close a patch ending at byte t-1 only upon observing a different byte at time t. The event is never retroactively made visible at the earlier endpoint. Actual source-to-logits future-target poison, prefix replay, gradient, parameter, isolation and segmentation-intervention tests are included.

**PNDS Lean 4 / executable Python reciprocal full-graph formulation and actual-source forward/backward, masking and numeric equivalence are OPEN.** The current software tests do not satisfy this mathematical release gate.

## Measured source-only diagnostics, 2026-10-31

On the local isolated implementation: 21 sparse-model tests pass; combined earlier and current mechanism tests: 54 tests and 278 parameterized subtests pass. No optimizer update, model fit or GPU execution.

Six installed CPython 3.13.5 original source files matched publisher Git blob IDs at official tag commit 6cb20a219a860eaf687b2d968b41c480c7461909. Their first 256 bytes each were used *only* for no-fit mechanism diagnosis: ast.py, inspect.py, csv.py, difflib.py, textwrap.py and email/parser.py. The script fails closed or excludes mismatching files. These publisher content matches do **not** replace independent real-source/outcome/reward/ACL attestations.

| Patcher | Recurrent update calls | Ratio to dense GRU |
|---|---:|---:|
| T0 fixed | 186 | 12.11% |
| T1 entropy | 653 | 42.51% |
| T2 sensory | 303 | 19.73% |
| T2b corrected | 303 | 19.73% |
| Dense per-byte reference | 1,536 | 100% |

All arms produced 1,536 byte logits on those same six verified 256-byte excerpts. These are **recurrent state update counts only**, not end-to-end FLOPs, latency, memory, energy, useful compression, model accuracy or intelligence. T2 and T2b have identical counts; potential utility depends on *when* patches update, not on fewer calls. The recurrent state is compact, but the current causal stream retains source bytes and completed patch payloads, so storage is **O(history)**. Python loops and per-byte tensor-to-scalar synchronization need optimization before throughput claims. Raw per-file prefix hashes and update counts appear in SENSE-TOK-003-RESULTS.json.

The tested original source and procedure are available at scripts/diagnose_patch_sparse_nofit.py. The prospective no-fit parity manifest at research/mathematical-engineering/sense_tok_003/FAIR_COMPARISON.json explicitly forbids model fitting or architecture-superiority claims.

## Future genuine learned study — blocked

**Primary question:** can actual *trained* sparse T2b improve held-out bits per byte or real continuation quality over equally sourced trained T0/T1/T2, the dense-byte control, and relevant prior-art adaptive tokenizer systems? T2b was selected posthoc from earlier code-lexer F1; these six files are NOT untouched TEST examples.

A prospective advancement threshold, **not an observed result**, is a T2b-vs-T0 improvement >=0.02 bits/byte with paired 95% seed CI excluding zero, at least 4/5 paired seeds positive on a newly frozen and qualified independent evaluation source. Separately measure exact byte continuation, distribution shift, end-to-end inference/learning compute, peak memory and time, including segmentation overhead, output-head work and retained bytes. Compare with stronger BLT/H-Net literature; these simple patchers do not reproduce those architectures.

**Fairness blocker:** sparse patchers have different actual recurrent call counts. Equal original data exposure and identical optimizer steps alone cannot imply equal consumed training FLOPs. Separately report exposure-matched vs compute-matched results unless **all** strict training-data, parameter, tuning, FLOP, seed and evaluation gates can be satisfied at once. Never pad wasted operations just to claim parity.

**Absolute blockers before optimizer execution:** authorized authentic real-world data with publisher-version/licence/identity/time provenance, independent outcome and goal verification, freeze and deduplicate FIT/DEV/new sealed TEST, registered complete architecture-comparison FAIR_COMPARISON with prefit numerical compute receipts, actual model-specific ResearchProbeAdapter and contamination checks, reciprocal approved PNDS whole-model Lean 4/Python math/source-equivalence record tied to exact source commit, trusted independently signed source/goal/reward/ACL attestations, complete checkpoint/replay evidence, and explicit owner-approved Kaggle dual-T4 quota. Any missing item is NOT AUTHORIZED, even if CI is green.

All historical studies and negatives remain preserved. **Keep novel model PR draft and unmerged.**

## Posthoc CPU performance falsification (separate untrained measurement)

A repeat host CPU whole-path test on six publisher-matching 256-byte excerpts, three timed repetitions after warmup, found **sparse/dense time ratios**: fixed 1.80×, entropy 2.43×, original sensory 1.94×, corrected sensory 2.06× (**sparse slower** in every arm). A separate first timing pass also found all four sparse paths slower. Exact timed paths, publisher input hashes and implementation limitations are recorded in `scripts/profile_sparse_patcher_nofit.py` and `SENSE-TOK-003-COST-AUDIT.json`. These are uncontrolled CPU measurements, not optimized deployment speed or end-to-end FLOP parity. Thus the selective-update mechanism test is positive, while an inference-efficiency claim remains **negative / unsupported**.
