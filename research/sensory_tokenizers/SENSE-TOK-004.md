# SENSE-TOK-004 — Learnable causal BYTE ↔ latent TOKEN switch

**Status: NO-FIT / UNTRAINED / DIAGNOSTIC ONLY.** Successor to SENSE-TOK-003 PR #670, itself stacked on draft #669/#666. Do not merge, fit, dispatch GPU, or claim generalization on the basis of successful fixture tests. Earlier negative findings are preserved, especially the sparse architecture's slower Python CPU path.

## Research problem

Previous T0–T2b segmenters choose patches by hard-coded class, entropy or position rules. They cannot **learn** the byte-versus-token decision. The current candidate instead produces a learned probability after observing each byte: either commit a **BYTE** representation (always preserves a single byte) or accumulate a **TOKEN** (a contextual, continuously encoded, order-sensitive multi-byte latent patch). No fixed BPE codebook is used. This is a dynamic *representation-level controller*, not a demonstrated semantic tokenizer or a replacement for BLT/H-Net/ByteFlow.

**Hypothesis for FUTURE training, not an observed result:** with correctly trained joint byte-prediction and measured compute regularization, the controller can allocate fine-grained computation only where it improves downstream predictive utility under a fixed full-cost budget. It may instead collapse to all BYTE or all TOKEN, overfit source structure, or be computationally slower.

## Full causal graph and measurements

For byte target \(x_t\), the model sees only \(x_{<t}\). At index zero the predictor consumes BOS only. After \(x_{t-1}\) has been observed, the controller computes

\[
q_t=\sigma\{[W_g (h_{t-1}, e(x_{t-1}), m_{t-1}, H(p_{t-1})) + b_g]/\tau\}
\]

where \(h\) is the global recurrent state, \(m\) summarizes the unfinished token span, and \(H(p_{t-1})\) is normalized next-byte entropy from the forecast *made before* observing \(x_{t-1}\). The entropy is detached to prevent an unprincipled gradient shortcut; it is not an oracle. An inference decision BYTE is made if \(q_t\ge\delta\); otherwise append byte to TOKEN. After at most \(K\) bytes, a token is forcibly committed. These constraints are resource and safety caps, not lexeme rules.

BYTE updates state with \(e(x_{t-1})\), after flushing any previously unfinished patch as a single token. TOKEN accumulates up to K byte embeddings then commits one contextual latent vector

\[
z_j[c] = \sum_{i\in j} \mathrm{softmax}_i(i\rho[c])\,e(x_i)[c],
\]

with learned per-channel position-recency \(\rho\) and the same recurrent cell. For prediction every byte receives a logit from the recurrent state, previously observed byte, and local unfinished-patch state. Raw input bytes plus span metadata are retained in no-fit audit output: **not** evidence of compressed storage. Token patches are continuous vectors, not discrete BPE IDs. An all-BYTE run makes one recurrent update per observed byte; an all-TOKEN run makes roughly one every K observed bytes, plus a pending tail. The mixed controller may do *more than one* recurrent update in a step when it must flush a pending token and commit a new BYTE.

### Differentiable training-ready estimator (not yet fitted)

The `surrogate` path evaluates candidate BYTE and TOKEN recurrent transitions and uses a straight-through derivative: forward state equals exactly the selected branch while gradients to gate logits flow through \((q_t-\mathrm{stopgrad}(q_t))(h_{BYTE}-h_{TOKEN})\). The `inference` path **skips unselected** recurrent candidates and executes only committed representations. These paths have matched no-fit forward behavior (within numeric tolerance) but different computation costs. The straight-through estimator is biased; whether it trains stable, useful gates remains unknown. Mode entropy, all-BYTE collapse, all-TOKEN collapse, reconstruction, predictive calibration, and the cost/loss Pareto frontier are compulsory diagnoses.

Potential FUTURE learned objective:

\[
L = \frac{1}{n}\sum_t -\log_2 P_\theta(x_t\mid x_{<t}) + \lambda\,\frac{\mathbb{E}_{q}[\text{recurrent updates}]}{n}.
\]

The cost surrogate is **not** a substitute for independently profiled total training/inference FLOPs, memory, latency or energy. Byte readouts and local processing still execute for every byte. Any future task-specific selection reward needs independently observed outcomes and an independently signed evaluator. Do not tune `lambda`, source mix, architecture or stopping with sealed TEST.

## Executed software and source tests, 2026-10-31

- Source-only 20 adaptive tests plus 54 existing SENSE-TOK-001/002/003 tests: **74 passed and 278 subtests passed** in the isolated local checkout. Zero optimizer steps or weights fitted. Falsifiers: invalid BOS/target channels, future-byte poisoning, prefix replay, lossless audit reconstruction, batch isolation, forced BYTE/TOKEN structural controls, threshold and maximal-patch guards, gate-gradient and token recency-gradient activity, forward equivalence, random binary inputs, and actual source-to-logits effect of switching. This is not training or model skill.
- CPython 3.13.5 official tag `v3.13.5` at signed tag target commit `6cb20a219a860eaf687b2d968b41c480c7461909` used for six code-file identity checks. Only exact publisher Git blobs (not labels or source metadata given to model) were used for the local no-fit forward. Full source SHA-256 and prefix SHA-256 are in `SENSE-TOK-004-NOFIT-SOURCE.json`. Git blob identity does not establish independent source/outcome/reward/ACL qualification.
- 6 × 256 initial bytes (=1536 byte logits, of which 1530 preceding raw bytes are processed as controller inputs). **Always BYTE**: 1530 BYTE representations and 1530 recurrent updates. **Always TOKEN**: 186 completed length-eight tokens, 186 recurrent updates and 42 bytes still pending at the predecision cutoff. **Untrained adaptive gate**: 655 BYTE reps, 220 TOKEN reps, 875 updates; other bytes remain in unfinished spans. These figures show a physically active two-mode controller, not a skilled learned selector. Causality is checked on the source-to-logit path.
- Actual sparse candidate works are based on current Python implementation; `surrogate` must compute both paths and can cost *more* than inference. Runtime acceleration, FLOPs benefit, model BPB, semantic relevance and retention **NOT DEMONSTRATED**.
- Live trainable & deployed learned weights: **100752**, compared with previous sparse **100544** (0.207% gap), and dense **100400** (0.351% gap). All within 95000–105000 and ≤1% relative size differences. Equal data/optimizer-exposure/training compute/tuning/test parity still missing.

## Independent novelty controls

- BLT (Pagnoni et al., ACL 2025) entropy-based dynamic byte patching, https://aclanthology.org/2025.acl-long.453/
- H-Net (Hwang et al., ICLR 2026) learned dynamic hierarchical chunking, https://openreview.net/pdf?id=ZbfLR9NbNF
- ByteFlow (Deng et al., ICLR 2026) coding-rate-adaptive byte compression, https://proceedings.iclr.cc/paper_files/paper/2026/hash/eaf5d2cdb582c058a078d4fdf52a20f9-Abstract-Conference.html
- October 2026 novelty patching https://arxiv.org/abs/2610.22100

Existing work already does adaptive bytes-to-patches. A future legitimate improvement must beat at least well-tuned strong controls at exact task goals and full system cost; a new name alone is not a new mechanism. In particular, the forward gate currently has an easily bypassed local byte head, and mean-like latent pooling may lose order/precision until learned position weights become useful. Ablate both.

## Explicit blockers and next experiment

**NOT AUTHORIZED TO TRAIN**. Obtain a licensed, independently source-qualified original real-world corpus with actual information-time, entity/split/near-duplicate safeguards and observed downstream objectives. Freeze FIT/DEV and a new untouched TEST. Specify at least 5 independent seeds, same 95k–105k active/deployed learned model budgets and ≤1% between arms, equal original data/exposure, matching planned AND consumed training FLOPs/tuning budget, checkpoint replay, fixed selection criteria, and independent real goal/evaluator and actor/assessor ACL boundaries.

Before any fit, `scripts/validate_research_parity.py --phase preflight --reward-evidence ...` must actually pass with the external operator-pinned verifying key, and `scripts/phone_lab.py plan` must authorize compute routing; the protected exactly TWO-T4 Kaggle quota must be explicitly approved. Completion needs verified postfit source/outcome/reward/ACL receipts and untouched final TEST evaluation, then a separate independent validation. Source-only pseudo-label or nofit tokens are NOT a pass. No secret signing key available to the model agent. PNDS corresponding typed **Lean 4**, executable Python full mathematical graph and actual-source forward/backward/time/mask/numerical equivalence record are **OPEN** and required for mathematical release. Keep as a draft branch until gates are satisfied.

### Recommended eventual primary metrics

Task-wise BPB, exact byte continuation, retrieval/coding utility, loss by byte class/novelty, calibration, OOD split, byte/TOKEN/forced ratios, policy entropy and collapse, causal lesioned gate impact, cross-session retention if later integrated, *total* end-to-end train/decode FLOPs, latency, peak RAM, emitted metadata storage, GPU seconds and real measured energy where available. Compare exposure-matched and compute-matched results separately if strict triple parity cannot hold, with no architectural superiority claim until every mandatory gate passes.

## Reproduce no-fit only

```bash
PYTHONPATH=src python -m pytest -q tests/test_adaptive_byte_token_nofit.py
python scripts/diagnose_adaptive_byte_token_nofit.py --source-root /usr/lib/python3.13 \
  --max-bytes 256 --output /tmp/sense-tok-004.json
python scripts/validate_research_parity.py --manifest \
  research/mathematical-engineering/sense_tok_004/FAIR_COMPARISON.json --phase preflight
```
