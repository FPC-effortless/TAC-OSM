# SENSE-TOK-005 — causal BYTE / learned discrete token ID / latent-token control

**Status (2026-10-31): NO-FIT engineering diagnostics only. 0 trained models, 0 optimizer updates, no capability, cost, code compression, or intelligence claim.** Successor to SENSE-TOK-004 (draft PR #673), with originals SENSE-TOK-001–004 and their documented failures preserved. This branch is draft, and its history is stacked on the prior draft research branches.

## Motivation and precise claimed mechanism

SENSE-TOK-004 has a learned BYTE/TOKEN gate but a *continuous* pooled latent vector for every completed token. That is not a reusable discrete token identity. SENSE-TOK-005 adds a small trainable vector-quantization codebook **only at the observed completion of a multi-byte patch**. It creates an actual reusable token **ID**, not a BPE/subword vocabulary entry. A single-byte representation remains the original byte embedding. Both representations select the recurrent-state input, and the controller still determines when to accumulate or commit.

The **local byte-level decoder remains active between token updates**: it consumes only the observed prefix. This architecture is *hierarchical byte+token representation*, **not** proof that byte processing or O(history) caches have been eliminated. Untrained codebook identities are arbitrary, may collide or collapse, and have no demonstrated semantic meaning.

## Causal, typed mechanism

Input to each next-byte prediction is `BOS,x[0],...,x[t-1]`. No `x[t]`, future target, correctness oracle, gold structure, reward, post-action outcome or arbitrary query is accepted by the model. The route at observed position `t` depends on the previous hidden state, byte embedding of `x[t-1]`, the pending prefix summary and entropy of the *previous* prediction. It chooses BYTE (commit single byte, potentially after a pending token) or TOKEN (accumulate until forced length bound). A pending token may be completed only when all its constituent bytes are observed. No premature EOS flush is used.

With embedding `e(x) in R^32`, trainable recency `r in R^32`, and already-observed patch `b[0:k]`, form coordinate-wise chronological pooling:

`z_j = sum_i softmax_i(i * r_j) * e(b_i)_j`.

For `k >= 2` and 8 trainable codebook vectors `c_0...c_7 in R^32`, assign a **hard** identity:

`q = argmax_l cosine(z,c_l)` (ties: first index); `token_rep = c_q`.

The backward approximation uses a straight-through softmax over the same observed-source similarities. It preserves the *hard* chosen code ID in the forward pass, giving a path from next-byte loss to the codebook, patch pooling and byte embedding. It is **not** an unbiased policy gradient estimator or evidence of learned identity quality. For `k=1`, the representation is exactly the individual byte embedding; codebook IDs cannot be assigned. The shared GRUCell only receives completed byte or token events. The output head still predicts every byte using the recurrent state, last observed byte embedding, and summary of any pending observed bytes. A maximum token size bounds accumulation. Parent gate and pooled-vector calculations are reused, and no fixed lexical rules determine token IDs.

A **diagnostic only** single-decision intervention `(intervention_at, intervention_mode)` is externally supplied *before* replay. It forks from identical earlier observations and state and forces BYTE versus TOKEN at one observed decision, while allowing the learned policy to choose other steps. This is used to build eventual controlled counterfactual decision tests: **the evaluator** can reveal the actual next byte and score two outcomes only *after* the gate's decision. It never supplies that byte to the gate.

## Evidence completed — source-only, no model fits

The local assembled 001–005 test suite produced **107 passed tests plus 278 subtests**, including **32 new S005 tests**. Four new source surfaces are source-only: model, pure-Python operator reference, tests and publisher-blob-matched diagnostic script. A separate previous file/baseline suite remains immutable. Source-to-output future-byte poisoning, prefix replay, token code availability, token-code knockout, matched parameter count, seed/batch isolation, active codebook/gate task gradients and one-step externally controlled counterfactual forks were exercised. Two reference operators (pooling and hard code ID assignment) match PyTorch on finite randomized inputs, but full-model numerical/gradient/finite-precision/Lean source correspondence remains **OPEN**.

**Actual live weights:** S004 reference 100,752; S005 codebook-8 101,008, a **0.254% relative gap**. Both meet the 95,000–105,000 band and within-1%-parameter prerequisite; no real training, measured training FLOPs or full comparison parity exist.

Six independently identifiable full CPython v3.13.5 original-source Git blobs (`ast`, `inspect`, `csv`, `difflib`, `textwrap`, `email/parser`) were byte-checked against pinned publisher Git IDs, and **only the first 256 bytes** per source were used for this exposed source-only DEV diagnostic. No data reallocation to TEST and no trained LM evaluation. Three *untrained* initializations x six files each produce the following cumulative measured events on 4,608 byte prediction positions / 4,590 previous observed bytes:

| Forced mode / untrained action | Recurrent updates | Completed multi-byte code IDs | Median active code IDs per source/seed | Median dominant code share |
| --- | ---: | ---: | ---: | ---: |
| Always BYTE | 4,590 | 0 | 0 | n/a |
| Always TOKEN | 558 | 558 | 7 / 8 | 43.5% |
| Untrained adaptive gate | 3,387 | 473 | 6 / 8 | 32.3% |

These are **mechanism usage counts**. VQ token assignments and the gate are randomly initialized, not trained; code usage is not evidence of semantic content, useful compression or model intelligence. Uncontrolled single-threaded Python wall times are recorded separately as diagnostics, **not** an efficiency claim. The model executes a 256-way output head at every byte and retains raw-prefix metadata during an audit. A latent-token assignment incurs further vector normalization, similarity and candidate overhead.

## Critical scientific falsifiers (planned, not reported as passed)

1. **True downstream utility:** five separately trained, source-qualified paired seeds; held-out bits-per-byte improvement over S004 and matched byte/patch controls, with previously registered S003 advancement criterion `>= 0.02 bits per byte`, 95% paired seed CI excluding zero, and >=4/5 positive seeds. Exact continuation on real unexposed source and out-of-domain shift separately scored. No tuning on sealed TEST.
2. **Controller policy validity:** on DEV only, one-step byte/token forks from exactly identical predecision state; compare after-decision prediction loss and separately the later horizon. Measure action-consistency and regret against evaluator-only oracle. Never train using TEST oracle decisions or expose counterfactual targets to the gate.
3. **True reusable structures:** shuffle token ID lookup, swap individual code vectors and zero the codebook; check causal code-to-behavior effect. Report token occupancy/perplexity, code collapse and novel-sequence behavior under held-out real data. Demonstrate transfer beyond repeated byte patterns; otherwise the token bank is only an auxiliary quantizer.
4. **Non-bypass:** ablate the raw local-byte readout at decision time and examine its incremental benefit; compare true recurrent code-ID gains to an equal-width noncodebook model. Retaining the byte readout is permissible for language modeling but must not be described as eliminating byte processing.
5. **Whole-path efficiency:** include entropy controller, code selection, patch pooling, each GRUCell candidate, per-byte decoder, retained history and memory; run both exposure-matched and FLOP-matched controls when strict three-way parity cannot be reached. Never count fewer GRU calls as lower inference latency without measurement.
6. **Anti-cheat:** publisher/source/expiry/rights and entity-time boundaries; train-only preprocessing, near duplicates, answer-bearing IDs, future perturbation, reset and cross-row state isolation, cache state, deterministic replay, actual gradient and checkpoint provenance; record PASS/FAIL/INCONCLUSIVE per applicable boundary.

## Hard training and mathematical gates — all pending

`research/mathematical-engineering/sense_tok_005/FAIR_COMPARISON.json` is a **noncomparative diagnostic** with `no_new_model_fits=true`. The following remain BLOCKED: independently source-qualified real downstream task/outcomes; frozen FIT/DEV/sealed TEST and baseline selection; strictly matched **actual** model size, original data/exposure, optimizer steps and audited FLOPs/tuning; trusted independent source+goal+reward+ACL Ed25519 prefit attestation; trained-checkpoint-specific evaluator/reward postfit evidence; model-specific `ResearchProbeAdapter`; reciprocal complete PNDS Lean4 + Python full-graph/source-equivalence tied to immutable source commit; approved protected Kaggle exactly 2×T4 quota if GPU justified. No source file, test, CI run or hypothetical math statement is authority to fit without these. No silent promotion or merge.

### Local no-fit reproduction

```bash
PYTHONPATH=src python -m pytest -q tests/test_adaptive_codebook_token_nofit.py
PYTHONPATH=src python -m pytest -q tests/
python scripts/diagnose_codebook_nofit.py --source-root /usr/lib/python3.13 --max-bytes 256 --output research/sensory_tokenizers/SENSE-TOK-005-NOFIT-RESULTS.json
```

A standalone independent Python mathematical reference is in `src/tac_osm/adaptive_codebook_math.py` for **two operators only**, not the full network. All failed and negative earlier observations remain valid and unchanged.
