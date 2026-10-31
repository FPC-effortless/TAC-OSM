# SENSE-TOK-002 — causal stream, T2b and unfitted learned-gate preflight

**Status (2026-10-31):** engineering-only / zero model fits / zero optimizer steps. Separate research successor to SENSE-TOK-001 (PR #666). This does not repair or overwrite any historical negative result. No architecture superiority, real-world skill, useful compression, neural training, mathematical release or commercial-readiness claim.

## What was implemented

- CausalPatchStream implements T0 fixed, T1 entropy, T2 sensory and T2b corrected-class-boundary streaming. Every committed patch has a **byte-availability time** independent of the patch endpoint, and EOS-only commits are labeled explicitly.
- Prefix-only next-byte inputs: at timestep t, previous byte input is BOS (t=0) or x[t-1], and 16 patch features are extracted only from x[:t] BEFORE x[t] is observed. No future target/answer field exists in model inputs.
- CausalByteGRU (common architecture for T0/T1/T2/T2b): live trainable **100,400** parameters including embedding, GRU and output head. Exactly matched between its four arms.
- LearnedSensoryGateGRU (L1 **untrained candidate**): **100,513** live parameters (0.1125% above common model); learnable soft local sensory memory gate connected to next-byte loss. This is NOT a discrete tokenizer, and it evaluates a GRUCell every byte, so no compute reduction has been demonstrated. It can be representationally redundant with the GRU.
- No training script, optimizer updates, model checkpoints, GPU submission or autonomous TEST evaluation introduced.

## Timing noninterference

A T2b patch covering bytes [s,i) may close when the class of byte x[i] changes. This is an **observed-at-i+1 decision**, so the closed patch has end=i and available_after=i+1, not i. Thus the predictor for x[i], which operates on x[:i], cannot see the boundary. Complete-string pretokenization is NOT valid as a causal training input without reconstructing exactly these availability times. The stream does not simulate EOS during next-byte prediction. Tests include counterfactual future poisoning, immutable committed patches, chunked-vs-monolithic streaming, exact byte roundtrips and unfitted model gradient connectivity.

## Measured no-fit boundary diagnostic

On eight additional locally installed CPython 3.13 source files, 62,102 identical bytes/arm:

| Measure | T0 fixed | T1 entropy | T2 original | T2b |
| --- | ---: | ---: | ---: | ---: |
| Patches | 7,765 | 23,038 | 11,622 | 11,627 |
| Mean per-file Python lexer-boundary F1 | 0.12514 | 0.18540 | 0.33191 | 0.41283 |
| Delayed patch commits | 0 | 0 | 0 | 11,619 |
| Byte roundtrip | PASS | PASS | PASS | PASS |

T2b improved lexical boundary F1 on 8/8 files. This is an ASCII-oriented **intrinsic heuristic proxy**, selected after earlier exploratory evaluation; it is NOT a preregistered sealed predictive endpoint. No learned model used it to generate text. Patch count is not compression, and Python script time is not optimized model inference cost.

## Source-authenticity audit

Official publisher: python/cpython annotated release tag v3.13.5, 2025-06-11, commit **6cb20a219a860eaf687b2d968b41c480c7461909**. GitHub reported valid signed tag verification. The original full-file Git blob SHA-1 from the official repository was compared with the local installed full-file blob SHA-1 (standard Git blob header included).

| Source | Exact official blob match | Official blob SHA-1 |
| --- | --- | --- |
| argparse.py | NO — exclude | bd088ea0e66a0484f8e9390d421e769956eb3519 |
| ast.py | YES | 37b20206b8af4fda67c631fd010eda0008985511 |
| inspect.py | YES | de2fe7e1ae0f6d345fdf1a53f4f0073079f50802 |
| csv.py | YES | cd202659873811a2ea47c7374cc268d26a9ac5f3 |
| difflib.py | YES | 33e7e6c165aa0c12c779331927c3a40f3b09d9e2 |
| textwrap.py | YES | 7ca393d1c371aad54a2d1d70fae15efa889175bb |
| email/parser.py | YES | 06d99b17f2f9c4992301c432f498b0a32339171b |
| logging/__init__.py | NO — exclude | 357d127c09048226780c7ac60db1e1ab792efd0f |

**Verified publisher-matching subset (6 files, 45,753 bytes):** T2 lexical F1 **0.32181** (8,589 patches), T2b **0.40621** (8,593 patches). This source selection was made after the initial eight-file evaluation using a provenance criterion, not a sealed test design. The two differing local files remain explicitly logged, not relabeled as original publisher data. CPython licensing: PSF License v2, subject to file-specific notices.

An official signed source tag is not a substitute for independently sourced task outcomes, evaluation-scorer access separation, trusted reward/goal/ACL signatures, or a frozen real FIT/DEV/TEST split.

## Reproduction

~~~bash
PYTHONPATH=src python -m pytest -q tests/test_causal_patch_stream.py tests/test_causal_byte_probe_nofit.py
python scripts/diagnose_causal_stream.py --max-bytes 8192 --repeats 3 --output /tmp/sense-tok-002.json
~~~

The streaming module and diagnostics run without torch. PyTorch is required for the **unfitted** model smoke checks. Local script medians are uncontrolled Python timings, not FLOPs/energy, and may vary by runtime. Exact source SHA256 and excerpt SHA256 are emitted by the CLI. Current research status should remain NO_FIT until all independent gates pass.

## Main scientific blockers, not waived

1. **Independent qualified data & pre-registration**: original real FIT/DEV with publisher checksums, causal timestamps, entity/site/period split where applicable, and sealed TEST. Historical exploratory source selections do not become prospective holdouts.
2. **Fair comparator**: same raw input bytes, true trainable/deployed model size 95k–105k and <=1% pairwise mismatch (counts checked above), same optimizer/exposure and auditable actual training FLOPs/tuning budgets. Different GRU implementations/soft-gate execution are NOT matched compute just because parameters match. Require separate exposure- and compute-matched reports if needed.
3. **Independent evidence**: source/outcome/reward/goal/ACL and independent verifier cryptographic preflight; no trainer-generated scorer signatures. Model-specific ResearchProbeAdapter and predecision/gradient/activation/poison audits, checkpoint and replay proof.
4. **Math**: PNDS reciprocal source SHA registration, complete Lean 4 and executable Python formulations and source equivalence remain OPEN. Unit tests / induction intuition do NOT count as formal proof.
5. **Compute**: reviewed phone_queue + phone_lab planner; any Kaggle 2xT4 training additionally requires owner-approved protected GPU quota and independently signed preflight. None has been requested or consumed.

**Interpretation:** The causal information-boundary defect has a viable software correction; T2b is only a hand-designed class-based segmentation heuristic. The decisive outstanding result is whether a genuinely learned sensory representation improves held-out real byte prediction or task completion over strong matched baselines without leakage and at acceptable end-to-end cost.
