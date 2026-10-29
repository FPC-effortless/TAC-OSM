# LANGPROFILE058 — original-trained decoder operation profiling (preregistered)

**Scope:** Internally measured CPU diagnostic, not a new fitted neural variant. Branch stacked on LANGDECODE057 PR #596, which has a mixed p95 result. The original 499,623-parameter seed1701 FIT320 frozen checkpoint and registered stateful graph are used as immutable inputs. New model fits, GPUs, protected TEST, DEV logits, checkpoints, training and neural performance claims are forbidden.

## Hypothesis and falsifier

The `_advance` path contains a tiny 6×6 Laplacian `torch.linalg.solve`, per-byte projections, repeated 8×8 and 2×16×8 associative memory products, fast/slow recurrence and FFN, and many small tensor allocations. The best mathematical kernel optimization target must follow measured ATen cost, not visual complexity or speculative profiling. We **do not** predict which will dominate: this run measures it.

An expensive operation is a credible first target only if consistently prominent in both 128/512 contexts in the **unmodified** transition's self-CPU operator accounting. If profiling is inconclusive or has no concentrated operator cost, report the negative and do not automatically revise the graph. If `linalg_solve` is dominant, separate research may compare a mathematically justified SPD alternative, but it is NOT included in this experiment and will require PNDS Lean/Python and FP32 parity gates. Do not change current output semantics without proof/replay.

## Exact test protocol

1. Original genuine licensed PSF CPython 3.13 FIT owner only; verify source owner split and original optimizer/RNG sample-order checkpoint SHA `0cbc1b89f53ea2a0fe1371a179a9ff14cc35ae84d09f76bdab1dffeac4ac86cb`.
2. For source prefixes length128 and512, replay original growing-prefix graph against original IncrementalTAC043 and BatchedValidationTAC057 as in LANGDECODE057; require every logit and all nine tensors bitwise between two incrementals, full graph numerical tolerance, stable greedy bytes, original bidirectional snapshot continuity, persistent 3904 bytes.
3. For each validated prefix, separately profile three stages on the SAME frozen prior state and SAME original-greedy next token: `step_with_guards`, `transition_only` via `_advance`, and `validate_only`. Verify `step` == `_advance` bitwise, all fields unchanged on input, model weight hash unchanged; no hook injection or graph changes.
4. Unprofiled wall latency measured with 2 warmups and 7×16 calls per stage (single host FP32 eager two threads, batch1); record median/p95/raw results. Separately use PyTorch CPU profiler with 12 repeated calls, record shapes and memory, collect ATen `self_cpu_time_total` and operator counts. Sum disjoint categories; report raw top operator keys. **Never compare profiled self-CPU microseconds directly with unprofiled wall milliseconds, or interpret percent categories as fraction of end-to-end wall time.**
5. Save original JSON evidence and sha-pinned source/result identity. Tests verify profiler failure modes and monotonic accounting, frozen state, clock, no mutation, and source method equivalence. No optional GPU/kernel work and no threshold chasing.

## Mathematical provenance and limits
This is instrumentation of the unchanged LANGSTREAM043 source, not an architecture graph revision. PNDS [PR #66](https://github.com/FPC-effortless/pnds-lean/pull/66) Lean `PNDS/LangStream043.lean` and independent Python `foundation/v1/langstream043.py` remain *partial*; source-wide IEEE/AD/gradient equivalence obligations **OPEN**. Profiler categories are dispatch operator groups, not causally isolated model components. The original LANGDECODE057 p95 FAIL stays preserved. Any future mathematical kernel optimization must become a separately preregistered candidate with exact input/output/error bounds and real-source checkpoint replay.

Outcome is **OPERATOR_PROFILE_CAPTURED** only if all source/equivalence and profiling guards pass; otherwise **INCONCLUSIVE**, with failed evidence preserved. No neural architecture winner, mobile throughput or memory efficiency claim follows from this run.
