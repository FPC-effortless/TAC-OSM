# LANGMATH098: contextual normalized associative TAC

This implements a mathematically engineered **untrained candidate**, following
the failed LANGACQUIRE097 capability experiment. It changes the architecture;
old trained weights, data, scores and failures remain unchanged.

- Contextual learned addresses and16D learned values replace local-only native
  keys and byte-bit-constrained values. Fast/slow banks have distinct addresses.
- The decoder receives both reads separately. A comparison gate sees their
  contents, disagreement and confidence; its mixture no longer hides the pair.
- Positive address mass normalization and coupled writes give a proved bank
  invariant and bounded reads. State boundedness is not useful-memory proof.
- Two work-efficient causal affine scans replace per-token concept solves and
  nonlinear upper feedback. A compact48D context process supplies history to
  associative writing; the whole model remains a dual associative architecture.
- A case/domain-controlled objective fixes explicit future-target coefficients
  while prefix supervision varies. Adam/clipping update shares remain distinct.

Actual size: **100,370 active/deployed parameters**, **432 floats =1,728B** of
persistent FP32 state per stream, excluding owner/clock metadata. Prior086 has
100,521 parameters and1,920B state. The candidate removes five throwaway state
allocations from each validation call without introducing a learned-value cache.

## Proof and correspondence

[PNDS PR97](https://github.com/FPC-effortless/pnds-lean/pull/97) contains22
kernel-checked exact-real theorems, including full persistent tensor invariance,
the actual bank update/mass/read inequalities, activation ranges, tensor affine
action and linear scan work. All inspected axioms are standard Mathlib axioms.
The expanded proof passed [run41147906379](https://github.com/FPC-effortless/pnds-lean/actions/runs/41147906379).
It does not replace a matrix proof with an assumed norm inequality.

Local no-fit verification: **54 checks passed** (35 candidate tests and19 generic
probe checks); the independent PNDS NumPy suite has five passing tests. Checks
cover the full source/reference graph, all state, serial/parallel/chunk forward
and all-parameter gradients, local acquired-state finite differences, causal
future poisoning, entity separation, boundary rejection, long constructed state
invariants and a future-only acquisition gradient with a detached negative
control. No optimizer step or original-data score was executed.

The real-number persistent-state proof is complete for its stated model and
hypotheses. **Whole-model IEEE/PyTorch AD equivalence remains OPEN.** Lean uses
the real tanh/sigmoid identity; independent numerical tests connect source tanh
to that formulation, rather than pretending IEEE operations are real arithmetic.
Exact native source/program verification, globally optimal runtime and learned
capability are not claimed. See [FORMULATION.md](FORMULATION.md) for the full
typed graph, semantics, assumptions and tradeoffs.

## Constructed engineering timing

Final local measurement: AMD EPYC9V45, Torch2.5.1 CPU FP32, one thread, five
warmups,30 samples. Models are untrained. Both initial and final receipts are
preserved; the repeated run followed the state-validation allocation change.

| Operation | Original086 p50 / p95 ms | Candidate098 p50 / p95 ms |
| --- | --- | --- |
| Prefill, batch4 ×128 bytes | 42.745 /43.786 | 2.592 /2.690 |
| One stream step after64 observed bytes, batch1 | 0.554 /0.677 | 0.173 /0.213 |

These are approximately16.5x and3.2x lower median times on this constructed
workload and10% smaller persistent state. They do not establish a trained
quality/cost tradeoff, matched-FLOP architecture superiority, production latency
or globally fastest architecture. The implementation has O(T) state-width work
and O(log T) ideal temporal span; actual kernels and hardware still matter.

Training stays disabled. A genuine next fit needs authentic fresh FIT/DEV,
FAIR_COMPARISON and the full coefficient, format/continuation, legal-evidence,
state-intervention, replay and raw-history controls. Improved learnability,
exact lexical reuse, ARC performance and intelligence remain unestablished.
