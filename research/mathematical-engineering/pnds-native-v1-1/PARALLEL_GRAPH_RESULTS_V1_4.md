# TAC-OSM Native Parallel Computational Graph v1.4 — Implemented and Verified

Date: 2026-10-24.
**Original source:** immutable DIAGKEY004 commit 081e534a074001c6a3b2251b3ff31ad20ec4fca8.
**Source-matched execution:** src/tac_osm/variants/parallel_partitioned_native.py.
**Distinct fully parallel candidate:** src/tac_osm/variants/full_time_parallel_native.py.
**Work-efficient memory:** src/tac_osm/variants/parallel_affine_memory.py.
**Complete typed 16-node DAG:** research/mathematical-engineering/pnds-native-v1-1/PARALLEL_FULL_GRAPH_CONTRACT.json.
**PNDS formal mathematical derivation:** https://github.com/FPC-effortless/pnds-lean/pull/7 and foundation/v1/PARALLEL_COMPUTATIONAL_GRAPH_V1_4.md.

## 1. What was actually completed

Two separate source-verified routes now exist:

**A. Original-model compatible:** All B×T independent event embeddings and input-only fast/gate projections are precomputed as batched tensors. The exact causal old-memory-read → nonlinear fast → slow → gated-delta write remains a T-length recurrence. Real-number affine splitting preserves the original source semantics; PyTorch FP32/FP64 outputs and all parameter gradients are compared with a tolerance. This is *not* a time-parallelization of the full original model.

**B. Full time-parallel new candidate:** Reuse exactly the native 101,527 learned source parameters, but change fast/slow propagation to a causal prefix summary. Memory gates become independently known from B×T parallel features. The 8×8 memory follows a right-affine update with chronological associative pair composition, computed using an O(T)-work O(log T)-depth balanced upsweep/downsweep prefix algorithm. Old memories are read in parallel by shifted prefix, and learned fast/slow state refines the reads after the memory schedule. A final native codebook readout produces the same 32-logit *shape*, NOT the same values as the original network.

**This is a new untrained architecture.** The original fast/slow recurrence does not remain intact in candidate B, and previous trained checkpoint performance is not inherited. No new weights were added, optimized, transferred or benchmark-trained.

## 2. Exact mathematical graph and parallel legality

For original normalized 8D key k_t, value v_t and externally or causally computed gates α_t,β_t:

  M_t=M_(t-1)P_t+Q_t,
  P_t=α_t(I−β_t k_t k_t^T),
  Q_t=β_t v_t k_t^T.

For an earlier and later interval, chronological composition is:

  (P_A,Q_A) ⋄ (P_B,Q_B) =
  (P_A P_B, Q_A P_B + Q_B).

The composition is **associative but not generally commutative**. PNDS/ParallelMemory.lean has compiled exact-real matrix-level chronology, associativity, identity and grouped-fold theorems with axiom audit. This permits reparenthesizing (never permuting) sequence updates for the conditioned affine subgraph.

Original native retention/strength depend on previous learned fast/slow/recall states, and native memory uses nonlinear global Frobenius clipping. Thus arbitrary original recurrence time-parallelization remains unproved. New candidate B removes this loop by computing its memory gates *before* reads, and replaces clipping with a conservative retention budget.

Set radius R=4 and alpha_eff=alpha_raw(1−β/R). Given ||k||₂,||v||₂≤1 and alpha_raw,β∈[0,1], the spectrum of I−β k k^T lies within [0,1]. Therefore its operator norm is ≤1 and the value-by-key update obeys

  ||M_t||_F ≤ alpha_eff ||M_(t-1)||_F + β.

Under ||M_(t-1)||_F≤R, alpha_eff R+β≤R, giving a projection-free invariant budget. PNDS/ParallelMemory.lean proves the **conditional scalar bound**, with the norm recurrence itself as an explicit premise. A full 8D tensor spectral-bound Lean certificate and source-float refinement remain outstanding; the analytical 8D eigenvalue derivation and Python numeric checks are documented separately.

Read-only memory uses P=I,Q=0, while a parallel last-write-index prefix gather copies exact bytes after floating-point reassociation. Numerical differences between sequential and parallel non-read-only arithmetic are checked by tolerance, not denied.

## 3. Reproduced CI results

**Actual [source and parallel graph verification run #40140051455](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40140051455): PASS.**

- Native source preprojection path: 184 output, trace and parameter-gradient checks.
- Conditioned memory associative scan (including BOTH Hillis–Steele and balanced Blelloch numerical/gradient comparisons to the pinned original writer): **130 checks**.
- Distinct full-time-parallel candidate: 230 checks including every time-prefix's causality, exact read-only trace identity, norm-bounded memory, bounded fast/slow representations, all-parameter finite gradients, full shape and explicit non-equivalence to original native recurrent outputs.
- Original 26 operator, 40 full original computation graph and 1,100 forward envelope tests remain independently passing under this PR.
- Original 101,527 trainable parameters reused exactly; newly fitted steps **zero**.
- Parallel path CPU and graph experiments are generated mechanism data, NOT real-world prediction evidence.

**Actual [PNDS Lean workflow #40140048668](https://github.com/FPC-effortless/pnds-lean/actions/runs/40140048668): PASS.** It checks real matrix chronological composition and the conditional bounded-budget theorem, including axiom inspection; there is no sorry in compiled proof output.

## 4. Measured single-thread CPU throughput — exploratory, not a skill gate

Actual run #40140051455 used Torch 2.14.1+cu130 installed on a CPU runner, single CPU thread, batch size 4, dtype FP32, two warmups and five forward timing repetitions (reported as medians). Different architectures *do not* produce equivalent predictions or perform the same total compute; this is a computational engineering comparison, not a fair training/superiority experiment.

| Observed sequence T | Original source recurrent | Original-compatible parallel input | New fully parallel candidate | Source/candidate CPU ratio |
|---:|---:|---:|---:|---:|
| 32 | 7.16668 ms | 6.93199 ms | 2.11094 ms | 3.40× |
| 128 | 27.79708 ms | 26.29624 ms | 5.26069 ms | 5.28× |
| 256 | 55.27422 ms | 52.73568 ms | 12.49601 ms | 4.42× |

The ratio describes one short single-thread CPU benchmark, not GPU throughput, training efficiency, energy, p95 production latency or deployment performance. There is no performance pass/fail qualification on these numbers. Independent real skill remains unmeasured.

## 5. Explicit proof/experiment status

| Property | Result |
|---|---|
| Complete candidate input→output computational DAG | Source-implemented and 16 nodes indexed |
| Native original-compatible mathematical preprojection | Forward/gradient numerically checked |
| Original full nonlinear recurrence parallel-time scan | **NOT established** |
| Affine conditioned matrix scan associativity | **Lean passed** |
| Work-efficient O(T)-work scan implementation | **PyTorch source-reference/gradient passed** |
| Projection-free candidate memory norm scalar budget | **Lean conditional theorem passed** |
| Candidate source code and causal graph tests | **Passed** |
| Full 8D rank-one operator norm in Lean | OPEN (analytical claim only) |
| Original native output equality in new candidate | False by design |
| New trained model performance and generalization | **Not measured** |
| Authentic held-out dataset, action outcomes and matched training compute | **Not executed** |
| All 17 historic TAC variant graphs completely reengineered | **No** |

Next capability-critical action is to train candidate B versus native and structured/GRU/Transformer controls on authentic logged longitudinal state and action/outcome data with equal trainable parameters, exposure, real data/splits and **training FLOPs**; CPU elapsed time is not a fairness requirement. Do not elevate source synthetic diagnostics, no matter how fast, to a real-world skill claim.

The mathematical improvement is substantive: TAC-OSM now has both a source-matched partially-parallel path and a **fully parallel causal candidate** with bounded native associative storage, exact algebraic chronological aggregation, compiled Lean proofs and exercised tensor/autograd software. The absence of training remains material.
