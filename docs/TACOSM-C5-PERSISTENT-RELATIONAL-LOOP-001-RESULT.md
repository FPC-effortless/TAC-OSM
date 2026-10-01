# TACOSM-C5-PERSISTENT-RELATIONAL-LOOP-001 — Measured Result

**Status: MEASURED**

## Provenance

- GitHub Actions run: **36845079770**
- Successful job: **110313168397**
- Repository gate: **820 tests passed**
- Source commit: `bae076eb6d06b2b6abda6261b6014858b549d604`
- Artifact: **11152968667**
- Artifact digest: **sha256:e3ce62436294d43e702dfdcd76faa03b42753579f27fcef160e08bc98771b563**
- Seeds: 0–4
- Training: 800 outcome-supervised router updates/seed
- Calibration: 64 episodes/seed/M
- Held-out: 64 episodes/seed/M
- Online: 64 episodes/seed at M=128
- Persistence delay: 3 boundaries
- Intervening unrelated writes: 3
- Relations: XOR, XNOR, AND, OR
- M: 64, 128, 256, 512, 1024

## Anti-leakage and temporal gates

The main task stores only the two operands and operation in persistent state. The target descriptor is derived from them and is not stored.

The public query contains only an opaque address and a one-bit-corrupted operation hint.

The target descriptor is absent from query text and query context.

The temporal control passed:

- state readable after the registered three-boundary delay: **100%**
- persistent relation reconstruction at M=128: **100%**
- reset relation reconstruction at M=128: **0%**

Therefore this experiment actually places persistence in the main routing path rather than using it only as a detached control.

## Dense learned routing

| M | Dense Top-1 | Dense P90 rank | Mean rank | K90 |
|---:|---:|---:|---:|---:|
| 64 | 77.81% | 3.8 | 2.603 | 5.4 |
| 128 | 68.13% | 12.4 | 5.250 | 17.8 |
| 256 | 63.44% | 19.6 | 9.031 | 21.0 |
| 512 | 58.44% | 26.4 | 14.238 | 57.4 |
| 1024 | 47.50% | 104.4 | 38.572 | 81.0 |

The full-range P90 power fit is:

`gamma = 1.0650`.

The local P90 exponents are:

| M → 2M | Local exponent |
|---|---:|
| 64 → 128 | 1.706 |
| 128 → 256 | 0.661 |
| 256 → 512 | 0.430 |
| 512 → 1024 | 1.984 |

Because the P90 statistic is integer/quantile-quantized and the local slopes oscillate strongly, the fitted `gamma=1.065` is a descriptive finite-range fit, not an asymptotic law.

## Persistence ablation

The reset control uses the same deterministic held-out episode stream but clears the state before routing.

| M | Persistent Top-1 | Reset Top-1 | Persistent P90 | Reset P90 |
|---:|---:|---:|---:|---:|
| 64 | 77.81% | 13.13% | 3.8 | 62.0 |
| 128 | 68.13% | 7.19% | 12.4 | 123.6 |
| 256 | 63.44% | 5.63% | 19.6 | 250.8 |
| 512 | 58.44% | 4.38% | 26.4 | 505.6 |
| 1024 | 47.50% | 1.88% | 104.4 | 1007.2 |

The state-dependent router therefore carries substantial information that disappears under reset. The exact magnitude is workload-specific and not a causal estimate of a generic memory benefit.

## Non-Hamming reference

The operand-Hamming baseline uses the minimum Hamming distance to either persisted operand rather than the composed target.

Its pooled Top-1 validity is:

- M=64: **24.38%**
- M=128: **19.69%**
- M=256: **17.50%**
- M=512: **10.00%**
- M=1024: **7.81%**

The learned persistent router exceeds this reference at every registered M, while no target descriptor appears in the public query.

This removes the exact public-query Hamming shortcut that invalidated the earlier binary benchmark as a semantic-routing proxy. It remains a synthetic compositional task, not a language benchmark.

## Sparse C5 funnel

The learned persistent router is then passed through the same sparse architecture:

`persistent state + noisy query → CDL → OR-LSH → K90 → CASM → verifier/repair`.

| M | K90 | Admission recall | Final success | First attempt | Tables | p1 | p2 | Rerank/M |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 5.4 | 73.44% | 73.13% | 66.25% | 4.8 | .780 | .494 | 12.10% |
| 128 | 17.8 | 70.94% | 68.44% | 61.56% | 7.4 | .779 | .511 | 10.92% |
| 256 | 21.0 | 67.81% | 67.19% | 54.69% | 8.0 | .789 | .498 | 7.01% |
| 512 | 57.4 | 62.19% | 58.44% | 47.50% | 9.6 | .793 | .489 | 5.23% |
| 1024 | 81.0 | 65.31% | 59.06% | 45.94% | 9.6 | .793 | .480 | 4.65% |

Final success is lower than admission because the registered repair budget allows only four executions; an admitted target can therefore remain outside the attempted prefix.

The sparse result is consequently a three-stage loss decomposition:

1. representation/ranking produces a large K90 at higher M;
2. LSH admission loses additional targets;
3. the bounded CASM/repair prefix can fail even when the target is somewhere inside the admitted K90 set.

That is materially different from the earlier noisy binary phase, where final success tracked admission because the admitted candidate set could resolve the action.

## Online loop

At M=128, using a fresh router for the online arm:

- admission recall: **74.38%**
- first-attempt success: **64.69%**
- final success: **74.38%**

The online measurement is descriptive; it is not a causal estimate of learning improvement.

## Research conclusion

This phase establishes three useful facts on the registered synthetic relational workload:

### 1. Persistence is now inside the tested architecture

The router receives temporally committed operands and operation through an addressed state read. Resetting that state collapses routing accuracy toward chance-like behavior.

### 2. The Hamming shortcut is removed

The public query contains no target descriptor. The target is a composition of two persisted operands under an operation. A simple Hamming-to-operands reference performs substantially below the learned persistent router.

### 3. The difficult problem moved upstream

The state-aware router learns a nontrivial compositional relevance function, but rank scaling degrades sharply by M=1024:

- Top-1 falls from 77.81% to 47.50%;
- P90 rank grows from 3.8 to 104.4;
- K90 grows from 5.4 to 81.0.

Thus persistence is functionally present, but representation/relation learning is not yet scalable on this workload.

The sparse funnel adds a further admission and bounded-repair loss.

## What is not established

- no language-semantic claim;
- no general memory or intelligence claim;
- no asymptotic claim from gamma;
- no causal claim for online learning;
- no claim that a particular LSH table count is optimal;
- no indefinite-persistence claim.

## Next research phase

The next intervention should not add more recurrence or verifier machinery.

The immediate target is the learned **composition operator** inside the representation layer.

The next controlled experiment should hold the persistent state mechanism, task generator, candidate sets, CASM, verifier, calibration protocol, and seeds fixed, then compare:

- the current block-gated linear CDL composition basis;
- analytic relational initialization;
- execution/outcome distillation of the learned composition;
- a product-key/factorized composition representation.

The primary endpoint should be P90 rank and K90 at M=1024, with persistent-vs-reset controls retained.

This is a representation/computation-interface experiment, not a memory-rescue experiment.
