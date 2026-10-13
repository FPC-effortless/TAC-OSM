# DEV-014 — Action-rule transfer: conditional CASM learns clean XOR, finite memory remains bottleneck

**Classification: REPLAY-VALIDATED EXPLORATORY DIAGNOSTIC, NOT CONFIRMATORY.** The privileged oracle arms supply the true historical bit at both train AND test and are **not TAC-OSM memory performance**. The non-oracle arms add a separately trained action adapter, **not** the original native `CASMAlignedModel.predict`. Never report these as end-to-end native world-model results or a production-ready architecture.

## Source and integrity audit

[GitHub Actions run 38444003182](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38444003182) succeeded with **8 new mechanism tests**, live suite cardinality integrity **988**, a smoke train/eval replay and full-grid source/checkpoint evaluation replay. Full raw artifact `DEV-014-RAW-NOT-CONFIRMATORY`, artifact ID **11837547545**, whole ZIP SHA-256 **7b642d74d90c03822002be767a23e16da415fb4f4b1190cc8bd155ab60baee9c**. Archive contains 12 saved action-adapter state dicts and report with 1,920 model-episode predictions, frozen bit posterior probabilities, no-CASM/swap/neutral-bit checks and full source hashes.

All parent DEV-013 memory/bit-decoder checkpoint files are read from immutable archive **11836120110**, whole archive SHA-256 **880d56e177b78d44961a8fc75e6e0a4151ebe6eeb7a55ce81afaa6c358172d65** and each parent checkpoint SHA/parameter hash verified. The learned 4×8×24=768-scalar PLM writer and pretrained auxiliary bit reader are **completely frozen**. No new oracle memory state or learned writer modification is permitted. Evaluation replay reconstructs predictions and provenance; **it does not reproduce SGD training** and does not establish independent external generalization.

## Experimental setup and leakage boundary

Three independent trained memory-parent seeds **43,59,71**. For each parent, train four CASM action adapters initialized with identical tensor bytes and shared pretrained `compute` and `answer` weights, 120 AdamW updates on the **same** four-example training batches across arms. Two adapter linear 4→32 layers convert historical bit distribution plus visible current bit into inputs for the **real TAC-OSM `ConditionalExecutor` operator bank**, then original answer layer outputs a learned binary action. The learned model never includes a hand-coded XOR in forward.

- `oracle_sparse` and `oracle_dense` receive **gold historical bits supplied directly at inference**; these are an explicitly privileged diagnostic for action-rule learnability with perfect information, never an actual TAC-OSM memory operation.
- `predicted_sparse` and `predicted_dense` receive **only the frozen parent model's predicted bit probabilities** computed from the bounded recurrent state and public historical writes/query. No gold hidden slot/answer enters the feature, other than training loss labels (normal offline supervision).
- Sparse routes to top-1 CASM operator; dense executes top-4; all models use exactly the same parameter count and starting weights per seed. Actual operator-sample calls are logged, not estimated fractional routing.
- Each checkpoint has an **inference-time no-CASM ablation** (skip nonlinear operators, keep the two linear input projections and existing trained linear answer), and an information-erased [0.5,0.5]-bit input control. The latter scores exactly 50% on pairs whose public present is identical but true opposite past labels differ.
- The no-CASM ablation is **not independently retrained**. It shows whether the trained adapter currently depends on nonlinear conditional computation; it is **not** a fair stand-alone trained linear baseline. On clean fully balanced XOR, a purely affine readout cannot correctly classify all 4 combinations; see unit test of the zero-interaction identity.
- Held-out synthetic retrieval uses distinct `dev014-eval` procedural namespace and training `dev014-train`; checked fingerprint disjointness. 80 opposite-past pairs/seed/arm, including H=32/128/512, collision fraction 0/1, retention versus recent acquisition and competing writes. Thus 3×4×160=**1,920** model-episode classifications. Still synthetic and adaptive to a repeatedly inspected benchmark family.

## Held-out exact action accuracy (per seed and mean)

| Arm | Seed 43 | Seed 59 | Seed 71 | Three-seed mean |
|---|---:|---:|---:|---:|
| **oracle_sparse** (PRIVILEGED true bit at test) | 100.00% | 100.00% | 100.00% | **100.00%** |
| **oracle_dense** (PRIVILEGED true bit at test) | 100.00% | 100.00% | 100.00% | **100.00%** |
| predicted_sparse (frozen PLM read) | 55.00% | 58.75% | 58.13% | **57.29%** |
| predicted_dense (frozen PLM read) | 53.13% | 59.38% | 57.50% | **56.67%** |
| frozen parent historical bit-classifier | 60.63% | 61.25% | 58.13% | **60.00%** |

The teacher result shows only: **given the correct bit explicitly, the inherited CASM operator bank and answer can learn the binary action rule in this simple benchmark.** At H=512 it still scores 100% because teacher bit is provided—**this is NOT long-memory retention**. When the answer must derive from the frozen finite state, both sparse and dense CASM stay around 57%, consistent with weak memory retrieval/prediction.

Native DEV-013 (unchanged original CASM model) reached ~51.88% on a similar but not exactly identical evaluation stream. The 57.29% here belongs to a **newly trained auxiliary input/action translator** with additional parameters and specific supervised information; do not call it a fair architecture or compute-matched native TAC-OSM improvement.

### Conditional-execution ablation and operation ledger

| Arm | CASM action accuracy | No-CASM ablation (same checkpoint, NOT retrained) | Average actual operator–sample calls per evaluated 2-episode pair |
|---|---:|---:|---:|
| oracle_sparse | 100.00% | 41.46% | 6.00 |
| oracle_dense | 100.00% | 41.88% | 23.32 |
| predicted_sparse | **57.29%** | **47.92%** | **4.96** |
| predicted_dense | 56.67% | 52.50% | 11.60 |

Sparse-minus-dense action accuracy at frozen bit input = **+0.63 pp**, exploratory 95% Student-t (df=2) **[-2.48,+3.73] pp**; no evidence sparse and dense differ reliably on task accuracy, and the conditional operator-call difference does not measure wall time, full attention FLOPs, or recurrent write cost.

Predicted-sparse minus its own no-CASM ablation = **+9.38 pp**, exploratory interval **[+5.27,+13.48] pp** across 3 seeds. This supports the narrow finding that this **trained instance relies on nonlinear CASM** for the XOR action translation; it does not establish an advantage over a separately optimized non-CASM model.

Predicted-sparse minus underlying frozen bit reader = **−2.71 pp**, exploratory interval **[-9.71,+4.29] pp**; no evidence the learned action adapter systematically improves upon the information in its upstream historical bit probabilities. The underlying reader remains weak.

### Retention, acquisition and long histories

| Condition | predicted_sparse | predicted_dense |
|---|---:|---:|
| Retention action | 53.33% | 53.75% |
| Acquisition action | 61.25% | 59.58% |
| H=512 action | 53.13% | 54.17% |
| swapped bit from opposite pair | 42.71% | 43.33% |
| uninformative bit [0.5,0.5] | 50.00% | 50.00% |

Acquisition appears easier than retaining older facts. Neither predicted-input variant shows robust H=512 performance; no general long-term-memory breakthrough. All oracle swapped-bit controls scored 0% (the correct teacher bit inverted across opposite-past partners), demonstrating dependency on the supplied teacher signal, not finite-memory reasoning.

## Interpretation and DEV-015 recommendation

**Supported within this experimental synthetic diagnostic:** the nonlinear TAC-OSM conditional operator system can learn the task XOR transformation once the correct historical bit is made directly available; ablating those operators afterwards destroys its learned behavior. The present finite-memory reader offers only ~60% recoverable bit accuracy, producing ~57% when followed by a new trained CASM adapter. This narrows but does not conclusively isolate the failure: posterior calibration, reader age/collision errors, teacher-vs-soft-input train distributions and tiny-budget optimizer confounds remain possible.

**Not supported:** that original TAC-OSM's native `predict` has learned the task, that CED or CASM achieves state-of-the-art world prediction, that memory is solved, that high oracle accuracy reflects real retrieval, or that any sparse mode is computationally superior end-to-end.

DEV-015 should address the true bottleneck with an **end-to-end differentiable history-bit-to-action interface** that uses the *actual* TAC-OSM PLM and retains conditional structural execution, with matched fully trained frozen/finetuned writer and reader controls and proper capacity/latency budgets. Also establish a differently generated outside-family benchmark before any strong claim; don't repeatedly optimize to the same synthetic counterfactual data.
