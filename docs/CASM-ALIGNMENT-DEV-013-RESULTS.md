# DEV-013 — CASM-aligned historical supervision: actual decisions still near chance

**Status: DEVELOPMENT NEGATIVE for native TAC-OSM decision improvement; auxiliary information signal present. NOT confirmatory science, not a solved learned-memory model, not a DeepSeek/CED improvement.**

## Immutable evidence

[GitHub Actions run 38440147571](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38440147571) completed successfully: eight new CASM-specific tests, full suite-size integrity (980 collected), mechanism smoke and **complete 3-seed checkpoint inference replay**. Full raw artifact: `DEV-013-RAW-NOT-CONFIRMATORY` (artifact ID **11836120110**, ZIP SHA-256 `880d56e177b78d44961a8fc75e6e0a4151ebe6eeb7a55ce81afaa6c358172d65`). Its report has 15 trained model state dicts, source hashes, paired held-out predictions, reset/swap controls and per-arm summaries. Validator reloaded all checkpoints, checked all source and tensor hashes and regenerated **2,400 native TAC-OSM decision predictions** (as well as auxiliary bit predictions) from the held-out generator. It **does not** reproduce training optimizer trajectories, establish independence of the reused synthetic benchmark, or justify a SOTA claim. Prior DEV-003–012 evidence remains unchanged.

## Exact intervention

All five arms retain the same TAC-OSM causal DeltaMemory/ProtectedMemory state (4×8×24 = **768 scalars**) and actual learned top-2 expert selector, existing conditional executor with selective operator calls and halting, and native decision answer head. The new `CASMAlignedModel.predict()` passes through a refactored but numerically equivalent form of the original `predict()`; its parity is directly unit-tested. A separate two-way auxiliary classification head optionally taps the **same post-CASM features** with a neutral current bit (0). Training-only rehearsal queries are made from previous **observed** address/value pairs with fresh noisy views. Neither the online writer nor inference path sees oracle target index/role/answer.

Five models per seed use **identical architecture, total/trainable parameter count (38,732)** and byte-identical initialization. Three seeds 43, 59, 71, 120 matched AdamW steps, four paired-example episodes/step; shuffled labels are independent of observed payloads. Training treatments:
- `task_only`: original TAC-OSM CE, no auxiliary supervision.
- `separate_aux`: original DEV-012 sidecar head historical-bit CE, independent of native CASM path.
- `aligned_aux`: true historical-bit CE applied to features that pass through the native routing and CASM operators, allowing gradients to these modules.
- `aligned_shuffled`: same CASM path and cost, random null historical-bit labels.
- `aligned_detached`: real historical-bit CE, but auxiliary representation is detached **before** the head: auxiliary supervision does not update the writer/router/CASM, while task loss still trains them.

Training on history H=32/128 and competing writes D=4/16, with noisy near-address interference. Evaluation has 80 held-out opposite-past pairs (=160 episodes) per arm/seed, 2 reps of 40 registered DEV-012 factorial conditions including H=512/D=32. All 15 arm-seed models receive identical held-out visible inputs; no train/test fingerprint overlap. Training-time auxiliary CASM passes cost extra compute; no wall time/FLOP advantage is claimed.

## Exact held-out results (mean over three independent model seeds)

| Arm | Native TAC-OSM task | Auxiliary historical bit | Retention native | Acquisition native | H=512 native |
|---|---:|---:|---:|---:|---:|
| task_only | 50.83% | 48.96% (untrained head) | 49.58% | 52.08% | 51.04% |
| separate_aux | 49.79% | 59.58% | 50.42% | 49.17% | 50.00% |
| aligned_aux | **51.88%** | **60.21%** | 50.83% | 52.92% | 51.04% |
| aligned_shuffled | 50.63% | 49.38% | 50.42% | 50.83% | 51.04% |
| aligned_detached | 51.04% | 56.46% | 51.25% | 50.83% | 52.08% |
| zero-memory reset | 50.00% | 50.00% | — | — | — |
| full-history Euclidean | **85.00%** task accuracy | — | — | — | — |

Euclidean stores and scans every event, O(H+D), and is not persistent-capacity or compute matched. The auxiliary **diagnostic XOR** of the recovered bit with public current bit attains the **same** accuracies as the auxiliary bit head by task construction. This XOR is an *offline deterministic evaluator diagnostic*, **not** the output of the native TAC-OSM conditional execution and must **not** be counted as system task accuracy.

### Native task accuracy per training seed

| Seed | task_only | separate_aux | aligned_aux | aligned_shuffled | aligned_detached |
|---|---:|---:|---:|---:|---:|
| 43 | 51.25% | 49.38% | 53.75% | 51.88% | 49.38% |
| 59 | 50.63% | 50.00% | 49.38% | 50.63% | 50.63% |
| 71 | 50.63% | 50.00% | 52.50% | 49.38% | 53.13% |

Paired differences **aligned_aux − control** in native task accuracy; *exploratory* Student-t 95% interval across 3 training seeds (not 2,400 pseudo-independent partner examples):

| Contrast | Mean difference (percentage points) | 95% interval (pp) |
|---|---:|---:|
| aligned vs task_only | +1.04 | [-3.95, +6.03] |
| aligned vs separate_aux | +2.08 | [-4.19, +8.36] |
| aligned vs aligned_shuffled | +1.25 | [-4.35, +6.85] |
| aligned vs aligned_detached | +0.83 | [-6.83, +8.49] |

All primary intervals include zero. **No transferable improvement demonstrated.** Small positive mean gaps are driven partly by seed 43 and reverse direction on seed 59. Acquisition and old-retention both stay near chance; H=512 remains near 50–52% across all arms.

For auxiliary historical-bit accuracy, aligned_aux − aligned_shuffled = +10.83 pp, exploratory 95% CI [+8.46, +13.21], but aligned_aux − separate_aux = +0.63 pp [-8.02,+9.27]. Thus true auxiliary content supervision works at roughly 60%, but **sharing CASM did not establish superior historical decoding** to the independent auxiliary head. The aligned vs detached auxiliary gap is +3.75 pp [-6.43,+13.93], also inconclusive.

Every arm returned **exactly 50%** on both zero-state native task and zero-state bit predictions. State-swapped native task accuracy lies between 48.13% and 50.21%, consistent with weak history dependence, not robust usable decisions. Recurrent state responds to a flipped historical bit, but more demanding memory retrieval still fails relative to the unbounded full-history comparator.

## Scientific interpretation and decisive next gate

**Negative transfer:** Supplying historical-bit labels directly to the actual CASM computation **still does not reliably teach TAC-OSM's original answer head to make the correct action**. This refutes the simple explanation that DEV-012 failed *only because* its auxiliary head was separate. It **does not** prove CASM is intrinsically insufficient; training budget, stochastic discrete routing, answer transformation and recurrent interference remain confounded.

The next focused experiment should **separately assay the three boundaries** using the same 768-scalar writer and saved checkpoints:
1. **Address → decoded bit**: use frozen writer and same query with a native selector readout; report retrieval accuracy and age/collision strata.
2. **Decoded bit + public current bit → task action**: train and test an explicit *learned* two-bit adapter through CASM (not hard-coded evaluator XOR), with fixed supervised budget; compare direct task-head supervision, task-conditioned auxiliary bit supervision and detach controls.
3. **CASM routing/halting**: count actual operator calls and isolate whether selector's hard dispatch/halt prevents native XOR learning versus a dense same-parameter compute control. Do not silently bypass conditional computation in the evaluated candidate.
4. Only after robust task transfer, test learned writer redesigns (schema indexed or erase/write gates), holdout generators and verified/hardened closed-loop OSM.

No new architectural success claim, no CED performance claim, and **do not merge this PR as a positive result**. This is a small controlled synthetic development experiment stacked on DEV-012 #168 and earlier draft work.
