# DEV-010 — Finite learned PLM with CED reader: negative development outcome

**Evidence status: REPLAY-VALIDATED-DEVELOPMENT-ONLY. No positive capability, model superiority, or scalable-compute claim.**

## Executed evidence and provenance

GitHub Actions run [38362662234](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38362662234) completed successfully. All seven new mechanism tests passed, the 958-test suite cardinality integrity gate passed, and both smoke and full-grid checkpoint-evaluation replay succeeded.

Archive `DEV-010-RAW-NOT-CONFIRMATORY` (Actions artifact ID 11795975699) contains the six final weights (`native`, `ced` for each seed) and `report.json` with source/checkpoint SHA-256, train-input fingerprints, pre-repair outputs and reset/swap controls. Replay regenerated **720 model-episode predictions** across 3 seeds × 3 inference modes × 80 episodes, from six checkpoints. This proves neither independent training reproduction nor science preregistration, nor unseen-domain validity. The training budget and experiment design are *exploratory*.

## Exact intervention and fairness

Uses existing TAC-OSM `ProtectionModel(mode='always_update')`: learned gated delta writer, 4×8×24 = **768 recurrent-state scalars**, learned memory read, original conditional-execution module and answer head. Extra learned CED readout uses permutation-equivariant expert-set encoding, encoder-projected shared KV and two query-only decoder reads. **Native-vs-CED parameter counts and FLOPs are unmatched**, so no efficiency/superiority inference from that contrast.

Training 72 steps per seed 43,59,71 with same balanced opposite-past pairs and all common PLM/compute/answer weights copied byte-identically at initialization. CED trained in dense mode once. **The same trained CED weights** were evaluated with either full four-expert decoder reads or top-two sparse access on the second decoder layer. No different optimization or gold expert labels between dense/sparse evaluation.

Evaluation: train/eval fingerprints disjoint. Eight extra H=512/D=32 stress cells supplement H=32/128, D=4/16, near fraction 0/1, noise 0.08/0.25 and retention vs acquisition. Pair members have identical present inputs and opposite historical target values. Reports native, dense CED, sparse CED and zero-memory reset on each case.

## Raw held-out decision accuracy

| Seed | Native | Dense CED | Sparse CED |
|---|---:|---:|---:|
| 43 | 50.00% | 47.50% | 48.75% |
| 59 | 51.25% | 50.00% | 51.25% |
| 71 | 48.75% | 56.25% | 52.50% |
| **Mean** | **50.00%** | **51.25%** | **50.83%** |

Each cell consists of an opposite-past pair; **80 episode decisions per mode per seed**, 240 per inference mode total. Exact 50% reset-control accuracy is expected by pair construction, because the current inputs are identical while the target is flipped.

Exploratory paired seed-level differences (95% Student-t df=2):
- Dense CED – native = +1.25 percentage points, interval **[-12.29,+14.79]**.
- Sparse CED – native = +0.83 points, interval **[-5.63,+7.30]**.
- Sparse – dense CED on same weights = -0.42 points, interval **[-7.59,+6.75]**.

With only three independent seeds and substantial uncertainty, there is no defensible advantage.

### Stratified accuracy, pooled 3 seeds

| Condition | Native | Dense CED | Sparse CED |
|---|---:|---:|---:|
| Retention | 50.00% | 53.33% | 52.50% |
| Acquisition | 50.00% | 49.17% | 49.17% |
| H=32 | 50.00% | 52.08% | 51.04% |
| H=128 | 48.96% | 51.04% | 51.04% |
| H=512 | 52.08% | 50.00% | 50.00% |

**All models remain close to chance, including long-horizon decisions.** Full-grid final training CE values are around 0.68–0.71; no strong learned discrimination has emerged.

### State sensitivity is not target information

Paired opposite-past states have nonzero L2 distance (mean around 0.20–0.28 across seeds). This demonstrates sensitivity of the internal state to a changed historical payload, **not** that the correct bit remains linearly/nonlinearly decodable or is preserved under competing writes. The evaluation did not train an independently frozen-state decoder probe; attributing the failure definitively to erasure vs readout would be unjustified.

### Computation and interpretation

The CED readout sees four expert slots. In a two-sample batch, 32-wide representations, two heads:
- Encoder attention: 64 logical query–key score pairs in both inference modes.
- First query decoder: 16 score pairs in both.
- Second query decoder: 16 dense versus 8 sparse.
- Total: **96 dense vs 88 sparse = 8.33% fewer of these counted score pairs**. Both modes store the same 512 global KV scalars per two-sample read. This is **not** actual FLOP, GPU time, full memory, or an inference cost improvement guarantee.
- The learned writer still processes **every** observed history/write event, O(H) work. The tiny second-read score reduction does not establish sublinear end-to-end TAC-OSM computation.

## Decision

**NEGATIVE.** In DEV-010, attaching CED to the actual 768-scalar learned memory does not reliably improve task quality. The failure remains *upstream or downstream of the readout*, not yet causally localized. The next decisive measurement is a **frozen-memory trained probe** comparing unambiguous isolated targets against competing-write histories, plus capacity-matched modern recurrence controls and stronger gradient tracing. Do not simply enlarge the CED.

For forward research, consider controls from [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791) and [SchemaMem](https://arxiv.org/abs/2609.33436) for targeted update/retention, after prospectively fixing a fair benchmark; these are independent literature precedents and not demonstrated improvements in TAC-OSM.

Prior negative and mixed DEV-003 through DEV-009 reports remain unchanged. PR is deliberately draft and stacked, not merged.
