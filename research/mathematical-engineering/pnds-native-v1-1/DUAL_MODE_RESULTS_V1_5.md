# TAC-OSM v1.5 | Dual-mode world-model engineering and real research status

Research date 2026-10-25. Original TAC draft PR #389 and PNDS draft PR #7 remain unmerged. Full formal companion: PNDS foundation/v1/DUAL_MODE_STREAMING_V1_5.md.

## 1. One finite-state graph, two modes

The fully-parallel categorical candidate reuses 101,527 source parameters but *changes* the original nonlinear fast/slow recurrence. A second, continuous next-state/action-conditioned robot world model has 93,324 parameters. They are distinct experiments; do not treat synthetic categorical task recall as physically measured world prediction.

The candidate stream state consists of a causal fast-basis sum (88 FP32 values), an addressed associative store (8x8 FP32) and an integer event counter. This is 152 FP32 values = 608 bytes per sample, excluding trained weights and runtime buffers; size is independent of sequence length.

The causal batch route precomputes learned gates without future observations and carries each event as an affine pair (P,Q). The right-affine state law is M_next=M_previous P+Q, with P=alpha_eff (I-beta k k^T) and Q=beta v k^T. Chronological pairs compose associatively; balanced scan has O(T) 8x8 matrix work and O(log T) parallel depth. Streaming applies the equal exact-real rank-one law at O(8^2) work/event without storing the history.

## 2. Original-source runtime and Lean evidence

- [TAC source dual-mode run #40142256798](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40142256798): PASS 334 categorical full parallel/batched vs fixed-state streaming checks, including gradients, original source-trace comparisons, 2,048 state steps and max intermediate difference 3.2782554626464844e-7.
- Same run: PASS 146 continuous physical world graph vs fixed-size stream and gradient tests, max trace difference 2.384185791015625e-7. Real-world continuous model trainable parameters 93,324; GRU control 93,454 (0.139% gap).
- Single CPU thread, batch size 4, FP32, 80 samples per regime: measured step median 0.652790 ms after history 0, 0.656021 ms after 128 events and 0.654352 ms after 2,048 events. P95 0.677660, 0.677360 and 0.683352 ms. Hardware-bound latency, fusion, deployment, full training costs remain OPEN.
- PNDS/ParallelMemory.lean proves matrix affine chronology and associativity. PNDS/DualModeMemory.lean proves exact-real stream-fold vs scan result and resumed chunk consistency. Source tensor and gradient parity use finite-precision tests, not a complete Lean full neural tensor proof.

## 3. Authentic robot training — original failure and honest corrective pilot

LeRobot ALOHA Mobile Cabinet, actual publisher source dataset revision 7a752b39f7e69de7e38aee485a6bab07528a061a, original tabular SHA256 a3409d3282b2cf9a94d32b301756fbd15dca5862cc1c32973c7316dd8853c57a. 85 real episodes, 127,500 original frames, genuine 14-dimensional observed states/actions and next-state targets. Episode disjoint TRAIN 0–58, VAL 59–71, original TEST 72–84. Source windows TRAIN=590, VAL=130, TEST=130. Fixed 64 AdamW updates, equal real data samples and optimizer steps, trainable parameter gap 0.139%. Critically full TRAINING FLOPs **not matched**, so no scientific architecture-superiority claim.

First direct-state readout [real source run #40141901322](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40141901322): original test normalized next-state MSE TAC=0.13621579, GRU=0.11862875 and persistence=0.00043685. Native did NOT beat persistence.

Train-only observed-motion-scaled residual correction (both TAC and GRU), [real source run #40142256755](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40142256755): descriptive next-state MSE TAC=0.00036728, GRU=0.00036988 and persistence=0.00043685; validation MSE TAC=0.00042043 and GRU=0.00042542. **Original test cohort had been exposed before the residual design.** These are real development metrics but NOT untouched test qualification. No independent skill gate is passed. The real pilot CI is now only explicitly dispatchable to avoid silent repeated access.

## 4. Actual actor-identity memory admission bottleneck

Fixed original-source INTERACTION multi-agent case evidence from ESWAM012 source commit d9db3a0ac68d2bbabe3febe6cfe3d40bf322ea1d. Dynamic entity admission is separately implemented at fixed eight slots, with external original actor IDs and time-observed relevance. No future target, future labels, source-model predictions or pseudo-action rewards are used.

[Authentic identity mechanism run #40142606195](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40142606195): 266 distinct real observed current-frame inputs from 37 actor clusters, 61 accepted writes, 205 denied, four evictions, no unrelated identity memory erasure. Sparse previously exposed DEV anchors are not complete original 256-frame training histories. This rejection bottleneck is a negative result and is NOT proof of useful long memory. Entity admission is NOT yet integrated into a jointly trained full-parallel world model; dynamic eviction requires a precomputable routing contract or an acknowledged sequential admission microstep.

## 5. Remaining gates

The existing authentic ESWAM012/QUAL001 track found that native 256-frame memory lost to native 8-frame memory, so merely longer context is not evidence of useful retention. The historical 17 indexed variants remain unchanged and are NOT fully source-proved in Lean. Global full-tensor neural numerical/gradient proof, implementation of memory admission in a jointly trained world model, fully matched training FLOPs/data/parameters, new independent untouched site, useful future decision cost and production GPU real-time kernels remain OPEN.

No model/checkpoint is promoted or merged. Keep original negative evidence immutable. S1 conditional algebra/S2 executable source evidence improved, S3 complete whole-model proof and S4 independently qualified real model skill are still unestablished.
