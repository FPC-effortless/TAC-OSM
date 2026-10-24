# TAC-OSM native dual-mode research: bounded evidence + controlled retention v1.6–v1.7

**Research date:** 2026-10-25.
**Authority:** Original TAC source DIAGKEY004 commit 081e534a074001c6a3b2251b3ff31ad20ec4fca8, existing continuous world-model development branch, reciprocal PNDS draft PR #7.
**Model status:** Source-inspired parallel causal world-state variants with 93,324 trainable parameters and fixed 88+64-float hidden state. NOT pretrained native-recurrence equivalence, independent world-model superiority, or a fully formalized 17-variant family.

## Hypothesis and mathematical solution

The earlier continuous TAC-OSM memory gate schedule, measured AFTER fitting to genuinely recorded ALOHA actions/states, had mean alpha_eff≈0.44–0.47. An actual product of consecutive ten learned alpha_eff coefficients had mean only 0.0003–0.003. However, this quantity measures the **scalar forgetting component**, not the complete factual read singular values or a learned-memory retrieval score.

We implemented TWO new, clearly distinguished hypotheses without trainable parameter additions:

1. **v1.6 four-band stable evidence:** Replace old all-history mean accumulator A_t/n_t with 88 coordinates of fixed bounded causal EWMA, decay band rates 0.125/0.0625/0.015625/0.00390625 repeated for 22 channels each. Batch uses a genuinely associative scalar-affine O(T)-work/O(logT)-depth prefix scan, streaming updates one fixed-size 88-vector; native 8×8 rank-one memory is unchanged. For |input basis|≤1 and g∈[0,1], the update h'= (1-g)h+g x preserves [-1,1] and is nonexpansive in h.
2. **v1.7 controlled retention/write gate:** Preserve the original continuous 8×8 addressed delta rule, source neural maps and 93,324 learned parameters; transform pre-memory gates using a=sigmoid(raw_a), b=sigmoid(raw_b), keep=1-(1-a)/50, beta_eff=b/4, alpha_eff=keep(1-beta_eff/4). The source-independent-gate right-affine matrix P=alpha_eff(I-beta_eff k kᵀ) remains time-parallel-scan-compatible, the stream still rank-one updates. For normalized key/value and old Frobenius norm≤4, ||new M||_F≤alpha_eff ||M||_F + beta_eff ≤4. The scalar alpha_eff≥147/160=0.91875 **does not imply the entire right-affine operator has minimum singular value 0.91875**, nor that stored facts are recalled correctly. Corrections along k can overwrite old information.

Neither candidate converts TAC-OSM into a Mamba SSM. Both retain its native addressed value-by-key associative matrix and original neural feature maps. No source weights were replaced in original master.

## Exact software checks

[Original-source verification CI #40144539674](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40144539674) PASSED:
- v1.6: 294 forward, intermediate, gradient, causality, scan associativity and 4,096-step bounded fixed-state checks. Maximum source-shaped parallel-vs-stream difference 2.384185791015625e-7. Same parameters 93,324, 152 FP32 hidden scalars =608 bytes per sample (plus step counter; excluding weights, activations and runtime).
- v1.7: 162 batch/stream, source parameter mapping, full model-gradient, 4,096-step stability and gate-bound checks. Largest tensor difference 2.384185791015625e-7. Gate ceiling beta_eff≤0.25, scalar alpha_eff≥0.91875 for all sigmoid inputs.
- Previous source 26 operator, 40 whole-native numerical, 1,100 original bounded-state, 184 original-compatible, 130 parallel associative and 230 full-time-parallel candidate checks remain present in the same source-verification workflow.

Lean formal mathematical work, PNDS/TimescaleEvidence.lean, PNDS/JointAffineTAC.lean and PNDS/RetentionBudget.lean:
- bounded/nonexpansive exact-real temporal evidence, scalar associative affine scans;
- joint 88D evidence + 8×8 memory scheduled operator product and stream/scan correspondence;
- bounded controlled retention scalar coefficients and conditional radius-four norm invariant;
- arbitrary deterministic downstream decoder yields same mathematical output given *identical* scheduled state;
- remains OPEN: source-PyTorch-to-Lean full nonlinear tensor graph, differentiability/FP error cert, actual source 17-variant mathematical refinement, learned memory usefulness.

## Actual real robot model findings — original source, NOT synthetic evaluation

Publisher source: LeRobot ALOHA Mobile Cabinet; immutable dataset revision 7a752b39f7e69de7e38aee485a6bab07528a061a; parquet SHA256 a3409d3282b2cf9a94d32b301756fbd15dca5862cc1c32973c7316dd8853c57a.
- 85 real episodes available; 59 TRAIN, 13 previously exposed DEV, 13 existing previously exposed TEST episodes **not used or rescored in these v1.6/v1.7 experiments**.
- Each v1.6/v1.7 experiment has 472 real TRAIN history windows and 104 original DEV windows, 64 frame context, same 93,324 trainable parameter counts, identical per-seed original weights, same exact original-state/action source windows/updates (64 AdamW updates; two reproducible seeds 20261025/20261026).
- v1.6 experiment [real original DEV run #40144022948](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40144022948). Real normalized DEV MSE (mean history vs EWMA) seed1 0.000391491 vs 0.000409109, seed2 0.000382060 vs 0.000404562. **The stable timescale model loses on both seeds.** Removing temporal context worsens, removing 8×8 memory read barely changes error.
- v1.7 experiment [real original DEV run #40144539727](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40144539727). Original vs controlled gate MSE seed1 0.000391491 vs 0.000392125, seed2 0.000382060 vs 0.000382458. **Scalar retention improved without predicting real outcomes better.** Actual compound ten-gate mean increased from 0.000414791→0.6551933885 seed1 and 0.0029869734→0.6406611204 seed2. Disabling associative address/read produces small/inconsistent DEV differences, so evidence of *useful retrieval* is NOT established.
- Each experiment writes four original data-trained PyTorch model/optimizer/RNG/schedule/source provenance checkpoint bundles to GitHub Actions artifacts (rather than altering existing frozen research checkpoints). The training source is genuine, but the DEV was previously observed, **full training FLOP budgets not matched**, and no independent site or world-action control gate is passed.

## Research verdict

- **S1 mathematics**: stronger for paired affine operators, state envelope and joint dual-mode graph.
- **S2 engineering**: both full-sequence parallel and per-event fixed-memory streaming modes validated, same parameter budget; checked extreme histories.
- **S3 complete Lean whole-model refinement**: OPEN, including full original 17 variants, full decoder nonlinear tensor and finite precision/source equivalence.
- **S4 capability/transfer**: OPEN. Making memory survive is necessary for some tasks but absolutely not sufficient to encode decision-relevant delayed information. Neither timescale nor naive retention gate improved real ALOHA one-step world-prediction.
- The much stronger known negative ESWAM012/QUAL001 original traffic results (native 256 frame losing to 8) remain unchanged. Do not invent artificial memory tasks to claim real-world improvement. The next empirical gate must identify an authentic delayed-information-dependent target where current observation/action/prior cannot suffice, with original stable entity IDs and verified downstream outcomes, then train a read/write controller through those real delayed losses. Hold out untouched physical sites/episodes and certify equal parameter, genuine data/exposure and full training FLOPs. CPU time equality is **not required**.

Both original pull requests remain draft/unmerged. No trained neural checkpoint is promoted based on unqualified DEV results.
