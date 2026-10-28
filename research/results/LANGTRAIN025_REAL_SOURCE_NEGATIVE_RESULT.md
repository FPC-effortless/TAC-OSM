# LANGTRAIN025 — genuine raw-byte native TAC training: negative paired development result

**28 October 2026 • internally measured System-1 language / authentic publisher PSF CPython3.13.0 original, PREVIOUSLY EXPOSED DEV. NO new protected TEST.**

## Scientific chronology
- Prospective five-seed, 320-step, 499978-param, no-hyperparameter-sweep protocol frozen at commit \`274e7ad8eba399a910a7851c49406b236522f5c1\` **before any new model fitting**.
- Initial automated CI attempt [40756182685] STOPPED before fitting because source rights/split metadata were absent from the manifest. Historical negative preflight is preserved. No original model started on that attempt.
- Repaired the **metadata only** in \`d55b32a2a34f6b5418adb0ae91f396f9fd725b11\`, preserving all model/data/training hyperparameters; passed strict author/publication original-source metadata and owner-500K scoped model prefit gate.
- [Genuine first trained GitHub CPU job 40756443660](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40756443660) **SUCCESS**. Six constructed byte/source-boundary, graph-AD, normalized prediction and causal-prefix controls passed; then five authentic publisher-data model fits 320 AdamW optimizer steps/seed, with original FP32 weights, optimizer/RNG and sample order saved; all five loaded into fresh model instance and reproduced the local score. CPU only, no GPU/TEST.
- [NEW study checkpoint and original numerical evidence artifact 12846836297](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40756443660/artifacts/12846836297), compressed ZIP 27,699,963 bytes, ZIP SHA256 \`87ebc3e69dbf0526d8093b264c37fe0e6c8a65ff04c10c4f66b7bfe8233e31e5\`, contained original numerical receipt SHA256 \`d25d61a8ae848227cdbf9424854e51a379afdcc3c0dc79c0bbbf160f2e105458\`. All five checkpoint bytes match their independent SHA in the receipt. Hosted GitHub artifact retention is finite (30 days); do not confuse it with durable Git/LFS permanent checkpoint hosting or user authorization to publish previous ORIGINAL weights.

## Exact 4096-label matched-case evaluation
Genuine original publisher CPython3.13.0 docs, original SOURCE023 thirteen FIT chapters + separately registered Unicode FIT supplement; FOUR previously exposed DEV chapters, each with EIGHT frozen 129-byte windows and 128 genuine next-byte predictions each. **4096 raw-byte targets per model seed**. Model starts from reset state per window and sees current+previous observed bytes only. P3 sees the same 4096 next-byte labels, has only up to three preceding actual bytes, FIT-only learned counts and fixed hierarchical tau8. The previous LANGBASE024 score on **69542 different position set** must NEVER be compared numerically to this 4096-case study.

| Seed | 499978p native TAC NLL (nats/raw byte) | exact case-matched P3 |
|---|---:|---:|
| 1701 | 2.112524645 | 1.877306112 |
| 1702 | 2.100032859 | 1.877306112 |
| 1703 | 2.077199273 | 1.877306112 |
| 1704 | 2.097086079 | 1.877306112 |
| 1705 | 2.109903249 | 1.877306112 |
| **mean** | **2.099349221** | **1.877306112** |

**PRIMARY PREREGISTERED DECISION: FAIL**, 0/5 seeds outperform FIT-only P3. Native NLL is **+0.222043 nats/raw byte** worse than P3. Native mean Brier **0.69781456** vs P3 **0.61605448** (WORSE); native top-1 accuracy **43.7012%** vs P3 **51.7090%** (WORSE); native top-5 **75.0342%** vs P3 **82.6172%** (WORSE). The native model does beat the weaker FIT-only unigram P0 on same raw byte positions (P0 NLL **3.497351821**), which confirms source-symbol statistics learned but not competitive native memory inference.

Per-document mean native TAC vs identical-case P3 NLL:
- \`controlflow.rst\`: 1.947519 vs 1.713909
- \`index.rst\`: 2.242744 vs 2.052049
- \`introduction.rst\`: 2.066796 vs 1.774381
- \`venv.rst\`: 2.140338 vs 1.968885

**ALL FOUR real publisher document means favor P3.** The selected 32 windows contain **ZERO nonASCII target bytes** even though the full publisher DEV contains 13 rare originals. We therefore observed NO authentic rare/nonASCII label prediction outcomes in this model study. The source-specific Unicode FIT supplement's 62 rare examples do not imply multilingual learning.

## Architecture, math and limits
- 256 raw byte IDs, 499978 trained/deployed learned coordinates, 501 FFN width. New from-scratch, not imported original 128-symbol checkpoints.
- Nonuniform six-concept posterior readout \(w_i\propto\exp(\operatorname{mean}_j G_{ij})\) fixes the original exact *uniform-pool graph cancellation*. The 6x6 Hessian \(H=1.25I+.25L\) stays SPD for symmetric nonnegative graph affinities, \(\|H^{-1}\|_2\le .8\) in exact real arithmetic. Program tests establish q/k gradients nonzero and prefix-causality; no full finite-precision/AD/Lean mathematical correspondence.
- 320 updates × batch8 × context128 per seed, actual original FIT positions sampled with replacement (327680 target-position presentations/seed) from 14 source docs, not 327680 unique bytes. Classical P3 fits full genuine FIT counts, so source exposure and training compute are NOT matched. Only a practical comparison to a simple strong solution; no SSM/GRU matched neural architecture contest, no heldout competency gain.
- The original 499290-parameter lossy 128-symbol Vim9.1 checkpoints and their TEMP021 calibration remain immutable and cannot be compared numerically across different corpus/tokenizer/target case sets.
- Actual publisher-source source boundaries and checksum validations were applied at fit; state reset at each sampled doc window. No future target fed to model forward. No GPU, customer data, physical decisions or trained multilingual capability claims.

**Status:** ENGINEERING_SOURCE_AND_TRAINED_REPLAY=PASS, MODEL_FIT=PASS, NATIVE_VS_P3_PRIMARY=FAIL, AUTHENTIC_EXPOSED_DEV_NOT_INDEPENDENT, TRAINED_SSM_MATCHED_COMPUTE=NOT_RUN, FULL_PNDS_PROOF=OPEN, NEW_PROTECTED_TEST=NOT_RUN. No model promotion.
