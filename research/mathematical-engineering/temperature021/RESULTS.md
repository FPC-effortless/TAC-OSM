# TEMP021 — recovered original 500K checkpoint scalar calibration on authentic EXPOSED Vim DEV

2026-10-28 · System 1 language; stacked on TEMP020 draft TAC #549 and unchanged PNDS Lean #61. **Evidence tier: genuine original-model frozen inference, previously exposed development only. Not a new model/independent test/intelligence gain.**

## Recovery, protocol and integrity

Local original user-owned archive \`tac_language_long003_replay.zip\` SHA256 \`cde9a91be40a4e613975c0bd8caa4c19ac24e3d1d0e993db5cacfbd002fecc32\`. Original unchanged WaveCED source SHA256 \`acf80cb9051b7f2d2c3900d1676eed212890208cd5213088c6751f9cfd24392c\`. Original source manifest bytes SHA256 \`4c7703dca3c5920055086687e07c24098766a2301b773cfb99c34f0a394a4b32\`; **36 of 36 original publisher Vim9.1 document byte SHA256** independently verified. Five original 499290-parameter trained checkpoint bytes match their original independent in-ZIP hash manifest, and original state dicts, optimizer and saved RNG metadata remained unchanged.

Frozen study role split preregistered in Git commit \`b00c89664d187a13b634e1f2b84006aae9937e52\` BEFORE scalar optimization:
- CALIBRATION: \`usr_45.txt\`, \`usr_50.txt\`, \`usr_51.txt\`;
- SCORING (no overlap): \`usr_52.txt\`, \`usr_90.txt\`, \`usr_toc.txt\`.

Exactly 8 original 129-symbol windows/document at source positions \`np.linspace(0,N-129,8,dtype=int64)\`; 128 next-character predictions/window; **3,072 targets/seed for CALIBRATION and 3,072 distinct-document targets/seed for SCORE**. Every model first replayed its original six-chapter/6144-target DEV NLL with maximum numerical drift **1.02e-7**, below the 1e-6 bound. All trained model weights frozen, no optimizer steps, no GPU or protected TEST. FP64 positive scalar inverse-temperature a in [.25,4], 64 fixed gradient-root bisections and identity fallback; fits only the three CALIBRATION chapter labels, applied directly to frozen logits for SCORE. An independent genuine-position scalar loss/derivative/curvature cross-check agreed within 1e-10. Class ranking unchanged by positive a.

## Actually observed original 500K exposed-DEV measurements

| Seed | Inverse temperature a=1/T | CAL identity NLL | CAL fitted NLL | SCORE identity NLL | SCORE fitted NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1701 | .932810 | 2.523515 | 2.518167 | 2.645121 | **2.638219** |
| 1702 | .920328 | 2.472782 | 2.465591 | 2.578313 | **2.571417** |
| 1703 | .949388 | 2.463556 | 2.460575 | 2.573827 | **2.569345** |
| 1704 | .932899 | 2.452918 | 2.447664 | 2.565940 | **2.559589** |
| 1705 | .897990 | 2.476572 | 2.464223 | 2.621837 | **2.603317** |
| **Five-seed mean** | — | **2.477869** | **2.471244** | **2.597007** | **2.588377** |

For the disjoint **SCORING** chapters, mean NLL reduction **0.008630 nats per mapped character** (0.33% relative); score categorical Brier **0.820480 → 0.817167**. Top-1 accuracy **0.314323 → 0.314323**; top-5 **0.663021 → 0.663021**. This is calibrated probabilistic confidence only, not more correct next-token class selections.

Per-document SCORE mean fitted minus identity NLL over five seeds:
- \`usr_52.txt\`: **−0.0177494**;
- \`usr_90.txt\`: **−0.0049967**;
- \`usr_toc.txt\`: **−0.0031444**.

All three SCORE documents improve in this limited same-manual diagnostic, but one fitted CALIBRATION document (\`usr_45.txt\`) worsens by +0.000809 after fitting the pooled 3-chapter sample; this is disclosed rather than hidden. This is NOT evidence of eight or fifteen independent cohorts; only three SCORE chapter clusters from one author/publisher/corpus, and repeated fixed labels across five trained seeds.

## Code, artifacts and gates

\`replay_original_500k_calibration.py\` is an executable all-original-checkpoint replay with hard fail on archive/model/36-source/checkpoint SHA mismatch, source role overlap, original six-document NLL drift, wrong parameter count and nonpositive scalar/rank change. Original source is loaded *from the SHA-pinned archive*; it is not reimplemented or silently altered. The detailed locally produced original per-seed/source receipt JSON SHA256 is \`4c197b5623b6dca74f7c8a25ba6931b7d83b25b1198a66898458860c672c34cd\` and belongs in the downloadable researcher archive. GitHub has report plus a replay runner, but **not the original 84MB archive/trained .pt or lawful original source files**: REPO_COLD_REPLAY=BLOCKED. Receipt alone is not independent proof.

The experiment applies TEMP020 mathematics without modifying PNDS #61 Lean4 proofs. Algebraic exact-real derivative/nonnegative variance and positive-temperature ordering remain established; full model AD, float-precision and uncertainty/generalization proof OPEN. The original 128-symbol lossy mapping (bytes >=128 become \`?\` ID63) means these are **nats and bits per mapped symbol, NOT raw bits per byte**. Adding a fitted scalar adds one extra deployed calibration coordinate if counted in a deployable model. Original neural parameters 499290 remain unmodified.

**Result classification:** LOCAL ORIGINAL CHECKPOINT INFERENCE PASS; AUTHENTIC EXPOSED-DEV DISJOINT SCORE NLL/Brier IMPROVEMENT; TOP1/TOP5 UNCHANGED; ORIGINAL GIT BINARY REPLAY BLOCKED; NEW TEST NOT_RUN; GENERAL LANGUAGE INTELLIGENCE GAIN NOT_ESTABLISHED. Do not promote, merge or retune from this same scoring split.
