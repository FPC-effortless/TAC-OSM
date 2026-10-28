# MLAG005 preregistration — real Vim language context320 matched horizon fit

Precommitted 2026-10-28 before fitted results. Exploratory five-seed matched objective study, **not independent TEST or production-representative language benchmark**. Current 100,680 active learnable parameters and original 408 persistent float values unchanged; no seventh TAC family.

**Why:** MLAG004 found a frozen 128-context-trained model's lag96 loss of 3.329 at current position96–127 but 4.811 at position128–191 on human-authored Vim9.1 DEV. A context320 experiment must compare **both arms freshly trained at context320** rather than using the earlier 128-context checkpoints as controls.

**Fixed protocol:**
- Source: byte-SHA-pinned original 30 whole Vim9.1 FIT documents, six already exposed DEV documents; no copyrighted source text redistributed, no protected TEST.
- Architecture: exact immutable `parallel_tac_language_gate.py` SHA256 `835212d74cc0d2d4d22ff7e75beb12771586cce58f2e63943690e5d859417457`, 100,680 active learned parameters, native8x8 + 2x16x8 aligned, SiLU source gate, trained multiscale single-lag output geometry.
- Seeds 1701–1705, batch8, **320-character contexts**, exactly **120 AdamW steps**, learning rate .002, weight decay .01, global grad clip1, exact same parameter init seed and per-seed **sample/target SHA256** across two arms; no parameter dummy counts or altered head.
- **Control A (fresh longer-context dual horizon):** `L_char+.15*L_history64+.15*L_history96`.
- **Candidate B (fresh longer-context tri horizon):** `L_char+.10*L_history64+.10*L_history128+.10*L_history256`.
- Total historical loss coefficient 0.30 in both arms; total **task mix differs intentionally**; updates and sampled data match. A third read for candidate causes **unequal training FLOPs**, so no equal-compute or equal-tuning claim.
- Eval: fixed four deterministic windows per genuine DEV chapter at length320, NLL and exact-character accuracy for next character and historical lags64/96/128/256; source FIT unigram prior (add0.5) and zero-source-memory plus wrong-address control for lag256, tested on identical valid label coordinates.
- Evaluate **per prediction current-position block**, including 96–127 vs ≥128, to isolate positional extrapolation; provide per-seed outcomes, actual training time, AdamW state, NumPy/Python/Torch RNG, hash receipts, cold reload equality.
- Predeclared exploratory admission gate: candidate lag256 NLL < FIT prior **and** source-aligned memory zero by at least 0.05 nat averaged across five seeds, correct-address better than wrong-query (+1/+8) on the **same target positions**, next-character NLL degradation relative to fresh longer-context control <=0.01, and no serious collapse at registered 64/96. Report all negative outcomes. Independently held-out TEST and cold GitHub CI binary replay remain mandatory before architecture promotion.

**Limitations:** Each lag is measured in ASCII/Vim **characters**, not semantic tokens. Document windows and current ordinal 0–319 are not conversational multi-session memory or general 512+-event learning. No fair head-to-head with SSM/Mamba/RWKV/GRU/Transformer; additional baseline implementation work needed. Do not tune these very DEV windows repeatedly to obtain success.
