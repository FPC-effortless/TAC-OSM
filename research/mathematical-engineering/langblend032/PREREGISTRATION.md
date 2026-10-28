# LANGBLEND032 — fixed probability mixture of existing TAC and FIT-only P3

**Preregistered 2026-10-28 before any blend DEV scoring.** No-fit internal post-hoc diagnostic, not native architectural improvement. Existing LANGCOMPARE031 trained checkpoints (1701–1705) remain immutable. No TEST, GPUs or retraining.

## Question and model

Can a fixed normalized mixture of the existing frozen TAC model and the source-frozen 3-byte conditional-count P3 improve next-byte NLL beyond P3 alone on the same four *already exposed* DEV documents?

    p_lambda(y|h) = (1-lambda) p_TAC(y|h) + lambda p_P3(y|h)

Frozen lambda grid: **0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0**. Report every point for all 5 seeds. Never optimize lambda on exposed DEV to make a claim. Lambda=0 is original pure TAC; lambda=1 is P3 alone. This is probability mixing, not logit summation, new native TAC or a learned memory.

## Source and controls

- Real model evidence: the five immutable source-verified LANGCOMPARE031 original TAC checkpoints from GitHub Actions run 40780215655, all trained at 320 AdamW updates on the same genuine publisher FIT text. SHA256 checked and exact original weights/optimizer/RNG strictly cold-loaded before inference. Original checkpoints and their manifest are also independently preserved in the user's ChatGPT Library under /TAC-OSM/Archives/LANGCOMPARE031.
- Frozen P3: original source-verified langbase024 byte_next_reference.train_count_model and law(..., order=3), Dirichlet alpha=.5 and backoff tau=8. FIT only: 13 PSF CPython publisher documents and the UTF8 FIT supplement. No future information, DEV-derived counts, or owner boundary contamination.
- Evaluation: exactly langtrain025.train_byte_native.windows(dev), four *already exposed* DEV documents, eight 129-byte windows each, 4,096 original next-byte targets. At byte t, P3 reads only window[max(0,t-2):t+1], TAC reads only window[:t+1], target is window[t+1].
- Five seeds must ALL be recovered and original checkpoint SHA-verified, otherwise FAIL instead of reporting a partial winner. Fixed lambda grid and scores: NLL, bits/original byte, Brier, top-1/top-5 and per-document decomposition.
- Never open protected TEST or claim a new benchmark qualification. Hypothetical blend gains must be labeled hybrid inference gains. If P3 alone still wins, preserve failure and do not select a different lambda post-hoc.

## Later mathematical engineering

A subsequent, separately registered LATFUSE write-gate experiment can ablate a per-channel content-erase rule M_t=diag(g_t) M_{t-1} + eta_t v_t k_t^T, with actual Frobenius stability bounds, measured key collision/interference, unchanged 500K parameters (reallocated weights), a no-erase control, native-only NLL and matched training compute. A successful hybrid P3 blend does *not* show a successful native gated write.
