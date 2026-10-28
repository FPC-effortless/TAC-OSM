# TAC-LANG-LONG-500K-003 — causal historical retrieval training and state-reset falsifier

**2026-10-28. Scientific gate: FAILED.** Authentic human-authored Vim 9.1 FIT/previously-exposed DEV only; FIVE paired training seeds, TEN saved checkpoints, no sealed TEST. The model source is `research/language/flow_500k/historical_500k.py` (exact Git blob of locally trained source); streaming base is `tac_wave_ced_500k_stream.py`. **Original binary checkpoint bytes and full machine audit are preserved in the originating ChatGPT conversation artifact but not in this PR. Thus cold GitHub checkpoint replay remains OPEN.**

## Precise experiment

Original five 499,290-param wave/CED TAC checkpoints after the first 120 updates serve as the frozen initialization. Paired fine-tuning for an additional **120 AdamW optimizer updates**, batch **4**, **256-character** real human-authored FIT windows, same exact 120×4 sampled `document:offset` SHA per seed, same optimizer LR 0.0006, 5 seeds 1701–1705.

- **Dual-history arm:** **503,514 learned active params** (base + 32→128 history decoder); next CE +0.3×mean(CE at lag64 and lag192). The decoder reads the **prewrite** two-head 16×8 associative state with a known past ordinal. True historical character appears only in loss, never in query.
- **Next-character-only control:** **499,290 learned params**, same original base initialized at identical prior checkpoints, exactly the same fresh FIT samples and training schedule, next CE only. Relative param gap **0.846%**. Full FLOP parity / identical objectives do not hold; classify as **training-objective exploratory comparison, not fair architecture-superiority**.
- Real source: original 30 FIT and six previously exposed DEV whole Vim9.1 chapters, 36/36 SHA256 bytes verified. Eval 48 original DEV windows ×256 targets = **12,288 next-character targets/arm/seed**; lag64 targets **9,216**, lag192 **3,072**/seed. These are NOT 48 independent documents or independent confirmations.

## Actual five-seed heldout results

| Metric | Dual history | Same-budget next-only |
|---|---:|---:|
| Next-character perplexity ↓ | 9.830196 | **9.611984** |
| Next-character NLL ↓ | 2.285159 | **2.262838** |
| Next-char top1 ↑ | 38.7972% | **39.2090%** |
| Next-char top5 ↑ | 71.8457% | **72.4316%** |
| Categorical Brier ↓ | 0.751742 | **0.747111** |
| 15-bin ECE ↓ | **0.016822** | 0.022943 |

Next-only wins next-character NLL in **5/5 paired seeds**, mean auxiliary-minus-control ΔNLL **+0.022321** nats/character. The seed-bootstrap 95% percentile interval [+0.0107,+0.0339] uses five random initializations and does not imply independent-document confidence.

**Historical true source-target reconstruction:**

- lag64 source-trained decoder NLL **3.944214**, FIT-only smoothed character-frequency baseline **3.532250** (decoder worse).
- lag192 trained decoder NLL **4.400676**, FIT-only baseline **3.633239** (decoder worse).
- At t192..255, erase the true prior state at t128 but keep absolute ordinal and the same latter128 real input tokens: lag192 loss **4.400648** vs original full-state **4.400676**; full-state advantage **−0.000028** nats/character, better for only **2/5** seeds. This demonstrates **no learned use of the earlier 128 characters** for the 192-lag historical target in this test.
- Frozen next-character prediction on same last32 targets: full-history NLL **2.511571** versus reset **2.520064**, minor directional improvement 4/5 seeds. Does NOT qualify old-identifier retrieval.
- Wrong ordinal key and zero-read lesions can worsen history-decoder NLL but source-position *sensitivity* does not establish *correct old-content retrieval*. No future-label exposure, no synthetic primary task.
- **10/10 original local checkpoint score/no-refit and sample-sequence SHA recomputation PASS**, source-permutation guard/causality tests 4/4, current Torch source Git-blob byte identity PASS. No original binaries committed, so independent cold Git CI checkpoint replay still OPEN.

## Mathematical diagnosis

For equal input-conditioned schedules M_{i+1}=M_iP_i+Q_i and states differing only at a reset boundary s, the old-state contribution at later t is exactly ΔM_t=ΔM_s(P_s…P_{t−1}); the additive Q terms cancel. At a known positional key q, read difference is Δr=ΔM_t q. With normalized keys and α,ε in [0,1] and P_i=α_i(I−ε_i k_i k_iᵀ), each ||P_i||₂ ≤ α_i, yielding an *upper* bound on old-state influence and **no positive lower bound on recoverability**. Stable scan and phase-correct ordinal address are insufficient. Same-address full corrective write can erase an old value exactly. PNDS companion module `PNDS/LanguageHistory500K.lean` specifies scalar conditional retention/overwrite lemmas; Lean kernel/axiom auditing must complete before citing those theorems as compiled.

## Next qualification — do not promote this lag decoder

Preregister a real source-identified event/citation/identifier retrieval task with genuine observed source IDs, longer than 512 tokens, exact source-known read/write candidate identities, wrong-owner/key controls, same-target state-reset comparisons and downstream prediction utility. Compare against a full-history-attention and genuine source-index lookup reference; require retrieval accuracy/log-loss better than source-only prior and genuine measurable incremental next-token benefit, not mere address sensitivity. Match parameters 475K–525K, architecture gaps ≤1%, complete equal compute, equal objectives and real source splits, five seeds, original checkpoint cold GitHub CI, and independent never-opened documents. Preserve all negative gates, six-family scope and prior 100K results.
