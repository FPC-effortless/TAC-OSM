# GRAPH014 — New 500K SSM with equal-cache TAC comparison on authentic, exposed DEV

**Date: 28 October 2026. Status: EXPLORATORY AUTHENTIC DEVELOPMENT. No sealed TEST, no architecture-superiority claim.**
Belongs to System-1/language within the existing six families. Graph009–013 archived results are immutable.

## What was trained
The original historical 500K SSM trained binaries are not recovered; therefore the 2026-10-28 work defines a **NEW**, separately versioned 498,288-parameter selective-diagonal SSM with input-dependent exponential decay, gated input writes and an input-only associative chronological parallel scan. It is not Mamba/S6 and must not inherit earlier reported SSM results.

Original TAC 500K: 499,290 actual active learned parameters, new SSM: 498,288, **0.201% gap**. Both use real Vim 9.1 30 original FIT whole documents, exact byte SHA checks, 120 AdamW optimizer updates per seed, batch 8, 128-character contexts, CE next-character objective, learning rate 0.001, and identical individual FIT sampling SHA in seeds 1701–1705. New SSM fully trained in this experiment; original TAC trained and frozen previously and its five original checkpoints restored/read-only in GRAPH013.

**Important resource mismatch:** single warm-step PyTorch 2.10 eager CPU operator profiling on the same batch8×128 constructed shape (forward+backward+AdamW) counted about **3.219 billion TAC vs 2.931 billion SSM operations**, an **8.94%** shortfall against TAC. Operator coverage is partial: these are *not complete certified training FLOPs*; execution scheduling and prior fits differed. Thus the comparison is NOT equal-compute and cannot establish a fair architectural winner. CPU runtime is an efficiency measure, not an equality requirement.

Five fresh original SSM checkpoint/optimizer/RNG/sampler archives saved locally; each reloaded with a restricted PyTorch loader, original 36 source file hashes checked and all original TAC FIT sampling schedule hashes matched. Exact original ZIP/weights have not been committed to GitHub, so **REPO_REPLAY=BLOCKED**; synthetic source-only CI is not real-data replay. No new TEST was accessed.

## Original authentic Vim9.1 EXPOSED DEV predictions (48 length129 windows; 6 chapters; 6,144 real labels/seed)

| Mean 5-seed next-char NLL ↓ | Original frozen TAC 500K | Newly trained SSM 500K |
| --- | ---: | ---: |
| Native neural model | 2.537438 | **2.518730** |
| Native + same GRAPH009 exact 3-gram count cache | 2.442860 | **2.430226** |
| Native + same exact + GRAPH011 trained approximate query | 2.445901 | **2.430577** |

The exact count cache (lambda .6, 512 prior source events, tau .05) improves **both** neural models on these same observed source contexts. The same frozen trained GRAPH011 2,816-weight context query, top8, temperature .25, additional lambda .1 **on exact misses only**, actually *worsens* aggregate DEV NLL for both models: TAC +0.003042, SSM +0.000352. Do not assert learned query always helps or call the Python3.13 external component result a full TAC backbone result. The shared sidecar creates 502,106 total learned TAC+query weights and 501,104 SSM+query weights, within 0.2% total neural counts, with external event storage/cost separate.

Native SSM advantage: 0.018708 nats/character; both-with-cache SSM advantage: 0.015324 nats/character. The original TAC wins the equal-cache NLL on **1 of 5 seeds**, not 5/5; original SSM version from historical FLOW 500K remains unmatched and its previously reported 2.39256 score is **not** from this new comparator.

Per-seed equal-cache NLL:
| Seed | TAC + exact + query | New SSM + exact + query |
| --- | ---: | ---: |
| 1701 | 2.482212 | **2.442081** |
| 1702 | 2.431588 | **2.428230** |
| 1703 | **2.434540** | 2.438662 |
| 1704 | 2.422032 | **2.398446** |
| 1705 | 2.459135 | **2.445468** |

All 5 frozen source windows use the exact same native labels/source bytes in both model arms, and a **single** per-seed shared count/query probability array is fused identically in the two arms. Before scoring, the independent count and query functions were required to agree exactly on the exact-cache eligibility mask; for each seed 809 exact hits, 5,143 approximate eligible misses and 192 no-hit contexts. Probability laws verified normalized and finite. Both native unmodified model scores independently matched each original restored checkpoint DEV NLL to 1e-6; all five new SSM checkpoint scores cold-reloaded locally.

## Mathematics and engineering
New comparator:
- z_t = RMSNorm(Embedding(x_t)).
- (d_t,v_t,r_t) = W_in z_t + b_in.
- a_t = exp(−softplus(d_t) ⊙ softplus(lambda)), 0 < a_t <= 1.
- b_t = (1−a_t) ⊙ tanh(v_t).
- h_t = a_t ⊙ h_(t−1) + b_t with h_(−1)=0.
- logits from gated recurrent read, skip path and 680-wide gated local MLP.
- Associative prefix composition (a2,b2) ∘ (a1,b1)=(a2 a1, b2+a2 b1). Construction preserves chronological order mathematically. FP32 scan not bit-exact to a sequential loop.

A local representative double-precision scan/serial agreement and active-gradient test passed; parameter count 498,288. The dedicated GitHub GRAPH014 source-only suite covers scan and serial derivatives, causal future poisoning and numerical shape/parameter contracts. Formal end-to-end Lean/IEEE/AD refinement is NOT proved by these tests. Prospective stronger SSM or Mamba runs remain needed.

## Research gates
- AUTHENTIC_SOURCE + ORIGINAL_FIT_SAMPLE_SCHEDULE: PASS locally (36 file hashes, 5 paired seeds).
- ORIGINAL_TAC_FROZEN_REPLAY: PASS locally on previously exposed DEV (GRAPH013).
- NEW_SSM_TRAINED_AND_LOCAL_REPLAY: PASS locally on previously exposed DEV (all 5 seeds).
- SAME_CACHE_PAIRED_DEV: EXECUTED; SSM wins mean and 4/5 equal-cache seeds.
- TRAINING_FLOP_AND_TUNING_PARITY: FAIL/UNVERIFIED; this is a provisional system comparison only.
- REPO_COLD_REPLAY: BLOCKED (new checkpoint originals not Git-held; original TAC binary likewise).
- SEALED_TEST + general long-context / semantic retrieval: NOT RUN.
- MODERN_MAMBA_COMPARATOR / production competitiveness: NOT ESTABLISHED.
- Complete Lean FP32/optimizer/AD proof: OPEN.

**Next:** commit Git/LFS original checkpoint archives and lawful clean-checkout source identity contract; independent replay on verified immutable published bytes; register stricter FLOP/equivalent data-objective experiments and a genuinely untouched corpus. Only then claim comparative merit. Do not reuse exposed Vim or prior Python files as fresh TEST.
