# GRAPH016 — Causal online evidence mixture, real-source research

**2026-10-28 | Mathematical operator verified; source-conditional NLL benefit on exposed Vim; NEW Python first-open primary outcome FAILED.** Existing System-1/language family only. GRAPH009–015 authentic positive/negative results remain unchanged. **No native model retraining** and no new claim of learned human-like intelligence, native associative memory superiority, planning, customer outcomes, or deployability.

## Problem and solution

GRAPH014 showed the same external source retrieval helped original TAC and SSM, but the learned approximate query harmed already exposed Vim DEV while helping CMake. GRAPH015 inserting a stable error-correcting delta into all native TAC memories hurt predictive quality; mathematically stable write algebra is not sufficient.

GRAPH016 tests a **classical Bayesian log-loss expert mixture** for *prediction-time evidence arbitration*. For each source window and each frozen neural backbone independently, offer exactly three strictly positive normalized predictive laws:
1. Native backbone next-byte law, unchanged frozen weights;
2. Backbone plus GRAPH009 exact 3-gram source cache (capacity512, tau=.05, exact weight=.6);
3. Law (2) plus original GRAPH011 2,816-weight frozen learned approximate query *on exact misses only* (top8, temp=.25, additional weight=.1).

The source-only cache and approximate query are computed once from a lawful preceding observed prefix and shared between TAC, delta TAC and SSM. For an owner, before y_t is observed:
  w_t,i = pi_i prod_{s<t} p_s,i(y_s) / sum_j pi_j prod_{s<t} p_s,j(y_s)
  p_mix,t = sum_i w_t,i p_t,i.
Once the actual next byte is observed, update in stable FP64 log-space; never use y_t in p_mix,t. Initialize pi_i=1/3 and RESET on each source window. Owner identifiers are supplied by caller and are **not authorization credentials**.

### Exact mathematical guarantee and limits

For any sequence of positive normalized, causal expert forecasts:
  L_mix = -log(sum_i pi_i exp(-L_i))
  <= min_i (L_i - log pi_i)
  = min_i L_i + ln(3)   [uniform three experts].
It holds **per actual 128-target owner/window** against the best *fixed* expert over that whole window, not the best model on every time step, not against an SSM that is not in TAC's own expert set, and not against an arbitrary model-switched oracle. A single owner/window has average excess upper bound ln(3)/128 ≈ 0.00858 nats/byte, before FP64 roundoff. With 64 independent resets the total excess bound sums 64 ln(3), not just ln(3). A pathwise regret guarantee does NOT promise a gain over the best fixed retrieval setting: the cost of initializing with uniform weights is real. FP64 roundoff is validated numerically; complete Lean/IEEE/AD refinements are OPEN. This is classical Bayesian aggregation, not a novel TAC recurrence. Sources: Hutter, JMLR 4 (2003) https://www.jmlr.org/papers/v4/hutter03a.html ; Orseau/Lattimore/Legg, PMLR 76 (2017) https://proceedings.mlr.press/v76/orseau17a.html.

## Real already exposed Vim9.1 DEV diagnosis (NOT new independent confirmation)

Original human-authored 30 FIT, 6 DEV chapters. Genuine 6,144 next-byte labels/seed, 5 seeds, 48 state-reset windows/seed. Original checkpoint SHA256, source bytes and original original frozen DEV native NLL verified locally; 500K TAC/SSM/delta original weight bytes are not Git-hosted. All methods evaluated on exactly the same real labels.

| Backbone, 500K | Native NLL | Exact-cache NLL | Exact+query NLL | Causal Bayes-mixture NLL |
|---|---:|---:|---:|---:|
| Original TAC | 2.537438 | 2.442860 | 2.445901 | **2.432678** |
| Delta TAC | 2.563868 | 2.467237 | 2.469636 | **2.457090** |
| New selective SSM | 2.518730 | 2.430226 | 2.430577 | **2.417278** |

Relative to each model's strongest *fixed, already-known* source-cache policy on these exposed documents, the online mixture improves NLL by 0.010182 (TAC), 0.010147 (delta) and 0.012948 (SSM). All five seeds improve within each architecture. These are exposed-development outcomes, *not* an independently confirmed new capability or native learning.

## Previously exposed CMake 3.31 cross-corpus diagnosis

All eight CMake docs selected and opened in GRAPH015; already EXPOSED for this future study. 8,192 real next-byte labels/seed, five frozen seeds.

| Backbone | Best static exact+query NLL | Online mixture NLL | Online minus static |
|---|---:|---:|---:|
| Original TAC | **2.755120** | 2.755576 | +0.000457 |
| Delta TAC | **2.765930** | 2.766600 | +0.000670 |
| SSM | 2.716341 | **2.715438** | -0.000903 |

**Independent source-hash transcription erratum:** GRAPH015 originally measured the CMake packages manual with correct SHA256 \`3522aad1233e5263aa1d59b6ef5925939d3f3714f5550f66404f58ceca7ed502\`, as preserved in original raw locally frozen GRAPH015_TEST001_FIRST_OPEN_CMAKE_RESULTS.json. A later copied *GitHub receipt/replay-script* accidentally appended \`c9\` to that one SHA string; this is a malformed 66-character hex transcription, NOT changed publisher data or a new TEST. The GRAPH016 local rerun fails closed on the incorrect value and was repaired to the exact original frozen raw-results SHA. Do not silently rewrite original receipts; retain this erratum. The original CMake source previously became exposed.

## First-open authentic Python source: preregistration before first opening

Prospective eight exact Python 3.13 original stdlib sources were registered in Git commit \`17f55e6df1062ae4101bf130bd95f148f0947871\` after only reading filename+byte-size metadata. Files: asyncio/base_events.py, asyncio/streams.py, email/message.py, json/decoder.py, email/utils.py, urllib/parse.py, http/server.py, concurrent/futures/_base.py. None belongs to GRAPH011's eight previously scored Python files or GRAPH009 six. These files are NOW EXPOSED for further tuning. Public Python stdlib original source; no redistribution of text.

Frozen 8×129-byte windows/file, 128 authentic next-byte targets/window: 8,192 actual labels/seed and eight source clusters; five paired seeds, original frozen models, no optimizer steps. Primary preregistered **TAC** minimum material gain: mixture NLL at least 0.005 nats/byte below *best of native/exact/exact+query fixed whole-evaluation arm*, ≥4/5 seed directions and descriptive document+seed bootstrap 95% upper endpoint less than -0.005. The CMake/Vim posthoc behavior did not change this frozen rule.

| Backbone | Native NLL | Exact NLL | Exact+query NLL | Online NLL |
|---|---:|---:|---:|---:|
| Original TAC | 2.914884 | 2.594059 | **2.537989** | 2.543740 |
| Delta TAC | 2.913554 | 2.599813 | **2.544223** | 2.549817 |
| New SSM | 2.927659 | 2.595350 | **2.535868** | 2.541778 |

**PRIMARY RESULT: FAIL.** Original TAC online minus best fixed = **+0.005751** nats/byte (worse); **0/5** paired seeds improve; descriptive source-document+seed two-level bootstrap 95% CI **[+0.00430,+0.00698]**, without independent-publisher sampling. Delta +0.005594; SSM +0.005909. Original TAC native NLL is lower than SSM on these new Python sources, in contrast to Vim/CMake. Thus earlier SSM native rankings are task/corpus dependent, not evidence of a universal algebraic advantage.

Additional registered calibration/accuracy on the new Python source (original TAC): fixed exact+query Brier **0.748295**, top1 accuracy **39.4165%**; online Brier **0.751053** (worse), accuracy **39.3433%** (worse). Negative result cannot be reclassified as gain.

Why: the frozen approximate-query expert was clearly best on the newly opened 128-target windows; equal-prior mixture pays a finite within-window inference cost. The exact ln(3) per-window bound held in all numeric records; worst observed regret was within tolerance. Choosing a different starting prior now based on this exposed source would be evaluation-driven tuning and requires a **new** separately preregistered future source.

## Honest status and continuation

- ENGINEERING_MATH: 7 local constructed tests PASS (posterior telescope, regret, causality, independent owners, malformed probability refusal, equal experts, sequential learning); GitHub source-only CI runs those.
- REAL DATA: original Vim DEV and CMake EXPOSED; Python **first-open then EXPOSED**, registered primary gain **FAIL**.
- NATIVE_TAC_NEW_WEIGHTS: NO; the mixer trains **zero** new neural learned parameters, and each backbone stays frozen. It is an adaptive inference policy, not an improvement in the native memory architecture itself.
- REPO ORIGINAL CHECKPOINT COLD REPLAY: BLOCKED (five per-family Torch trained weights not Git/LFS hosted). Genuine publisher source not in Git. The reproducible standalone real-source replay code and raw numeric receipts are in the user evidence bundle, with all original SHA checks.
- FAIR TRAINING COMPUTE: NOT ESTABLISHED; GRAPH014–015 SSM original-vs-TAC full FLOP parity remains unmatched. Our three forecast arms share each frozen backbone; inference CPU/memory overhead must be accounted separately before deployment.
- INDEPENDENT EXTERNAL TEST ATTESTATION/REPRESENTATIVE BUSINESS VALUE/GENERAL INTELLIGENCE: NOT ESTABLISHED. This does not prove long-history factual retrieval, reasoned action or any Woyengi paid workflow.

**Next mathematical engineering candidate (NOT trained/tested here):** initialize source-local expert priors from lawfully eligible FIT-source performance and/or add a bounded-switching Bayesian model for within-owner distribution shifts, with a source-conditional prior-dependent regret theorem. Pre-register and freeze any FIT-derived prior and HMM switching cost before opening an entirely new authorized real corpus. Separately strengthen native TAC's useful source-dependent state representation and forecast objectives against equal-compute models rather than assuming a stable delta transition suffices.
