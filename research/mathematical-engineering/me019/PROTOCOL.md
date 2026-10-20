# ME019 — Longer genuine memory, not another simulated success claim

**Stage:** original real training/previously inspected development only. **Source:** original publisher KuaiRec big_matrix authentic chronological video impressions and observed watch responses. Preserves ME001–ME018 positive and negative evidence and native TAC identity (no SSM conversion).

## Falsifiable question

ME017 verified trained 32-event slow state contributes to observed prediction and does not show the hypothesized 32-step oldest-event sensitivity collapse, but **nothing established true use of past event number 33 through 128**. ME018 achieved exactly equivalent native recurrence CPU execution, enabling a cheaper controlled trial. Determine whether a longer genuine event history improves action-response forecasts under the same measured real targets and equal active model capacity, rather than adding another memory component.

## Training and data registration BEFORE running

- Original authentic ZIP SHA256 c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c; MD5 261550d472c48eff4990fb13c0e5bcf7.
- Use original training user_id % 16 == 0 only. Training target timestamp **strictly before Aug 1 2020**. Previously inspected development target timestamps Aug 1–Aug 14 2020, which MUST NOT be described as held-out independent scientific validation.
- **Never parse** reactions dated Aug 14 onward or other users 1, 2, 3, 4. An untouched new user4 cohort may be reserved for a later separately preregistered once-frozen test after the training checkpoints have been committed and SHA256 sealed. Users 1/2/3 have already been exposed, so they cannot serve as fresh.
- Select only windows with at least **128 real, chronological, same-user prior observations**; any experiment with fewer real windows fails rather than padding or synthesizing missing history.
- The exact *same* authentic target rows, watch outcomes, current offered video, duration, train-example sampler indices and seeds are used for 32 and 128 context as well as native/GRU. Build original full 128 context first, and use EXACT suffix 32 for the short variant. The 128 metadata gap at position 96 is deliberately zero, preventing the short variant's first gap from leaking outside its accessible 32 events.
- Source-only deterministic original eligible-fit striding to at most 12000 genuine training target anchors and at most 6000 genuinely measured but **previously opened DEV** anchor rows; no target-conditioned selection.
- The explicit causal user-response prior logit always uses **last 32 genuine responses** and never the current response even if the neural context has 128 events. This isolates historical recurrent information from changing analytic priors.

## Fixed preregistered 18 fits

- **Real history:** 32 vs 128 records, each record carries prior action, prior actual response, actual duration and elapsed time; no invented actions, users or outcomes.
- **Arms:** existing accepted native chrono+causal TAC; strongest original unanchored GRU42 with active action refine; equally anchored GRU42. No new learned module. Native uses TorchScript-compiled identical equations from ME018. Three seeds 14001, 14002, 14003.
- **Training:** all exact **135050 active trainable parameters**, three seeds, 300 AdamW updates per arm, batch 128, exactly same random source target row indices in each seed, lr .001, weight decay 1e-4, BCE, norm clipping 1.0, single-thread CPU. No early stopping, cherry-picked seeds, extra tuning, or history-specific optimizer choices.
- Check 32-event arithmetic against the original ME016 EngineeredAgent forward in numerical preflight. Check TorchScript 32 and 128 native forward parity. Record parameters, genuine record counts, actual train CPU wall time, scores, all original checkpoint SHA256s and negative findings.
- **Primary development diagnostic**: genuine Aug1–14 real past 128-eligible same target windows; Brier and binary logloss, same anchored prior32 rate. Only engineering evidence because previous studies already inspected this DEV. No model promotion.

## Compulsory fair comparison constraint — fail closed

All arms match **trainable parameters** EXACTLY (mismatch=0), real example target rows, batches, seeds, optimizer updates and target exposure. The 128 context deliberately exposes more *historical conditioning information* than the 32-context variant; that is the isolated treatment. **Training CPU time/FLOPs cannot be assumed equal even when all optimizer steps are matched.** Per-arm measured times MUST be archived.

Original strictly requested head-to-head architectural fairness additionally requires equal training compute/time budgets. **ME019 DOES NOT SATISFY IT unless independent metering demonstrates qualifying equality**, so all TAC/GRU performance differences are engineering diagnostics, NOT fully matched architectural victory. Separate future compute-budget-matched and data-exposure-matched replications must be conducted and mutually interpreted. Never waste computation artificially to manufacture apparent parity; report unavoidable mismatch and withhold comparative promotion.

### Hypotheses to test (fixed now, not selected after seeing data)

- H1: for same original true real target, native128 development Brier less than native32 in all 3 seeds, measured as Brier(128)-Brier(32) < 0. This is *reused-DEV engineering*, not real scientific success evidence.
- H2: native gains from 128 exceed strongest original GRU gains in all 3 paired seeds at identical true targets; still not a fully fair compute-controlled comparison.
- H3: the gain, if present, exceeds the existing causal prior32 full-watch rate. Score absolute Brier too, not only differential gain.
- H4: native128 fits remain numerically stable at all preregistered updates.
- If H1 fails, do not blindly increase to 512 or add another recurrent connection; inspect saturation, state evolution and actual decision-relevant timescales.

## Next independent test boundary

Once all 18 weights from **this original run** are sealed with checksums and execution parameters fixed, a separate frozen follow-up can evaluate genuine **uid%16==4**, target dates Aug14–Sep6 2020, subject to enough real same-user 128-event histories. No weight updates, threshold selection, or source retuning on that cohort. Report the first source opening with checkpoint-before-source integrity; if fewer than 128 past real events exist, do NOT invent/pad data and acknowledge failure of this task construction. Do not open an independent set before freeze.

## Scope limits

KuaiRec actions are logged exposure impressions and real watch feedback, but not randomized policy interventions and not independently signed verification receipts. Counterfactual CASM action value, policy safety, 512+ retention and commercial benefit are **not established** here. Keep the native TAC core, learned representation and broader full mathematical-engineering master plan intact; this experiment diagnoses a specific longer-memory failure, not whole-system completeness.
