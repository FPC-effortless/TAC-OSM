# ESWAM004 — actual real model intelligence experiment and frozen evidence

**October 23, 2026. Final scientific result: MIXED_TRADED_OFF / combined physical gate FAIL.** Five original trained genuine-real-data native 101,706-parameter models; all original checkpoints, source, optimizer/RNG and raw scores Git-owned. Existing ESWAM003 and other checkpoints remain unmodified. Cross-run numerical replay receipts differ: original independent replay FAIL (NLL); separate independent no-fit complete-field audit PASS; therefore full consistent cross-CPU strict replay NOT ESTABLISHED. No architecture superiority, action policy, multi-entity long memory or deployment claim.

## Why this was tested

ESWAM003 had good real moving-state predictive gain but excessive false stationary corrections, weak movement ranking, poor confidence and next-command imitation worse than a trivial reference. ESWAM004 adds a new factual kinetic-input event correction (instantaneous absolute motion, 0.8 EMA of observed absolute motion, previous command change, known command tracking error) and a kinetic-gated measured velocity continuation to the same native fast/slow 90D recurrent + role-protected eight-address exchange memory. The extra learned model weights are 480 (101,226 ->101,706 =+0.474%). Training loss, sample order, actual future targets, real-source split and optimizer updates are the SAME as ESWAM003, rather than introducing extra budget or model-specific data.

Five paired seeds 76001..76005, 640 guarded real AdamW steps × batch12/seed, exact sampled original FIT episode/window order, identical original normalized state scales and 960 original real one-step query cases/seed. Actual pinned lawful BodenAI source hash 4d01753b2de68ef2eb863b94aeb790f6b2f98c6068aede3bf60e315f72a8b776, 120 prior real FIT training episodes, 40 previously exposed disjoint DEV episodes; 84 actual moving and 876 nonmoving next observations per seed. No synthesized trajectories, future motion input, protected source TEST or physical device action. Equal training presentations 7,680/seed, nominal parameter gap under 1%; **actual whole-compute/FLOP parity NOT measured**, so no fair full algorithmic architectural winner assertion.

## Original new five-seed actual physical outcomes

Lower MAE, Brier and command L1 are better. Motion AP higher is better. Each seed scores the SAME 960 original DEV cases; the 40 episodes, not individual frames or five seeds, define source episode-level dependence.

| Original seed | All physical MAE | True moving 84 MAE | True stationary 876 MAE | Movement AP | Brier | Logged next-command L1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 76001 | 0.066691 | 0.527411 | 0.022512 | 0.235232 | 0.613152 | 0.010031 |
| 76002 | 0.066358 | 0.529967 | 0.021902 | 0.178362 | 0.635390 | 0.010630 |
| 76003 | 0.065263 | 0.526816 | 0.021004 | 0.222168 | 0.596823 | 0.010239 |
| 76004 | 0.068029 | 0.529895 | 0.023740 | 0.258840 | 0.570807 | 0.010490 |
| 76005 | 0.066553 | 0.528923 | 0.022216 | 0.271454 | 0.569988 | 0.010211 |
| **Five-fit mean** | **0.066579** | **0.528602** | **0.022275** | **0.233212** | **0.597232** | **0.010320** |

Prior registered frozen ESWAM003 five-fit means: all MAE0.063229, moving0.558250, stationary0.015762, motion AP0.144958, Brier0.147101, next-command0.010471. New real observations show:
- Actual movement-rich MAE improved **5.31% versus ESWAM003**, all five seeds, and **28.88% versus persistence** 0.743241; it passes the registered moving 5% gate.
- Aggregate next-frame MAE **worsened 5.30% versus ESWAM003**, though 0.066579 is still 3.04% below persistence0.068668. All five original seeds fail the registered at-least-5%-below-persistence total-MAE limit0.065235.
- Near-stationary MAE **worsened 41.32% relative to ESWAM003**, approximately5.6× stationary persistence0.003983 and13.1× frozen ME077 stationary0.001701. All five fail the strict <=0.001871 original stationary-preservation gate. Combined physical capability gate **FAIL**.
- Movement ranking AP rose from .144958 to .233212 (+60.9% relative) but remains far below original ME078/ME079 per-fit AP .61717/.60775. Worse, actual Brier grew from .147101 to .597232, which is a severe event-probability calibration/scoring regression. The new gating machinery did not solve motion-regime uncertainty.
- Logged source next-command imitation L1 .010320, marginally better than ESWAM003 .010471 but **8.21% worse** than the source repeat-last-action baseline .009537. It is not a useful learned policy, let alone safe, causal or capable of choosing real interventions.

Historical separate ME077/ME078/ME079/ME080 original models have different objectives/budgets, and are context, not a parameter/compute-matched winning rank. ME080 improved certain horizon/coasting metrics but also fails its full stationary/movement preservation gates. None of these historical results establishes that the Pauli-inspired exclusion relation improves physical predictions or that TAC is the best neural family.

## Original real source, trained checkpoint preservation and replay limitations

Complete original five final trained model+optimizer+RNG, FIT sample-order identities, registered hashes, all 960 original factual target/prediction rows/seed, original per-seed metrics, original real source qualification and live budget receipts were committed in the original training job **before** any independent replay. Full 1,150 software tests PASS; prospective 101,706 active/deployed model and original scientific-preflight PASS. All five original models fitted successfully without synthetic data, with no original weights changed afterward.

- The first fresh independent GitHub job restored original real checkpoints and source, but **strict 1e-6 replay FAIL** on at least one normalized Laplace NLL case; its original exception remains published at the GitHub Actions log. No tolerance relaxation is authorized.
- A separate independent read-only cross-CPU audit restored the same five frozen original weights, checked all 960 same real source/episode/frame cases/seed, and found **0 violations** across forecast mean/scale, actual target, physical MAE, NLL, action L1, movement probability/Brier and factual movement, all at unchanged atol=rtol=1e-6. This is a passing *separate audit*, but it does not erase the earlier original-workflow failure. Original bytes and scores unchanged. Audit receipt: research/replay-bundles/ESWAM004/CROSS_CPU_REPLAY_AUDIT.json .
- Treat the discrepancy as evidence of cross-worker numerical reproducibility risk until repeated runs consistently meet the original strict criterion. Unchanged scoring equations and registration have not been retrospectively altered.

## Intelligence and business gate summary

- Native physical future-state prediction improvement on *real prior-exposed development* moving observations: YES.
- Combined stationary-and-moving physical prediction: **NO**.
- Correctly calibrated movement probability: **NO; Brier substantially worsened**.
- A useful learned observed-command predictor: **NO; loses to repeat-last**.
- Qualifying authentic multi-entity, 256-event persistent memory: **NOT EVALUATED IN ESWAM004**. The distinct ESWAM005 source audit has since found real long tracks, but no trained source-qualified retrieval/effect result exists yet.
- Genuine counterfactual action selection, verified robot task success, safe actuator authority, unseen source/independent site, commercial ROI: **NOT MEASURED**.
- Fair architecture superiority relative to original ESWAM003, GRU/Transformer: NOT ESTABLISHED without audited matched full compute and fitted learned-family controls.

**Decision:** Preserve negative and positive evidence, keep ESWAM003 as the preferred *within-family overall physical MAE* checkpoint of v3/v4, and do not promote ESWAM004 to production. Future research requires joint motion/stationary conditional distributions with source-causal evidence and more reliable calibration, plus true multiple-entity long-horizon forecasting rather than further micro-gate tuning on the exposed robot DEV cohort. Original v4 study must not be retrained or retuned.
