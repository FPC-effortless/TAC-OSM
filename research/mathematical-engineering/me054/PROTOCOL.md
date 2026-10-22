# ME054 — train integrated physically anchored TAC after genuine feedback failure

Registered **2026-10-22 BEFORE any ME054 optimizer update**, following:
- actual ME052 full source-fit / TEST failure (TAC 30.552, GRU 2.613, last-state persistence .06695 MAE)
- actual ME053 frozen-world-model DEV diagnostic: TAC feedback state norms up to 1054 and increasing errors after 1024 original real robot frames

This is a substantive integrated learned-world capability experiment. It is NOT
another memory expert/gating ablation, fake reward environment, JEPA/SSM
replacement, parameter-padding exercise or runtime-optimization benchmark.
We do not claim residual next-state forecasting or convex state updates
are novel: these are well-known physical system-identification and dynamical
stability tools. TAC's scientific research contribution remains the integrated
persistent learned associative relevance/retention/verification representation,
if it can ever beat competent physical baselines under fair original data.

## Exact native math and established rival

Both systems keep ALL their original learnable weights. Native TAC still
has fast32, slow32, 8×8 associative state and native relevance recurrence;
GRU has its full trained observation/feedback GRUCells. Neither gains
privileged future observations, labels or rewards.

Physical next-state, conditioned only on actual observed original x_t and a_t:
`predicted_x_(t+1)=x_t + D_FIT * g_theta(z_t,a_t)`
where diagonal per-joint `D_FIT` derives from the genuine original training
episode's mean absolute consecutive joint-position change, divided by
FIT-only state scaling, minimum 0.0005 in normalized units. It is identical
for both architectures and freezes before first optimizer step. Both last
physical head layers start at ZERO for exact physical persistence as the
untrained initialization. Supervised factual physical error =
`MSE((predicted_next - real_measured_next) / D_FIT)`.
Human takeover next frame remains independently masked if already human
controlled at t; binary takeover BCE mixed at fixed .2 for both models.

Original native TAC's correction is now mathematically
`z_new=(1-alpha)z_old + alpha*tanh(proposal(z,a,innovation))`,
`0<=alpha<=.1` and `alpha=0` when observed prediction error=0.
The physical error is normalized by exact same FIT-only delta scale;
the scalar human-onset correction is independently masked and
innovation-dependent. Both state and associative value corrections
are convex combinations of bounded values, with original native
associative state projected to tanh range after observation. The learned
belief can no longer grow additively toward norm >1000 under physically
valid finite inputs. These are checkable numerical invariants, not
proof of learned physical dynamics or long-history intelligence.
GRU retains its competitive stable GRU feedback transition.
The error-centered prediction prior is IDENTICAL across arms.

## Original training budget equality (user-specified compulsory)

- Native TAC trainable parameters 37,985; GRU control 38,295; gap <1%;
  no fake/unconsumed weights.
- Original physically logged 14D robot states/actions and actual human
  control: BodenAI RW-RL HIL, immutable revision
  `9290f5a96dc91ea58256e81d9ce2dd9621890b51`, original
  `scoop_cat_litter_into_trash_can` task, FIT 325 original episodes,
  DEV 80, TEST 100 original source-ID-disjoint episode cohorts.
- ORIGINAL full FIT source SHA256 verified before model fits; fit-only
  scaling and fit-only physical delta scale computed and frozen.
- Five paired original independently initialized seeds
  `52001,52002,52003,52004,52005`.
- **600 optimizer updates** per model, **batch size 2**, 256 physically
  consecutive real history frames (burn-in) followed by 16 real BPTT
  gradient-bearing physical action/target pairs. **19,200 genuine original
  next-state supervised transition exposures per model**, 15 times
  ME052's 1,280. Exactly the same source window sampler/order/seed and
  rare ACTUAL takeover-onset source oversampling for each paired arm,
  before either gets first gradient update.
- Same AdamW lr .001, weight decay .0001, gradient clip norm1, outcome loss
  and tuning trials ZERO for both arms. Identical original DEV/TEST scoring
  positions, physical error units and postprediction real feedback timing.
- **CPU time is only efficiency metadata and does NOT have to match.**
  No invented full-operator FLOP equality. The scientific strength
  of a result is limited to mandatory matched parameters, source
  inputs, optimizer/data exposure and training budgets verified.
- All 10 original model and optimizer SHA receipts frozen before opening
  DEV or TEST physically observed frame-level labels; no re-fit on
  original heldout scores.

## Real success criteria, with source caveat

Primary research usability: on actual real physical robot heldout outcomes,
each trained model should at MINIMUM be competitive with raw last-state
physical persistence. Merely beating catastrophically unstable ME052
TAC is NOT a success. Compare native TAC to fully trained GRU on identical
original physical state targets across all five original seeds. Report
per-original seed TEST/DEV real-unit MAE, normalized RMSE, masked
operator-takeover support, learning-curve and checkpoint replay.
If either remains worse than persistence, preserve that result.
Report strict missing/corrupt publisher episode IDs; no replacement.

**Major limitation and external corroboration:** ME052 TEST 100 episodes
were already examined, including metrics on all seeds. ME054 was explicitly
designed AFTER examining that heldout failure. Therefore even if ME054
scores well on the SAME original TEST, this is development/provisional
research, NOT a newly independent prospective confirmation or commercial
qualification. A different authentic robot task or source with independent
operators/session split and verified task outcomes is required for robust
general-world-model action-intelligence claims.

No autonomous action selection or verified task success/reward
measurements are present in this corpus. Both models only learn
factual next physical states and actual human takeover events after
logged actions; counterfactual action benefit remains unidentified.
