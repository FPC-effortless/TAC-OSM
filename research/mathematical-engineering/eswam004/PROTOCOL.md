# ESWAM004 — source-causal kinetic activation, first original real-fit registration

**Frozen BEFORE first original optimizer update; October 23, 2026.** One candidate, no tuning search, no hidden further fit, no synthetic evidence of real capability. Original ESWAM003 source/optimizer/score/checkpoint commits remain immutable. Base branch ESWAM003 evidence is previously exposed real DEV, so another gain does NOT constitute independent-source generalization.

## Source and actionable hypothesis

ESWAM003 improved moving-state forecasting but falsely projected physical motion in stationary cases and had weak future-moving discrimination (AP .145). New ESWAM004 preserves the native fast/slow recurrence and Pauli-inspired deterministic distinct evidence-role memory. A small 465-parameter 56→8→1 kinetic event residual uses ONLY observed absolute per-joint velocity, its exponential running mean, absolute last recorded command tracking error and the change in previously recorded command. It augments the historical movement logit, and a causal opening factor combines that movement estimate with physically observed kinetic change. A learnable per-joint velocity continuation contributes only under the same constrained gate. These new components are trainable with actual source next-state/movement supervision.

The final correction is physically interpretable:
  e_t = mean_j( |v_tj| + EMA(|v|)_tj + .25 |a_previous,t - a_previous,t-1|_j );
  k_t = 1 - exp( - softplus(g) e_t );
  p_t = sigmoid(original_ESWAM003_logit + small_kinetic_MLP(input_t));
  G_t = 1 - (1 - p_t^2)(1 - k_t);
  delta_t = ESWAM003_predicted_delta + sigmoid(c_j) * v_tj;
  predicted_next_t = x_t + G_t * delta_t .

A future movement label is NEVER fed into any input; all present/previous velocity and command quantities are real factual source observations at decision time. This mathematical gate may improve quiet-state preservation but may suppress unanticipated movement onset. That is an explicit falsifiable risk. Neural output still depends on true factual prior action history, NOT a do(a) intervention.

## Identical matched developmental recipe

- Same actual real 14D BodenAI/RW-RL-HIL physically measured robot state source, same pinned publisher 160-episode digest 4d01753b2de68ef2eb863b94aeb790f6b2f98c6068aede3bf60e315f72a8b776, CC-BY-4.0 publisher/parent attribution.
- Exactly same 120 original authentic FIT training episodes, 40 formerly exposed disjoint DEV episodes, exact same deterministic sample order, same 960 truly observed H1 predictions/seed, 84 future moving and 876 observed no-change, no protected source CAL/TEST or new source.
- Same five seeds 76001–76005, original 4×160 AdamW updates × batch12, same 8 observed historical frames, same previous recorded qpos, same 0.0007 learning rate, weight_decay0.0001, gradient norm clip1.0. Both original ESWAM003 and new ESWAM004 get exactly 640 updates and 7680 factual source-window presentations per seed. Parameter count old101226/new101706 (+0.474%); hardware FLOP and full operation-parity not yet measured, so no confident matched-compute architecture superiority claim.
- SAME training objective and source targets: normalized real next-state Laplace NLL + 0.5 physical-normalized absolute forecast error + 0.20 original FIT-prevalence weighted real movement BCE + 0.05 SmoothL1 next recorded qpos action imitation. No added force labels, oracle regime labels or artificial future targets.
- Single final checkpoint/seed, no original DEV-driven hyperparameter sweep, no added training source. All source hash, model+optimizer+Torch/NumPy/Python RNG, original real case predictions and original case target arrays saved to Git BEFORE replay. All original scores and checkpoints kept unchanged.
- Full original live learned 101706-weight training guard, checked optimizer steps and compulsory source/license/split qualification, exact source sample order+hash, torch.compile-free reproducible floating arithmetic. CPU resource time measured only, not forced to match.

## Prespecified evidence gates and controls

Compulsory real-world prediction gate unchanged: <=0.95*0.068668=0.0652346 all-case MAE and <=0.95*0.743241=0.706079 moving MAE; stationary <=1.10*0.001701=0.001871. Report every seed, no seed selection. Movement risk AP/Brier separately, next-command imitation must beat repeat-last historical command L1=0.009537 on same actual source episode/time. Original frozen ESWAM003 real mean all0.063229, moving0.558250, stationary0.015762, AP0.144958, command L1 .010471; original ME077 stationary .001701 and ME078/ME079 movement still competing; no retrospective ranking by mismatched multi-horizon tasks.

Request an independent GitHub checkout zero-fit original-score replay of every original saved trained model on identical original source case IDs, requiring original registered rtol1e-6/atol1e-6 for normalized NLL, physical MAE, forecast means/scales, movement probability/Brier and command. A strict replay discrepancy is a FAIL, not an adjustment of acceptance tolerance. Numerical drift must be preserved and separately quantified as an audit, not dismissed.

**Capability boundaries:** eight-frame single-episode robot history cannot establish long-horizon multi-entity memory, content-keyed recall, CED CASM counterfactuals, real robot action authority, safe task completion or customer economic value. Moving-case classification uses a factual future observed label for training only, never as model input. Strong physical state prediction on previously exposed DEV is not an independent reliable control capability.
