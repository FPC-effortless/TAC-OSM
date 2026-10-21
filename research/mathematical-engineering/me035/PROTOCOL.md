# ME035 — Architecture-first native residual forecasting

Registered2026-10-21 before fitting. User direction: build/train accuracy and
efficiency improvements; diagnosis supports fixing mechanisms and is not a
standalone prerequisite. This supersedes the master plan's proposed mandatory
new-task headroom study. ME034 KuaiRec stop remains intact: no KuaiRec access,
new remainder or retraining of that recipe.

## Real task and source

Official ETDataset ETTh1 hourly transformer measurements, Zhou et al., Informer,
AAAI2021; collected with Beijing Guowang Fuda. Source
https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv .
Actual2589657bytes SHA256
f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066,
MD5 8381763947c85f4be6ac456c508460d6. Publisher LICENSE is **CC BY-ND4.0**,
SHA25683cbb5c31415b1d0ee1d20d35dae3fda3a045b85fae9364cc57facc9cd6e0946.
Preserve unmodified source CSV/license/attribution for offline reproduction;
do not distribute a modified source dataset. Window normalization is internal
model computation. Source README hash and complete license retained.

New objective: forecast all7measured channels HUFL,HULL,MUFL,MULL,LUFL,LULL,OT
at1,6,24hours using96past observations. Earlier ME011 forecast OT quantiles
at24/168hours; its scores cannot be compared directly with this new objective.
This uses already-inspected early source periods, NOT independent first exposure.
Publisher timezone unspecified. Enforce consecutive hourly timestamps, original
schema and finite measurements; abort on missingness, no imputation/synthetic data.

Parse numeric rows[0,11640) only. Fit normalization mean/std solely[0,8760),
std floor.001. Train anchors95..8735 atstride3; development anchors8760..11615
atstride6; each target <= its split end−1. History may overlap earlier observed
training rows. All later ETTh1 rows and ETTh2 remain unopened by ME035, but they
were exposed in older research and are NOT newly pristine qualification data.
No future covariates/current target values enter inputs. No physical-action,
causal intervention, maintenance ROI or commercial claim.

## Build and train the upgrade, not another architecture-ranking study

All four TAC stages have **18924active weights**, identical direct/nonlinear
readout, same raw96x7observations and targets; no inert matching parameters.
Original native fast32/slow32 gated convex state updates remain the foundation.
All readout/context/memory coefficients train jointly; no frozen common head,
centering constraint or mandatory state-value diagnostic.

- native_joint: original gated native updates, uniform initial gates1/8,1/24;
  trainable direct prediction path + joint SwiGLU context/state readout.
- native_revin: same trainables; add causal per-window mean/scale normalization
  and inverse output scaling. Local sigma=sqrt(window_variance+.0025).
- native_residual: also change slow proposal to tanh(s+tanh(W[f_next−s;s]+b)),
  giving slow memory an explicit identity path and innovation coordinates;
  initialize slow W normalstd.01 and bias0. State remains coordinate-bounded.
- native_multiscale: also initialize fast rates across2..32events and slow rates
 32..256events via gate bias−log(tau−1), orthogonal fast transition gain.9.
 These are learned initial rates, not frozen decays or proven effective timescales.

State readout split RMS eps.01. Context42 channels: last, meanslast6/24,
last−mean6, global-normalized window mean and log local sigma, seven each.
Including level/volatility explicitly avoids removing useful conditioning with
RevIN. Head: direct42->21 plus joint106->96, split48/48, SiLU(a)*b ->21.
Predict residuals around the last observed value; reverse local scale when used.
Small normalstd.001 output/direct initialization preserves near-persistence
while permitting memory gradients immediately. Train the full prediction function.

These are progressive engineering stages, not proof each mechanism is necessary.
No synthetic primary fits. Unit checks may verify causal math/stability/gradients.
Separate practical references: last-observation persistence and full-history
ridge AR(96x7flattened+intercept ->21), alpha100, unpenalized intercept; train
all original training windows once. No parameter/exposure matching claim for refs.

## Locked training/development search budget

12fits:4stages*3seeds35001/2/3;750steps*batch64 each =48000histories and
1008000scalar future responses per fit. Same sampler seed=seed+50 and identical
sample indices across stages. AdamW memory.001/head.003,decay1e-4,clip1.
Loss mean squared error over identical21globally train-standardized targets.
Rotate stage execution order by seed. One CPU thread,150CPU-second ceiling per
fit, no extra search/retry. Save distinct250/500/750 checkpoints, Adam/RNG,
every step loss/CPU/gradient, original JIT exports; partial/failure receipts on
any failure. Preserve all attempts. No cost-incurring external training.

Freeze/digest all12fits and36checkpoint endpoints before first new development
metric. Evaluate each endpoint with MSE,MAE, per-horizon/per-channel MSE and
recorded costs. No changing losses, source, stages, steps or baselines after scores.
Select one development candidate by lowest mean MSE over three seeds among
4stages*3endpoints, ties lexicographic. Same3selection opportunities for every
stage. Primary fixed endpoint also reported for all four at750steps.

Development improvement target: selected candidate >=2% lower mean MSE versus
native_joint750, favorable in all3seeds, no horizon mean MSE >5% worse.
Always publish absolute errors and whether persistence/ridge remain stronger.
Selected scores are selection-biased development evidence, not an untouched-test
significance result or broad architectural superiority. No population CI fishing.
Equal parameters/exposure/selection do not establish equal actual compute; report
actual spend/operator subtotals/inference/state/tensor bytes. Full compute parity
unqualified if any work unaccounted. No architecture-superiority claim on FAIL.

If an upgrade works, retain its actual checkpoint as a research candidate and
make the architecture available through an optional Torch module; do not promote
or merge. If none improves, preserve failure and do not claim model improvement.
Further iterations require a new bounded registration; no silent extra trials.

## Reproduction

Verify official source hashes and provenance before training. Verify bounded
updates, identity slow path, finite active gradients, source target indices,
TorchScript parity and causal input-only normalization with numerical tests.
Replay all36original checkpoint predictions and exports, sample fresh-process
Torch-only exports, reconstruct selection from saved arrays, check optimizer
clocks/sampler RNG and corrupted-payload rejection. Preserve source/code/models
in a checksum-indexed archive; independently extract and replay offline. Run full
repository checks and hand off a draft PR with exact improvement/remaining limits.
