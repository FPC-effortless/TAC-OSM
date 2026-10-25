# ME089 — native global context and parallel local evidence

The completed previous models exhibit a real tradeoff: ME087 has stronger
moving forecasts but fails quiet precision; ME088 clears quiet precision but
loses moving forecasts. Neither is a full success. This new single registered
architecture keeps ALL5fixed ME087 originals as initializations and adds a
2233-parameter coordinate-shared evidence path. Total101964active/deployed
parameters, all trainable; no second inference model or best-seed selection.

## Parallel information-preserving branch

For each actual coordinate, construct32known values:8raw past positions relative
to the current value and normalized by FITdelta_scale;8observed ages;8current
local position/control/velocity/acceleration/change/innovation/elapsed features;
4known horizon values;4learned known-coordinate identity values. No future
state/command/capture timestamp, source task/episode identity or true future
movement label enters inference. These are alternate representations of already
available observations, not new information beyond the prefix.

A shared32→64→1SiLU MLP computes local evidence for every batch/time/horizon/
coordinate in vectorized tensor operations. It adds to the native global
mixture logit. Both branches remain trainable and the full distribution is
still the same coherent two-Laplace mixture. No argmin-variance hard switch or
outcome-aware selection is used. The stronger original full movement residual
is retained. FIT-noise-bounded quiet centers and predictive-noise units stay.

Known command error is explicitly normalized by typical FIT coordinate motion
before asinh conditioning, so small control errors are not hidden in a global
state representation. Shared per-coordinate processing and learned coordinate
identities are established techniques (e.g. DeepSets2017); no new memory,
Bayesian-independence or equivariance theorem is claimed.

State remains1331values/entity; local evidence reuses the existing raw/time cache
and current features. The native nonlinear time recurrence stays sequential.
Encoder/filter/horizon/coordinate computations expose parallel axes; no full
native scan or GPU-fusion performance is assumed.

## Training and gates

Five fixed-index ME087 warm starts, new optimizer/RNG,2048new updates,B24,T32,
lr0.0003. Existing base weights4096+2048=6144total; new routing weights2048only.
At most2independent one-thread seed workers. No hidden restart, LR sweep,
checkpoint/seed selection or inference ensemble. Source checkpoints and source
code are hashed before fitting; exactly five new routing tensor keys are
allowed missing on warm initialization. All final original keys load strictly.

Use prior CRPS+0.2NLL+0.2globalmotionBCE+1implicit-medianMAE, plus1coordinate
change-above3FITpredictive-unit*sqrt(h)BCE and2one-step factual-quiet median error.
These future outcomes are FIT labels only. Auxiliary labels are observed-change
proxies, not independently known latent physical modes or causal interventions.
Multiobjective training does not guarantee unweighted posterior calibration;
unchanged proper-score/coverage guards must be measured.

Every original ME081 and stronger ME082 gate remains unchanged. Additional
99.375%paired CI describes these8declared attempts only; repeated adaptiveDEV
is not independent qualification. All5originals/optimizer/RNG/receipts are
Git-frozen before REPORT; float64whole forward from original learned values,
original rtol1e-5/atol1e-6. Protected tissue-box/CAL/TESTstay CLOSED. Single-model
capability work only, no matched-architecture winner, deployment or product claim.
