# ME092 — integrated visual/native forecasting with shared task gradients

Inference architecture is unchanged from ME091:104225active weights, shared
observed-image/numeric encoders, native time-aware fast/slow/associative memory,
parallel horizon readout, proper two-Laplace mixtures and common bounded location
translation. No new comparison/winner, causal policy or intelligence claim.

This is ONE registered whole-model training recipe after ME091's joint failure.
Initializers are ALL5fixed index donor pairs, not selected seeds:94907native
values from ME087;7792learned image values from ME091;1526common-location values
zeroed. Everything remains trainable. Source original bytes/negative scores stay
unchanged. Parents'4096root+4096visual-training updates are fully charged, plus
2048new updates per seed (10240pipeline updates). Retained native and visual
values each receive6144updates along their own lineage; zeroed location2048.

## Proper shared score, factual training-only partitions

At each actual observed prefix/horizon the score is

`CRPS/(FITunit*h) + 0.2*NLL_physical + |median-target|/(FITunit*h) + 0.2*motion_BCE`.

The distribution supplies its own median, not an independently inconsistent
point head. The old coordinate-component BCE is removed: a latent mixture
component is not an observed motion-class label. No extra quiet penalty is
added. The factual targets partition this SAME importance-weighted score into
quietH1, movingH1, and longer horizons. Every valid target appears exactly once;
the three scalar losses sum to the original unsplit score before projection.
The real-FIT preflight independently verifies this and its gradient against the
original ME087 proper-score function. These future labels never enter inference.

One shared forward graph supplies all3losses. A single batched vector-Jacobian
product with identity task sensitivities yields a3x104225gradient matrix. This
is equivalent to separate reverse sensitivities, not a claim of fewer gradient
FLOPs or automatic hardware speedup. CPU/backend vectorization/fallback controls
actual speed; actual training CPU is retained. No extra3forward passes.

Following Yu etal, **Gradient Surgery for Multi-Task Learning**, NeurIPS2020,
https://arxiv.org/abs/2001.06782, a conflicting gradient `g_i` is projected against
original `g_j` when their dot product is negative:

`g_i <- g_i - (g_i dot g_j)/(g_j dot g_j) * g_j`.

Other-task order rotates deterministically by update index, no hidden randomness
or tuning. Projected rows are summed, clipped to1, then one guarded AdamW step
(lr0.0001,wd0.0001). With3tasks successive projections can reintroduce earlier
conflicts; AdamW momentum/preconditioning and finite step size do NOT inherit a
raw-gradient descent guarantee. Neither this method nor its unit tests guarantee
forecast calibration, learned intelligence or the predictive gate.

## Unchanged qualification

Same licensed120FIT/40exposedREPORT episodes and exact Git-owned ME091observed
pixels. No future image/reference/capture time or episode identity is an input;
protected CAL/TEST/tissue-box remain closed. All5final originals before any
REPORT score. Existing joint quiet/moving/H4/H16/NLL/interval/task/seed/95%CI gates
stay fixed. Additional99.54545%paired episode interval records11declared recipes,
not independent confirmation of an adaptive DEVsearch. Strict original replay
at unchanged rtol1e-5/atol1e-6 remains compulsory. One recipe, no restarts or extra
fits under this ID after failure. Customer/operator value remains unmeasured.
