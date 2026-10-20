# ME-014 — Mathematical contract, identifiability and evaluation limits

## What TAC must learn, not just what it contains

A logged interaction is `(u,t,a_t,y_t,d_t)`: user identity used exclusively by the harness to retain a *separate* causal history; `a_t` is a video that was actually shown by the original platform; `d_t` is its known duration; `y_t = 1[actual watch duration / offered video duration >= 1]` is observed after the impression. Let `H_{u,t-1}` be exactly 32 interactions timestamped before the present user impression. Then the only authorized native predictor is

```math
p_\theta(y_t=1\mid H_{u,t-1},a_t,d_t)
   =\sigma(w^\top [z_{u,t-1},s_{u,t-1},e_\theta(a_t),d_t]+b).
```

Native recurrence for each *previous observed event* `x=(e(a_j), y_j, d_j, timegap_j)`:

```math
q_j=\tanh(W_xx_j+W_z z_{j-1}),
\quad z_j=(1-g_j)z_{j-1}+g_jq_j,
\quad g_j=\sigma(W_g[x_j,z_{j-1}]).
```

Slow state `s_j=(1-g_j^s)s_{j-1}+g_j^s\tanh(U[s_{j-1},z_j])`.
These equations define a *native* learned nonlinear fast/slow world-history update; they are **not** a linear SSM or a symbolic planner. Given initial `z=s=0`, bounded tanh proposals, and sigmoid gates, every state coordinate stays in [-1,1] (convex-invariant recurrence, under finite well-defined inputs). This formal property says **nothing** about calibration, correct user preference or long-term retention. Per-user history segmentation is imposed by the audit harness using real user IDs; therefore this experiment **does not** establish spontaneous entity identification.

**Causal order:** history contains `y_j` only for `j<t`. The target `y_t` is passed to the loss but **not** to prediction. For new recommendations in a real deployment, a server would need authenticated feedback receipts before those outcomes could enter persistent belief; the ME006 PNDS/Lean boolean admission gate is a separate *verification interface*, not proof that KuaiRec logs are independently verified.

## Why this differs from the recent forecasting-only studies

The next offered video is a real logged **action** and the full-watch response a real observed **action outcome**; the model must condition its belief on both entity history and current action. But a single observed outcome for a shown action is **not** proof of what would have happened under an unshown video. No logged propensity model, randomized policy, observation-cost or counterfactual decision reward is available here. Therefore a smaller Brier score demonstrates better factual outcome prediction under the original exposure policy; it does not qualify optimized autonomous CASM recommendations or causal policy value.

## Proper scoring and falsifiable comparisons

Use actual Bernoulli log-loss `L=-y\log p-(1-y)\log(1-p)` for equal-budget neural fitting, and original genuine heldout Brier `E[(p-y)^2]` as primary paired metric. Brier is proper; an unconditional constant `p=\bar y_{FIT}`, train-only per-video posterior smoothed rate, and causal prior32 user mean establish simpler baselines. Both learned models use the same train windows, current video embeddings, optimizer batch ids, seed and steps. Full active parameter mismatch <=5%, exact CPU training cost documented and **not** assumed equal. Blank out the historical input as a posthoc diagnostic, then report Brier increase/decrease. The frozen heldout user test may use user feedback that has already occurred *earlier in that same test period* when building each succeeding prediction; this is a valid online *prequential* evaluation but **not** a fully isolated batch holdout. Those future responses are never used to update network weights.

These distinctions must be stated when publishing any positive or negative ME014 result, and all previous ME001–ME013 work remains separately preserved.
