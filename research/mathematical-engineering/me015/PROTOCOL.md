# ME-015 — Correct TAC's diagnosed optimization bottleneck before adding mechanisms

**Registered on 2026-10-20 before ME-015 fitting.** No reuse of previously exposed Aug14–Sep06 outcomes as independent test. All source, FIT and historical DEV split exactly ME014B. This is an **engineering optimization/falsification study**, NOT a new data-set component search. Native TAC remains a nonlinear two-timescale recurrent model, not SSM/GRU.

## What the original trained weights actually say

In ME014B's three authentic-trained native TAC checkpoints, slow gate trainable biases average -3.9658 / -3.9579 / -3.9648, versus original -4. Fast gate biases -2.4434 / -2.4477 / -2.4396, versus -2.5. The bias-only effective update fractions are around 1/54 slow and 1/12.5 fast. Under constant gate approximation, slow state's fraction refreshed over 32 updates is about 45%; fast state much faster. **This is NOT a measurement of actual data-conditioned gates**; actual gate inputs and recurrent Jacobians vary and can materially alter effective timescales. Slow/fast readout L2 norms: seed14001 0.232/0.454, seed14002 0.275/0.440, seed14003 0.228/0.406. This motivates studying underutilization and optimization, not asserting slow-memory erasure.

Strict ME014BF heldout real Brier: TAC0.196248, GRU0.194484, causal prior32 user rate0.195408; GRU faster and all model architecture-superiority gates FAIL. Using already-opened original future outcomes to retune would be invalid.

## Explicit mathematical engineering hypothesis H1: timescale mismatch

TAC bounded state recurrence remains exactly
\`z_t=(1-g_t^f)z_{t-1}+g_t^f tanh(W_x x_t + W_z z_{t-1})\`;
\`s_t=(1-g_t^s)s_{t-1}+g_t^s tanh(W_s[z_t,s_{t-1}])\`.

Keep ALL operator weights, dimensions, parameter counts and optimization unchanged, but initialize gate biases using observed history H=32:
\`b_f = logit(1/8) = -1.94591\`;
\`b_s = logit(1/24) = -3.13549\`
rather than \`-2.5\` and \`-4\`. These are preregistered a priori from 32-length causal histories, NOT tuned from any heldout result. At zero gate-input contribution, memory refresh over 32 steps for slow new bias is ~74.4% vs ~45%. Convex state bounds [-1,1] remain proved coordinatewise under finite inputs and sigmoid in (0,1). State gradient/Jacobian stability and correct long-delay recall DO NOT follow from box boundedness.

H1 falsification: on the same historic DEV, the bias-only TAC must beat unchanged TAC in all 3 paired seeds by Brier. If not, reject gate initialization as the dominant bottleneck. No new neural layers.

## Explicit hypothesis H2: better prediction under causally observable base rate

Observed real same-user prior-32 response rate already rivals native TAC. The recurrent model should learn **incremental action-conditioned information**, not relearn the prior. For the H=32 strictly historical observed full-watch outcomes,
\`p_0=(1 + sum_{j=t-32}^{t-1} y_j)/(H+2)\`;
\`l_0=log(p_0/(1-p_0))\`;
\`p_theta = sigmoid(l_0 + f_theta(z_{t-1},s_{t-1},video_t,duration_t))\`.
This is a mathematical prior log-odds offset; each logit residual is initialized exactly zero as in original fitted heads, so initial prediction is causal p0, not 0.5. The p0 is not a future target and is computed from the same 32 historical labels available to EVERY arm. The added deterministic offset has **zero trainable parameters**.

H2 control: apply the **identical causal base-rate logit** to the exact-parameter GRU as well. It is invalid to claim TAC superiority merely because GRU was denied the same useful historical statistic.

## Registered diagnostic arm matrix and unexposed-data rule

**Five arms**: (A) unchanged exact ME014B native TAC; (B) native TAC with changed gate biases ONLY; (C) unchanged TAC + historical base-logit ONLY; (D) TAC with BOTH matched timescale biases and historical base-logit; (E) exact-parameter GRU + identical base-logit, including its same original active 68-parameter correction.

All five models: EXACTLY **135,050 active trainable parameters**, **same authentic FIT labels** from KuaiRec.big_matrix, same 32 causally ordered history, current logged action, train-only users, same random seeds 14001/14002/14003, same **300** AdamW optimizer updates, batch128, identical per-step sampling IDs, same BCE-with-logits objective and no train/test mixing. Source publisher ZIP MD5 \`261550d472c48eff4990fb13c0e5bcf7\`; SHA256 \`c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c\`. Historical development interval Aug01–Aug14 was **previously exposed**, so use only for diagnosed engineering direction, NOT for independent scientific success claims. No future Aug14–Sep06 reread, retune, or first-exposure re-evaluation allowed; old original results remain archived separately.

Record per-seed genuine DEV Brier/logloss, train-time CPU seconds, parameter counts, FIT loss trace, historical outcome prevalence, history blanking delta, and complete checkpoint SHA256. **Compute budgets differ if measured time differs**, despite matched step counts; include the actual training times and do not claim fair FLOP/time control.

Go/no-go on historical DEV: (1) D must beat A and C in all 3 seeds; (2) D must beat E in all 3 paired seeds and be better than prior32 statistical baseline; (3) verify no history/label leakage and no high variance in full-watch calibration. If either fails, current hypothesis is unsupported; do not add further features until examining diagnostic loss dynamics. If D wins, still NO real-world superiority until **a NEW, untouched, preregistered genuine logged-action evaluation** with matched compute and strong baseline, not ME014BF now-opened future test. Candidate data must be an independent authentic source with corresponding source checksum, temporal action/outcome order and favorable/nonfavorable outcomes.

## Diagnostic distinction and next math

A direct way to test whether the readout ignores slow memory: real data intervention \`s=0\` versus \`z=0\` at fixed model/current action with no retraining; this measures model dependence, **not true causality**. Report actual gate statistics on genuine histories and Jacobian gradient throughput; avoid calling an initial-bias approximation observed gate behavior. Longer history H128/H512 must preserve real chronological user records rather than synthetic memory worlds, with matched budgets and no claims from evaluation already seen. CDL-PHR Hebbian retention, entity-structured memories, executable operators and neural CASM should be integrated ONLY if one of these specific diagnostics demonstrates that simple recurrence engineering fails.

All earlier unified research ideas and experimental negative evidence remain archived. No auto-merge, no production/deployment claims, no SSM conversion.
