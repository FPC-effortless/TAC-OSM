# ME017 — Jacobian and retention engineering note (no outcome claims)

Companion to frozen diagnostic preregistration PROTOCOL.md and numerical gate in scripts/me017_frozen_real_credit.py. This is an algebraic research note, not a trained architectural result.

## Existing accepted ME015 recurrent system

For a real event encoding x and native fast state z, the bounded update is
    q_f = tanh(W_x x + U_f z_previous)
    g_f = sigmoid(G_f[x,z_previous])
    z = (1-g_f) z_previous + g_f q_f.

For the native slow state:
    p = tanh(W_s[z,s_previous])
    g = sigmoid(G_s[z,s_previous])
    s_new = (1-g) s_previous + g p.

Both are coordinatewise bounded when initialized in [-1,1]. This does not imply retention, calibration, stable gradient products, or correct prediction.

The original ME016 failed experiment replaced p by tanh(W_s[z,s_previous]+z). A direct linear term inside tanh can **saturate** when the learned W_s already produces high activation, limiting effective sensitivity. This is a mathematical risk, not a measured proof of why the previous experiment failed.

## Optional convex slow interpolation

Define
    p = tanh(W_s[z,s_previous])
    g = sigmoid(G_s[z,s_previous])
    q = (1-beta) p + beta z
    s_new = (1-g) s_previous + g q,
with fixed beta = 1/4.

If z, p and s_previous are coordinatewise in [-1,1] and g in [0,1], q and s_new remain in [-1,1]. If beta=0, s_new is exactly ME015's accepted recurrence. No trainable parameters are introduced. Unlike the inside-tanh modification, the direct beta*z term does not itself pass through the candidate tanh, but the gate remains sigmoid-state-dependent.

## Complete first-order Jacobians (no missing gate derivative)

Partition W_s = [W_sz, W_ss] and G_s = [G_sz,G_ss]. Let Dp = diag(1-p^2), Dg=diag(g*(1-g)), Ds=diag(q-s_previous).

Then:

    d s_new / d s_previous
       = diag(1-g)
         + diag(g)*((1-beta)*Dp*W_ss)
         + Ds*Dg*G_ss

    d s_new / d z
       = diag(g)*((1-beta)*Dp*W_sz + beta*I)
         + Ds*Dg*G_sz

This includes every pathway through learned proposal and learned gate. The statement that a direct beta*g derivative exists holds **with gate and proposal held fixed**. It does NOT imply a minimum singular value for the total Jacobian because other terms may cancel, and g itself may be near zero.

Using ||Dp||_2 <= 1, ||Dg||_2 <= 1/4, ||Ds||_2 <=2 for state components in [-1,1], conservative bounds are:

    ||d s_new / d s_previous||_2
       <= 1-g_min + (1-beta)*g_max*||W_ss||_2
          + (1/2)*||G_ss||_2

    ||d s_new / d z||_2
       <= g_max*((1-beta)*||W_sz||_2 + beta)
          + (1/2)*||G_sz||_2.

These upper bounds may exceed one, so bounded forward states do not establish gradient stability.

For constant scalar gate g ignoring candidate feedback, old-state carry decays as (1-g)^H. Any reported effective time constant derived from actual mean gates is ONLY a rough reference because real gates are state-dependent and full learned Jacobians also contribute.

## Why frozen gradient measurements now

The experiment computes per-lag ||partial current logit / partial prior input|| and ||partial logit / partial native state|| on actual early FIT histories, and measures the effect of setting a final state component to zero at readout. It can reveal a sensitive/insensitive history region and whether fast/slow information currently contributes to trained readout. It cannot prove memory sufficiency, causal action effects or safe policy outcomes. Its old FIT labels and artificial internal ablations are diagnostics, not fresh test evidence.

## Next decision point

If unchanged native slow state has weak contribution and early-lag sensitivity is small, follow up on independent genuine action histories with an explicitly registered training trial of beta=1/4 convex interpolation versus unchanged ME015, ME016 rejected identity and both strong GRUs. Keep trainable count exactly 135050, original real input/seed/batch access identical, and report parameter/data/update parity separately from measured FLOPs/time parity. Fair superiority requires both parity gates, not merely equal steps. Use a new, unopened cohort and frozen checkpoints for scientific testing.

If slow state already contributes meaningfully, adding a direct path without evidence is unjustified. Investigate which actual input class and horizon is neglected, or whether the causal prior dominates. The mathematically correct action may be retaining the existing recurrence and improving training optimization rather than altering it.
