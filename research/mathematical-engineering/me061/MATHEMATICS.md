# ME061 predictive state and null-safe residual dynamics

Retain ME060's learned bounded fast/slow recurrence, 8x8 associative delta
memory, norm projection and non-mutating action/horizon reads. The measured
state correction now has an explicit shared reliability observer:

    v_t = .75 v_previous + .25 tanh((x_t-x_previous)/delta_scale)
    e_t = .9 e_previous + .1 min(20, |x_t-x_previous-servo_previous|/delta_scale)

Both use newly received physical observations only. They are deterministic
observed statistics given the history, not extra information and not a claim
of learned semantic memory. v stays in [-1,1], e in [0,20]. Both arms receive
the same statistics and preserve them across each real history.

Let b_h denote the source-FIT servo integration for the supplied action prefix.
The horizon decoder sees current x, b_h-x, time, v, e and learned recurrent
features. TAC also reads action-conditioned associative memory. It emits
three bounded residual slots r_k, four softmax coefficients pi, signed physical
gain u and positive scale parameters q:

    gain_h = tanh(u_h) / (1 + e_t)
    mean_h = x_t + gain_h * (b_h-x_t)
                 + h*delta_scale * sum_(k=1..3) pi_(h,k)*tanh(r_(h,k))
    scale_h = (softplus(q_h)+.001)*max(h*delta_scale,.0001)

The zero-initialized final map gives exactly mean_h=x_t for every horizon and
action before any optimizer step. Unlike multiplying two zero gates, gain has
a nonzero derivative at u=0, and the final residual weights can immediately
learn. Earlier encoder/recurrent weights become gradient-active after the
readout's first optimizer step. State and learned residual bounds follow from
convex updates and tanh; these do not prove stability of all Jacobians or
long-history credit assignment.

Residual slots may be redundant, especially under symmetric initialization;
specialization is not claimed. The capability question concerns the full
forecast and interval quality. Query computation does not mutate observed
world state and earlier horizons cannot access later action prefixes.

Loss scores physical error p=abs(mean-target)*state_scale:

    L = mean_masked(p) + .01*FIT_physical_step_mean*
         mean_masked(p/physical_scale + log(2*physical_scale))

Use the same 16-horizon and weighted dense one-step losses for TAC and GRU.
The primary term now has the same physical-coordinate weighting as the
reported MAE. Scaling the probabilistic term by a FIT-only common physical
step magnitude prevents normalized coordinate scales from redefining the
primary optimization target. This is established residual/observer/proper-
scoring engineering; no new control theorem or general intelligence is claimed.
