# ME069 categorical, elapsed-time native dynamics

Known event inputs are activity/lifecycle, resource, preceding time gap,
elapsed case age, UTC hour/week periodic terms and observed prefix progress.
All fields are available at the current recorded event. There is no case-ID
embedding, future event, future command or hindsight completion attribute.
Unknown categories produce deterministic zero input embeddings.

Encode the current event as u. Native fast f and slow s have64 coordinates
each. Learned nonnegative rates decay f,s and associative M according to the
observed elapsed time, capped at168 hours for this registered candidate.
Gated tanh/convex fast updates and residual slow updates preserve coordinate
bounds[-1,1]. The8x8 associative delta write uses a normalized key and bounded
value, then divides by max(1,Frobenius norm). Thus||M||_F<=1. These are
numerical bounds, not a theorem of useful semantic memory or contractive
gradient flow. The GRU receives the same elapsed-time information.

The decoder reads fast/slow state, an event-conditioned associative query and
the current encoded event. A shared SwiGLU head produces three categorical
regime weights and an action-free next-event residual distribution:

    P(next event) = pi_repeat * onehot(current event)
                 + pi_transition * P_FIT(next | current)
                 + pi_neural * softmax(log P_FIT + learned_residual)

This adapts the established persistence/rule/neural regime principle to
discrete business events. The empirical transition expert is a FIT-estimated
statistic, not an invented contractual rule. Router weights do not establish
calibrated confidence. Proper log loss trains the entire probability mixture.

Observed waiting time D uses a hurdle distribution. P(D>0)=sigmoid(eta), and
log(D)|D>0 is Normal(mu,sigma). Initial parameters use FIT current-event
statistics; a learned residual adjusts eta,mu,log(sigma). Exactly tied source
timestamps have zero delay. Positive-event likelihood and Bernoulli occurrence
scores are proper up to target-only constants. Median and90% quantiles invert
the mixture CDF. A long-wait probability integrates this same distribution
above a fixed FIT-only positive-gap quartile; this is not an SLA or failure
probability. No policy or counterfactual decision claim follows.

Full32-event backpropagation and dense factual supervision train both models
on the same windows. Forecast queries do not alter observed world state;
actual subsequent records can be assimilated as later observations. This is
observational persistent prediction, not independent outcome verification or
online weight learning. Configurable domain adapters remove the earlier14D
robot interface limitation without claiming that a specific latent width is
mathematically necessary for enterprise data.
