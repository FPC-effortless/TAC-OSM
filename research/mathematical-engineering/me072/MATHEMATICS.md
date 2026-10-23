# Native observed state and coherent future readout

Let `c=[f;s;M*q;e]` concatenate observed native fast/slow state, associative
recall and encoded actual event. The core still updates only from actual
observations. Querying forecasts leaves that state unchanged.

For learned horizon embeddings `u_h`, use

    z0 = tanh(W*c)
    z_h = z0 + sigmoid(G[z0;u_h]) * tanh(F[z0;u_h])

Each latent coordinate lies in [-2,2]. This is a bounded future readout, not
a simulated observation, repeated autonomous action or System2 proof.

## Source-record existence

At horizons h_j=1,2,4,8, `S_j=prod_(i<=j) sigmoid(a_i)` is the probability
that the original record has an h_j-th future observation. Conditional mark
probabilities normalize learned logit residuals plus FIT-only history/future
priors. The joint categorical distribution is

    P(activity=a)=S_j*p_j(a); P(EOR)=1-S_j.

Its mass sums to one and `S_(j+1)<=S_j`. EOR has no timestamp. Source-record
absence is not identifiable physical/process termination; its label cannot
be treated as available at the last-event time or a causal stop action.

## One discrete-continuous clock

Given that a future record exists, zero-time probability and positive mixture
parameters are

    p0_j = sigmoid(b0) * exp(-sum_(i=2..j) softplus(d_i))
    mu_jk = mu_1k + sum_(i=2..j) softplus(v_ik)
    sigma_k = .1 + 1.9*sigmoid(r_k), pi=softmax(w).

Weights and scales are shared across horizons; locations can increase and
zero mass can decrease. The conditional measure is

    p0_j*delta_0 + (1-p0_j)*sum_k pi_k*LogNormal(mu_jk,sigma_k).

Together with existence and marks this defines a normalized joint law. The
conditional independence of mark/time given latent state is a modeling
restriction; it does not establish true business-process or causal semantics.
Two mixture components allow positive-delay modes without a disconnected
quantile head. Reported quantiles invert this exact same clock CDF.

For any positive t, increasing mu with fixed pi/sigma decreases each component
CDF. Decreasing p0 further decreases the total CDF. Thus
`F_(j+1)(t)<=F_j(t)`, and all conditional clock quantiles are nondecreasing
with horizon. This is marginal consistency; full valid multi-event business
paths, action effects and an autoregressive environment simulator are not
guaranteed. Conditional clocks may still be inaccurate or very broad.

## Training and evidence

Use actual future activity/timestamp labels when present, and a separately
labeled retrospective source-boundary proxy otherwise. Joint categorical
log loss and Brier train the event/existence distribution. Conditional zero
and positive-delay likelihood train the same clock used for CDF quantiles.
The target-only log-seconds Jacobian is retained in reported likelihood.
There are no invented future times after EOR.

All gradients run through the shared model, with native state detached after
32 observed events. This preserves forward values while limiting credit
assignment. Mathematical normalization/ordering and finite-gradient tests are
necessary engineering checks; only original real-data forecasts establish
developmental performance, and no cross-architecture winner is claimed.
