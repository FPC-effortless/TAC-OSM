# ME073 mathematics: shared native imagined futures

Let H_t contain only recorded prefix observations. Initialize bounded native
fast/slow context from its first observed event; carry factual state S_t with
native elapsed-time fast/slow/associative updates, detaching gradients rather
than resetting at each32-event boundary. This is the inherited TAC recurrence,
not a linear SSM and not an associative parallel scan.

Create a separate imagined copy Z_0=S_t. At imagined step k, decode a conditional
record continuation s_k, mark distribution p_k(a), and activity-conditional
hurdle clock q_k(d|a). For every actual branch a,d:

    P(EOR at k | earlier branch, H_t) = 1-s_k
    P(record a,d at k | earlier branch, H_t) = s_k p_k(a) q_k(d|a)
    Z_k = TAC_observe(Z_{k-1}, imagination_encoder(a,log1p(d),1[d=0]), d)

The next prediction depends on the previous generated activity and time through
the shared native dynamics. EOR is absorbing. Summing branch continuation gives
nonincreasing unconditional survival across steps. An imagined step is never
accepted as an actual observation or written back into S_t.

Positive time is a three-component lognormal mixture:

    q_k(d|a) = p0_k delta_0(d) + (1-p0_k) sum_j pi_kj LN(d;mu_kj,sigma_kj)
    mu_kj = FIT_pair_log_quantile_j + 2 tanh(delta_mu_kj/2)
    sigma_kj = .15 + 1.1 sigmoid(FIT_scale_logit_j + delta_scale_kj)

Thus .15<=sigma<=1.25 and learned location corrections stay within2 log-seconds
of FIT pair anchors. No arbitrary wall-clock cap, positive-outcome truncation,
posthoc conformal width promise or Student-t-tail cure is assumed. Sum d_1..d_h
along the same path: each elapsed sample is nondecreasing. Mark/time dependence
is explicit through q(d|a); future source resources remain unavailable.

## Training versus generation

Teacher-forced likelihood evaluates exact autoregressive factors for an actual
suffix, masking all steps after its first absent record. It is a proper joint
path probability under this model, **not** independent forecast accuracy at each
teacher-fed step. Original-seconds clock likelihood includes the log-time
Jacobian. FIT optimization omits only that target-constant term. Early prefix
weights and the fixed combined training loss are recorded in the protocol.

Free-run forecasts never receive future labels. Generate M=256 quasi-Monte-Carlo
conditional mark/time paths with fixed scrambled Sobol points. Let
W_ih=prod_{k<=h}s_ik and T_ih=sum_{k<=h}d_ik. Then estimate

    P(record at h) = mean_i W_ih
    P(A_h=a | record at h) = sum_i W_ih 1[A_ih=a] / sum_i W_ih
    F_h(u | record at h) = sum_i W_ih 1[T_ih<=u] / sum_i W_ih

Marginalize continuation by weights instead of random hard EOS samples.
At h1 the categorical distribution is analytic; later categorical masses and
clock quantiles use finite deterministic QMC integration. Normalized epsilon1e-6
mass smoothing prevents undefined categorical log scores from finite-particle
empty categories. This is an explicit approximation, not exact marginalization.
Quantiles invert the weighted empirical CDF; nominal coverage is not guaranteed.
Factual suffix density NLL is computed exactly and reported separately.

Conditional-survival quantiles may reverse over h because high-latency branches
can have different continuation probabilities; monotonic path clocks remain
valid. Forcing monotone conditional quantiles in this setting would restrict
selection effects rather than prove path coherence. No transition-grammar
legality, causal intervention, unlimited memory or business completion follows.
