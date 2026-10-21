# ME043 implemented mathematical architecture

This is an engineering combination of conditional computation, contextual state initialization and native innovation memory. It is not a claim of priority over all related work. Model quality must be established empirically.

For a complete observed window X of shape96 by C, compute causal window mean mu and scale sigma, normalized local history Z and the across-channel shared history. Flatten each channel separately. Concatenate local/shared histories into w in R192. The explicit c in R16 contains twelve patch means, latest value, recent difference, original mean and log scale. All are observed at forecast time.

Cheap latent and router:

    l = Linear128(SiLU(Linear20(w)))
    q = sigmoid(Linear1([c;l]))
    d0 = a(c) * tanh(Head([c;l]))

Head is the shared SwiGLU/basis/output head, and a(c) is the inherited bounded horizon-decaying correction amplitude. The AR backbone processes all96 normalized observations; neither expert gets extra observations.

The native expert seeds f0=.1*tanh(l[:32]), s0=.1*tanh(l[32:64]) and M0=0. Pair twelve observed eight-step tokens into six tokens u_j=.25*p_(2j)+.75*p_(2j+1). At each token retain the original ME038 nonlinear fast/slow recurrence and address-normalized prediction-error write:

    k = normalized_key(u_j,f_j)
    v = tanh(value([f_j;s_(j-1)]))
    M_j = rho*M_(j-1) + beta*(v-rho*M_(j-1)*k)*k^T

rho remains in (.95,.9999) and beta in (0,1). Query-recalled associative state modulates the fast and slow steps exactly as in InnovationMemory.step. Seeding changes the architecture; it does not assert prediction equivalence with the old zero-initialized recurrence.

    h = RMS([f6;s6], epsilon=.01)
    l1 = [l[:64]; l[64:] + sigmoid(g)*h]
    d1 = a(c) * tanh(Head([c;l1]))
    d = d1 if q>.5 else d0
    Yhat = mu + sigma * (AR(Z)+d)

Dispatch occurs per channel-window. Only the selected subset enters the recurrent expert and second readout. Selected examples still pay for the cheap readout. Router/slicing/normalization and both readouts where used are charged in the full latency measurement. No token autoregression, fictional observation, temporal test leakage or external action is introduced.

Both exits train densely from genuine subsequent observations. Let e0/e1 be per-channel mean standardized squared errors and lambda the registered compute preference. The router's target is 1[e0-e1>lambda], detached from expert gradients; router expected-error and cross-entropy losses backpropagate through its context encoder. Auxiliary losses train both exits regardless of routing, avoiding native expert starvation. This is an observed predictive utility label, not identified causal action utility. Mean native fraction is measured, not constrained to a claimed speedup. Lambda is an error preference, not measured joules or CPU seconds. Actual CPU must be benchmarked separately.

The deployment path is a new native architecture. Dense training is intentionally more expensive than its conditional deployment, and any training-cost disadvantage remains part of the result. A route collapsing to always-cheap or always-native is a preserved empirical outcome, not evidence of successful adaptive allocation.
