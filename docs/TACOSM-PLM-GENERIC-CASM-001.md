# TACOSM-PLM-GENERIC-CASM-001

Mechanism experiment: generic learned compositional computation.

The benchmark samples episode-local continuous operators from three registered
families. Each episode provides eight unordered demonstrations containing only
(u, v, y), followed by four new query pairs. No operator ID, family label,
operator parameters, or query targets are passed to the model.

The model has two separable stages:
1. infer a permutation-invariant operator code from demonstrations;
2. execute the code through a learned basis of generic bilinear interactions
   plus learned linear residual terms.

Training and evaluation parameter streams use separate RNG namespaces and
episode-local parameterizations. Parameter fingerprints must have zero
train/evaluation overlap.

Primary criterion: aggregate held-out query MSE <= 0.02 and no seed above 0.05.
The oracle is the exact benchmark operator and is only a representability
control.

Controls include support-row permutation, shuffled support outputs, and
zero-support evaluation. No post-run model or seed selection is permitted.

A positive result is bounded evidence for generic learned compositional
computation on these registered continuous synthetic families. It does not
establish open-ended operator synthesis, multimodal capability, persistence,
or scaling.
