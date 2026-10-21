# ME032 mathematical engineering — centered conditional memory correction

Let h=phi(f,s) denote the unchanged split-RMS native32fast/32slow memory, C the
26normalized common features from the frozen R-only context encoder, and p0(C)
the R-fitted common prediction including prior32. Define

\[
\ell(h,C)=\operatorname{logit}p_0(C)+\delta(h,C),\qquad
\delta(h,C)=w^\top h+v^\top[\tanh(Bh+AC+a)-\tanh(AC+a)].
\]

The centering identity is **delta(0,C)=0 for all C and all finite parameters**.
The original common predictor is recovered exactly at zero state. Unlike an
unanchored joint head, there is no independent context-only correction at h=0.
There is no final intercept: it would cancel identically and be inactive.

Write z=Bh+AC+a and z0=AC+a. Then

\[
\nabla_h\delta=w+B^\top\operatorname{diag}(\operatorname{sech}^2z)v,
\qquad
\nabla_C\delta=A^\top[(\operatorname{sech}^2z-
\operatorname{sech}^2z_0)\odot v].
\]

The tests verify the first identity against double-precision autograd and verify
exact zero-state cancellation with nonzero learned coefficients. Context
sensitivity of the correction also vanishes at h=0. A constant NONZERO learned
state can still enable context-dependent corrections, so these identities do not
establish that varying state contains conditional information. The pooled control
and real-user validation remain necessary, and neither proves information theory.

## Bounds retained, guarantees not added

Each native state update remains the original coordinatewise convex combination
of a previous bounded state and a tanh candidate, initialized at zero; fast/slow
coordinates stay in[-1,1]. Split RMS uses epsilon.01. For each32block,
\(\|z/\sqrt{\|z\|^2/32+.01}\|_2\leq\sqrt{32}\), hence \(\|h\|_2\leq8\).
The pooled control uses the same RMS map and has this normalized-feature bound,
without the native pre-RMS coordinate bound.

For fixed finite readout weights, tanh being1-Lipschitz gives

\[
|\delta(h,C)|\leq(\|w\|_2+\|v\|_2\|B\|_2)\|h\|_2
\leq8(\|w\|_2+\|v\|_2\|B\|_2).
\]

This is a weight-dependent bound; weights are not spectrally constrained. It is
not a uniform training-time Lipschitz/gradient-retention guarantee for the whole
nonlinear recurrence. No associative native scan, SSM conversion or larger state
is introduced. Two tanh evaluations per correction cost extra arithmetic; no speed
advantage follows from the algebra alone.

## Conditional training and actual credit

For binary log loss and memory parameters theta with frozen p0/context,
\(\nabla_\theta L=E[(p-Y)\nabla_\theta\delta]\). At zero correction p=p0;
remaining prediction error supplies the initial head-learning signal. Direct
state/output coefficients start at zero, so the first step may have zero body
gradient. Subsequent nonzero body gradients are measured, not assumed.

The joint control starts with the same functions/weights but also updates the
common head; it can reduce error through that route. The conditional arm freezes
that route (requires_grad=False and excluded from AdamW). The unit tests and
recorded final weights verify actual gradient routing and absence of decay.
These different optimization constraints are the mechanism being tested; freezing
a baseline does not mathematically force the state to retain unique information.

The loss always scores original real binary outcomes. It does not create
synthetic residual labels, privileged beliefs, alternative actions or costs.
Memory gradients now reach an encoder trained on S, unlike ME031's frozen-state
probe. All students start memory from scratch, and all receive identical R-only
common information. Native and pooled recurrence bodies have different inductive
biases and compute costs; the experiment is not a fully matched architecture win.

## Capacity and evaluation limits

Readout coefficients:90x13+13hidden bias+13output weights+64direct=1260.
Native memory134976 and pooled135512 parameters; conditional active counts
136236/136772. Joint native additionally adapts1205common coefficients,137441.
Global maximum gap.8845%; nonembedding allocations differ appreciably and are
reported. Frozen context128000embedding coefficients are deployed in every arm,
not hidden from the storage ledger.

R/S/T roles are user-disjoint for every prediction model. This reuses known FIT
records in entity-separated crossvalidation, not a new independent or out-of-time
cohort. Future qualification requires its own frozen protocol. Finite model score
differences include approximation/optimization/sample effects, not only state
information. User-cluster intervals condition on fixed fitted models/seeds and
overlapping rotated training sets; they do not capture full refitting uncertainty.
