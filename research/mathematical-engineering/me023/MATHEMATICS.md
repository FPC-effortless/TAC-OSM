# ME023 — query-conditioned predictive native TAC

The objective is mathematical engineering of useful learned state. Adding
complexity or calling a model an SSM/LLM does not establish that objective.
This first step combines a bounded context-dependent readout with causal
multi-depth prediction, while retaining native TAC's nonlinear recurrence.

## 1. A precise representational limitation

The current logit is `ell(h,a,d)=w_h h+w_a e(a)+w_d d+b+logit(prior32)`.
At fixed prior and duration, history-dependent log-odds differences do not
depend on the offered action. The mixed derivative with respect to continuous
state and action embedding is zero. Equivalently the four-point logit contrast
`ell(h1,a1)-ell(h1,a2)-ell(h2,a1)+ell(h2,a2)` is zero.
Shared embedding parameters do not remove this functional separation at fixed
states. Probabilities can have nonzero mixed derivatives after sigmoid; the
limitation concerns additive **log-odds**, not every probability interaction.

## 2. A bounded action-conditioned state readout

Write `h=[z;s] in R^64`, and define:

```
v = h / sqrt(64)
q_a = tanh(e(a)) / sqrt(8)
B(U) = beta * tanh(U) / sqrt(64*8),  U in R^(64x8), beta=4
r(h,a) = v^T B(U) q_a
ell_new = ell_old + r(h,a)
```

This is a bilinear readout with effective interaction rank at most8, not a
Transformer attention layer or new SSM state. The primary equal-parameter
control uses `q_h=tanh(h[:8])/sqrt(8)` in the same equation. It can learn a
nonlinear state correction but its correction cannot depend on current action.
Both add512 **active** parameters, for 135,562 total; no padding or frozen dummy
capacity. Zero-initialized U preserves the original initial logits exactly.

Inductively the native states remain in[-1,1]: each update is a coordinatewise
convex combination of the previous bounded state and a tanh proposal, starting
at zero. Thus `||v||2 <= 1`, `||q||2 <= 1`, `||B||2 <= ||B||F <= beta`, and
**`|r| <= beta` for every parameter value and valid bounded state**. This is an
interaction-logit bound; the original unconstrained linear head is not bounded
by beta. Bounded states are not proof of contractive memory or generalization.

For the action query, holding action/state inputs independent:

```
grad_h r = B q_a / 8
norm(grad_h r) <= beta / 8 = 0.5
mixed_Hessian_(h,e) r = B diag(sech(e)^2) / (8 sqrt(8))
norm(mixed_Hessian) <= beta / (8 sqrt(8))
```

The self-query gradient also includes the derivative of q_h, with norm bound
`beta/8 + beta/sqrt(8)`. These are bounds for the added readout term, not the
entire network Jacobian. Unit tests compare analytical derivatives with
autograd and construct a nonzero four-point interaction. Neither is a generated
capability benchmark or a learned real-data performance claim.

## 3. Causal prediction supplies temporal supervision

For a genuine128-event parent window, snapshots at depths 32,64,96 predict the
**next logged event**, before consuming its feedback. At depth k, state uses
events0 through k-1, query sees offered item[k] and duration[k], target is the
actual observed response[k]. The prior uses feedback[k-32:k]. The final depth 128
predicts the parent's current target with its original prior32.

```
L_prefix = 0.5 BCE(ell_128, y_current)
         + (BCE(ell_32,y_32)+BCE(ell_64,y_64)+BCE(ell_96,y_96))/6
L_terminal = BCE(ell_128,y_current)
```

Weights sum to 1; the objective deliberately changes credit allocation. Prefix
arms supervise4 real responses per sampled window versus 1 for terminal arms.
They receive the same recorded observations, not extra sources or generated
labels. This difference prevents treating cross-objective contrasts as equal
supervision/compute claims. Primary comparison holds prefix supervision fixed.
Perturbing feedback at/after k must not change the k forecast; tests enforce it.

## 4. Computational and scientific boundaries

Native recurrence costs O(T) sequential steps and retains64 floating state
coordinates per example. It is nonlinear and state-gated: no associative scan
or parallel SSM execution is claimed. The extra final readout uses520 affine
MACs per example plus bounded nonlinearities. Prefix training adds three
readouts, not attention over every history pair. Standard training autograd
still stores O(T) activations; constant inference state is not constant training
memory. Parameters are135,562; deployed float32 state is256 bytes/example.

This borrows the principles of context-dependent computation and causal
next-event supervision. It does not implement a language model, Mamba selective
SSM, symbolic CASM discovery, or prove equivalence to any of those systems.
Longer-term work needs evidence for depth/residual composition, useful
timescale spectra, calibrated state uncertainty and efficient exact execution.
Those changes are separate hypotheses, not capabilities conferred by this one.

Conceptual references: Vaswani et al., *Attention Is All You Need*, arXiv1706.03762;
Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*,
arXiv2312.00752. These motivate design principles, not reproduced benchmark claims.
