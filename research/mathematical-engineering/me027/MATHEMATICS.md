# ME027 mathematical contract

For past events i<t, define u_i=[embedding(item_i),metadata_i]. The shared
observable vector is [embedding(offered_item), standardized_duration,
EWMA_2/8/16/32/64, FIT_item_popularity, mean(u_i)], dimension 26. Every component
uses only information available before the forecasted response. The current
item/duration are observed before response; its feedback is never included.

Native TAC keeps the original gated fast/slow convex updates and split RMS
features phi(h), with epsilon .01. The GRU is PyTorch's standard cell with 34
coordinates followed by a 34→64 projection and the same split RMS map. The
Transformer has the previously fixed two causal attention layers and a 16→64
projection. Their readout is

```
logit(p) = logit(prior32) + w_h^T phi(h) + w_x^T x + b
```

The MLP is `logit(prior32)+v^T tanh(A x+a)+b`, with 252 hidden units. All head
weights/biases begin at zero, so initial forecasts agree exactly. Initial zero
heads cannot establish causality: nonzero-head tests and a deliberately
unmasked attention mutant are required. The mean pooling and EWMA coefficients
have respectively uniform and normalized exponential derivatives with respect
to observed features; neither can include an unobserved response.

TAC retains 135067 coefficients; GRU 135125; Transformer 134907; MLP 135057.
Embedding is 128000 in each. All coefficients belong to active scored paths;
there is no inert parameter padding. Raw hidden coordinate counts are not equal
capacity or compute. The pooling/statistical input exposes a common sufficient
feature interface, not a proof that those features are sufficient for the task.
The neural sequence models can also use order and event-conditioned content;
the MLP deliberately tests whether the shared summaries already suffice.

For a frozen learned-state model, let mu be mean(phi(h)) over FIT-first1024.
Replacing phi(h) by mu holds the common features/prior fixed and preserves its
FIT-average linear latent logit, because w_h^T E[phi(h)]=E[w_h^T phi(h)]. Do not
use phi(E[h]), which would insert a nonlinear normalization/intercept shift.
An error increase after replacement is an OOD reliance diagnostic, not proof
of causal state utility, identifiability or independent generalization.

CPU checkpoint k is the first optimizer step n with charged_CPU(n)>=k. This
introduces a bounded measured overshoot, not exact continuous-time matching.
The primary gate explicitly requires all 20s spends <=20.2s. Update checkpoints
fix n and therefore target presentations; CPU checkpoints vary n. Both lanes
are needed because different per-step computational costs prevent asserting
simultaneous equality of examples and actual spend. Full resource-parity claims
remain barred even if a bounded development gate passes.

TAC/GRU inference could retain raw hidden state plus 11 pooled input sums,
one count, five EWMA sums/counts and a prior32 FIFO when weights are frozen.
This is 64+22 or 34+22 float coordinates plus 32 responses; no such new streaming
API is implemented in this study. All scored paths recompute real 128-event
windows. Transformer full-window attention uses quadratic working activations;
no KV-cache implementation is benchmarked. Training BPTT/attention working
memory is distinct from persistent hidden state. Profiler matrix FLOP subtotals
omit opaque kernels/nonlinearities and cannot certify a full compute match.

## What would establish additional predictive state information?

Interpretation addendum; no fitting rule or gate changes. Define C at the raw
observable level: offered item/duration, causal prior, fixed rate summaries,
FIT popularity, prior-item histogram and metadata means. Each fitted model
projects that histogram through its own learned item embedding, so the 26
feature **channels** are shared but their learned numerical values need not
match after training. Let H be one frozen model's learned ordered-history state.
For the true response distribution, write

```
m0(C)   = E[Y | C]
m1(C,H) = E[Y | C,H]
R0* - R1* = E[(m1(C,H)-m0(C))^2] >= 0
```

Here R0* and R1* are the corresponding Bayes-optimal Brier risks. This follows
from the conditional-variance decomposition and defines additional predictive
information; it is unrelated to intervention identification. For finite trained
readouts f0(C), f1(C,H):

```
R(f0) - R(f1) = (R0* - R1*)
               + E[(f0-m0)^2] - E[(f1-m1)^2]
```

Thus observed score differences mix information with representation, readout,
estimation and optimization error. At CPU checkpoints, exposure also differs.
The MLP has a nonlinear common-feature head while the recurrent arms have
linear additive state/common heads; this is a genuine allocation difference,
not a controlled switch that removes state from an otherwise identical model.
Neither this experiment nor mean-feature replacement estimates the Bayes
information term in isolation. It can locate practical failures and guide a
subsequent native readout intervention, but cannot identify intrinsic state
utility or a causal memory effect from a favorable CPU score alone.
