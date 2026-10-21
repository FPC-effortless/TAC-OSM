# ME026 — exact native execution and affine observational summaries

The native learned state equations remain those of ME024 RMS-only and ME025:
fast memory is updated from its previous state and the observed event; slow
memory reads the updated fast state and its own previous value. All trained
matrices, gate biases, normalized duration, prior and feature order are copied
from each frozen original checkpoint. There are no new trainable parameters.
Both native states stay coordinatewise in [-1,1] from zero by convex gated
updates. This is not a whole-network contraction or long-history guarantee.

The state readout applies the unchanged split RMS transform, with epsilon .01,
then the same linear head. The runtime supports only coupling=0 and RMS=True;
it rejects other native contracts. Compilation changes execution, not equations.
It does not establish training acceleration; no training is performed here.

## Affine segment monoid for fixed observation summaries

For half-life k, gamma=2^(-1/k), and observed binary responses y:

```
S' = gamma S + y
C' = gamma C + 1
summary = S/C
```

Starting with zero, after L events:

```
a = gamma^L
b = sum_i gamma^(L-1-i) y_i
c = sum_i gamma^(L-1-i)
(S_out,C_out) = (a S_in+b, a C_in+c)
```

Represent a segment by (a,b,c). Applying segment A and then B gives:

```
B compose A = (a_B a_A, a_B b_A+b_B, a_B c_A+c_B)
```

Composition is associative over real arithmetic and the empty segment is
(1,0,0). This follows directly by substitution into the two affine maps.
The implementation computes arbitrary segment lengths, including empty ones;
float64 tests check irregular splits and both association orders. Rounding
means floating-point composition is tolerance-equivalent, not bitwise equal.

For nonempty histories C>0, each normalized summary is in [0,1], and its exact
feedback derivative is gamma^(L-1-i)/C. Derivatives are nonnegative and sum to
one over observed responses. No future/target response enters this sum. The
sum and count states have the linear derivative gamma per event; this fact
applies only to fixed summary statistics, not learned fast/slow dynamics.

The summary subproblem can use an associative prefix scan or independent block
summaries. The current standalone window path precomputes the same normalized
128×5 coefficient matrix used by ME025 and performs the same matrix product.
The incremental path updates five sums and five counts. No custom parallel
scan kernel is implemented or benchmarked, and no scan claim is made for TAC's
nonlinear state-dependent recurrence.

## Whole-forecast streaming contract

RMS-only streaming state: [fast32,slow32], 64 float32 values = 256 bytes/entity.
Kernel-native state: [fast32,slow32,sum5,count5], 74 float32 values = 296 bytes.
The caller separately maintains a causal FIFO of the last 32 observed responses
(128 bytes at float32), valid-count/reset rules and entity identity. The model
rejects kernel readout with nonpositive counts. It requires the caller to supply
the same prior=(last32_sum+1)/34 at the forecast boundary, after the historical
event is observed and before the offered item's response is known.

`stream_step(item,metadata,state)` consumes an actual historical event;
`readout(state,next_item,next_duration,prior)` forecasts the subsequent response.
The target item's known duration and FIT-only popularity are allowed; its
outcome is not. Readout explicitly consumes the correct 79 kernel features.
ME025 inherited the original 73-feature streaming readout, which cannot be
called directly after replacing its head; this adapter supplies the complete
forecast API without editing the frozen fitted implementation.

All equivalence claims concern a fresh zero state followed by the same 128
events as the original checkpoint. Continuing beyond 128 events is not equal
to resetting on each sliding window: both native state and summary support
would differ. No long-stream predictive advantage follows from the API.
Full-window inference still materializes O(T) input features; streaming state
is constant with history, while training BPTT remains O(T). The optional
TorchScript runtime is validated only in its pinned Torch version.
