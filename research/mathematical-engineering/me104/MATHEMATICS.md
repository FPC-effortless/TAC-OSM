# ME104 — MedianSplitSensor

**Status:** PARTIAL exact-real scalar kernel; source and whole-model correspondence OPEN.  
**Variant:** `median_split_sensor` (ME104, descended from the inherited forecasting model).  
**Formal source:** `PNDS/MedianSplitSensor.lean`.  
**TAC source pin:** pending immutable source commit and path; `SOURCE_COMMIT` is a placeholder, not a pin.  
**Python reference:** pending source-aligned serial reference and tests.

## Representation and model boundary

The complete model has 99,731 learned parameters; the added head has 1,236. It leaves the inherited location `mu` unchanged and predicts two independent raw modifiers for the inherited width `sigma`. The model uses an equal-half Gaussian quantile transform: for probability `p`, select the left or right scale at the median split, transform `z = mu + s_side * Phi^-1(p)`, then apply the existing censored-lognormal output map `Y = max(0, expm1(mean + std*z))`. Each side has probability mass one half by construction. This description records the intended representation; the Gaussian probability normalization and the complete neural graph have not been formalized here.

The scalar scale equations are

```text
s_left  = sigma * exp(log(4) * tanh(raw_left))
s_right = sigma * exp(log(4) * tanh(raw_right))
```

The two side scales are positive whenever the inherited `sigma` is positive. At zero modifiers both scales equal `sigma`, while `mu` is independent of width and modifiers. The Lean kernel also proves a generic bound: if an exponential modifier is at least `-log(4)`, its positive base scale is reduced by no more than a factor of four. This generic theorem is not yet connected to the modifier `log(4)*tanh(raw)` by a Lean proof of the `tanh` bounds.

## Proof scope and open obligations

`PNDS.MedianSplitSensor` proves `leftScale_positive`, `rightScale_positive`, `medianLocation_unchanged`, `leftScale_modifier_zero`, `rightScale_modifier_zero`, and `exponentialScale_lower_bound`. These are exact-real scalar claims only. No proof is claimed for equal-half distribution normalization/CDF behavior, the censored-lognormal transform, the neural graph or gradients, IEEE behavior, calibration, or any whole-network/source invariant. The inherited affine core composition remains inherited and is not re-proved by this module.

Source-age capping at 256 hours remains an implementation-level condition to verify against the pinned source. A side-scale lower bound of `old_sigma/4` would imply a side variance of at least `old_variance/16`, under a positive inherited scale. That conditional bound does not establish preservation of the old full variance floor. Whole-network and IEEE invariance, calibration, and the precise source correspondence remain OPEN.

The Python serial reference, forward/backward and finite-precision comparison, source pin, and source-to-Lean declaration mapping are pending. Until those artifacts are installed and audited, this entry is not a whole-model mathematical completion or an empirical result.

## Exact implementation tensor graph (ME104)

Inputs are `values,observed,ages:[B,T,13]`, `clock:[B,T,6]`, supplied
`owners:[B]` and recorded `hours:[B,T]`. Pollutants use FIT-standardized log1p;
other measured channels retain the inherited FIT standardization. Source age is
external causal bookkeeping capped256hours, not inferred from hidden values.
Masks zero unavailable measurements before encoding. The inherited source13sum/
13mass,62evidence coordinates, count and8×8input-conditioned associative affine
scan are unchanged. Encoder/gates run on observed-prefix inputs; a read-only
postscan decoder cannot feed back into that same scan. Fixed-size streaming and
parallel sequence execution retain their original mathematical conditions.

The old current-state head and new width head use the same typed102feature layout:
13post-admission source means,13masses,62evidence means,6ages and8inclusive memory
read coordinates. The new module reuses the inherited belief read and query key,
without another encoder or memory write. New Linear(102,12) maps to left/right
modifiers. All99,731model weights are trainable; the extra head has1,236parameters,
not99,731. Global weights/buffers and activations are separate from the inherited
608byte FP32 core (plus metadata/source-age/runtime overhead). No on-device
latency is claimed.

Probability mass is normalized analytically by placing a Gaussian density with
its own scale on each half-line: each integral is1/2. CDFs meet at1/2, but density
can jump at the median. This is not the usual continuous-density two-piece normal
whose side masses vary with widths. Numerical CDF/quantile/NLL/zero-atom tests
include both threshold sides and FP32/FP64; the measure/integration argument is
NOT a Lean normalization theorem. Censoring creates an atom at rawzero; its CDF
uses the side containing the zero threshold, which may be the right side if the
uncensored median is negative. Positive outcomes include the log1p Jacobian.
All reported90%quantiles come from this same distribution, not an independent
quantile head or conformal adjustment. Arbitrary probability queries use ndtri;
the registered.05/.5/.95query uses the inherited fixednormal constants to preserve
initial FP forecasts exactly. Arithmetic means generally change with width.

The integrated training graph includes proper predictive NLL, physical median
MAE, log-location MSE, genuine masked-current reconstruction and raw interval
score. Gradients reach all31parameter tensors on authentic FIT batches. Joint
training may change shared state/median; the fixed-weight median invariance does
not establish trained point preservation. Statistical coverage and sharpness
remain empirical gates, and whole neural/IEEE correctness remains OPEN.
