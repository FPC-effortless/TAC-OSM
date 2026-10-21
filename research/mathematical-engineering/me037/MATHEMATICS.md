# Native memory with context and a bounded horizon response

Write the globally train-standardized96-step per-channel window as x. Compute
mu=mean(x), sigma=sqrt(mean((x-mu)^2)+1e-5), z=(x-mu)/sigma. All operations use
observations available before the target origin. There are no future covariates.

A channel-shared nonlinear encoder maps the entire ordered z through96->42->64.
In parallel the native fast32/slow32 recurrence processes12ordered8-step patches
with ME036's learned gates, bounded tanh proposals and residual slow innovation.
Concatenate its64coordinates with the full-window64coordinates and16explicit
features (patch means, last value/change, mu/log sigma). This144-vector enters
SwiGLU64, a32-dimensional horizon basis and a learned32->H projection.

A separate context map gives two scalars u,v per observed window/channel:

    tau = 96 + 288 sigmoid(u)
    k   = 2 + 4 sigmoid(v)
    alpha_h = sigmoid(k (log(tau)-log(h))), h=1,...,H
    prediction_h = mu + sigma * (AR_h(z) + alpha_h*tanh(R_h)).

Initialize tau256,k4 and the final R projection at zero. The initial prediction
is exactly the inherited float32 AR prediction, rather than an approximately
copied float64 fit. After the first update, train both paths jointly. The first
update can have zero body gradients because the correction projection starts
zero; numerical tests and real traces require subsequent nonzero active gradients.
Step0 is never eligible as a native candidate.

For fixed context, alpha is nonincreasing in h:

    d alpha / d log(h) = -k alpha(1-alpha) <= 0.

Since tau<=384 and k>=2, for h>=384:

    alpha_h <= 1 / (1 + (h/384)^2)
    |prediction_h - (mu+sigma*AR_h(z))| <= sigma / (1+(h/384)^2).

The tanh is important: simply placing a learned gate in front of an unbounded
linear output would allow that output to cancel attenuation by increasing its
weights. Here the correction stays inside an explicit input-scale envelope.
These statements hold in exact arithmetic; unit tests allow a prospectively fixed
float32 rounding margin for the envelope. The initial AR-equality and saved
checkpoint replay requirements remain bit exact.

The bound concerns correction amplitude. It does not guarantee lower squared
error, safe physical control, stable gradients, useful memory or monotonic
accuracy with horizon. Native fast/slow coordinate bounds remain[-1,1] in exact
arithmetic. No associative scan or720-step recurrent rollout is implied.

Joint training minimizes global standardized forecast MSE, plus10times mean-square
AR weight/bias drift from the inherited teacher (means separately summed), plus
.01times mean-square *applied correction in global standardized units*. The latter
is sigma*alpha*tanh(R), not the locally normalized residual. Thus its optimization
units agree with the scored physical forecasting objective, avoiding ME036's
original ridge-reference weighting mismatch.

Context and GRU controls have the same144-input readout, correction envelope,
optimizer budget, AR teacher and selection opportunities. The context encoder
96->60->128 matches native's parallel encoder/recurrence parameter budget within1%;
the GRU control retains the64-context branch and substitutes GRU34->64 for native
memory. Other comparison lanes are explicitly unmatched practical references.

Serving must still encode the full normalized96-step window. Channel-shared
weights do not mean a single64-state serves all channels: native state is64*C
float32 values per batch element. No claim of lifetime streaming or microsecond
operational latency follows from these equations.
