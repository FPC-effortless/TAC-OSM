# Native clock/value world model — ME085/086

This is a native TAC model, not a replacement SSM. All proposed improvements
are unqualified until the frozen complete-model gates and original replay pass.
The master plan's capability-first priority applies; this is the physical lane,
not enterprise transfer, causal policy control or a commercial product.

## Full observed-input graph

At each cutoff, inputs are actual measured qpos, the same lag-one known command,
and elapsed **past/current** capture time. Publisher retained-frame timestamps
are uniform15FPS; capture time is a distinct measured field. Only relative
elapsed time is encoded; no future capture timestamp, episode ID, absolute epoch
or unobserved future command enters the graph. Targets remain future retained
frame endpoints, not guaranteed fixed wall-clock durations.

Let r_t=15 delta_t>0. Velocity is (x_t-x_previous)/r_t and acceleration is its
elapsed-scaled innovation. A cumulative-sum clock and prefix maximum compute
per-coordinate time since an observed value changed. The99-dimensional input
contains qpos, command-relative error, asinh velocity/acceleration/command change,
log observed age, log elapsed ratio and asinh time-weighted local innovation.

The encoder and both depthwise temporal filters operate in parallel across
batch/time/channel. Five causal taps per filter are multiplied by exp(-age/tau),
with tau4/16 nominal ticks, before the learned tap-weight reduction. All temporal
windows are supplied by unfold views and small rolling caches; no replay of
older history is needed for one incremental update.

The native fast/slow/8x8 associative recurrence remains sequential in time:

    M_decay = exp(logsigmoid(retention)*r) M_previous
    read = M_decay normalized_key
    g_f = -expm1(logsigmoid(-fast_logit)*r)
    f = (1-g_f) f_previous + g_f tanh(candidate)
    g_s = -expm1(logsigmoid(-slow_logit)*r)
    s = (1-g_s) s_previous + g_s tanh(s_previous+tanh(innovation))
    M = project_Frobenius_ball4(M_decay + sigmoid(write)*(value-read) key^T)

Zero-initialized f/s remain within[-1,1], and the associative state norm<=4.
L2-normalizing keys controls scale, **does not guarantee learned query alignment**.
The nonlinear recurrent dependency has no claimed associative parallel scan.

## Direct value preservation and shared distributional query

Each horizon query jointly accesses local encoded features, native state and
known context through one shared SwiGLU readout. Its quiet-center prediction is

    c_q = x_t + sum_{j=0..7} softmax_j(l_j - age_j/8) (x_{t-j}-x_t).

Thus c_q is a learned convex combination of actual recent observations, lies
inside their coordinate-wise envelope, and exactly preserves a constant input
up to arithmetic precision. It avoids requiring dense encodings to reconstruct
fine-valued observations. It does not restore input precision already lost in
the inherited float32 loader and does not guarantee improved denoising.

The motion center retains a learned, bounded native correction from measured
velocity, FIT-identified servo proposal and nonlinear residual. A two-Laplace
mixture provides one coherent CDF, density, CRPS and quantiles for each physical
coordinate. Quiet noise is FIT-only; moving scale is at least twice quiet scale.
All three horizons and coordinates are decoded in parallel; they are marginal
forecasts, **not an autoregressive joint trajectory or optimal action plan**.

Training uses float32,4096actual updates per seed. The qualified normative
inference promotes the original trained values and the complete forward graph
to float64; original rtol1e-5/atol1e-6 stays unchanged. No FP32-fast-path accuracy
or throughput claim is inherited from this reference.

## Computation and prior art

Only native state evolution requires a time loop. The encoder, elapsed feature
prefixes, time-decayed filter reductions, raw-value retrieval, horizon readout
and distributional inversions expose parallel axes. The decoder can skip
unrequested historical outputs at inference. State is caller-owned and pure
input/output; no cache mutation or execution permissions are hidden in the
model. Functional/full/chunk/step equivalence is tested separately from learned
quality. No sub-millisecond or GPU fusion claim is made without measurements.

The design applies existing time-aware recurrent models, causal dilated
convolutions, attention-based value retrieval, mixture density networks and
gated delta memory; these mechanisms are not inventions of TAC-OSM. The
experiment concerns whether their native integration improves actual learned
forecasts. See manifest literature links and the original preserved attempt
ledger. Parameters99731; no padding or inactive learned tensors.

## Trained original reference execution audit

The first original ME085 checkpoint's actual FIT-input-to-quantiles export has
3723nodes for32-frame prefill and1491nodes for one incremental step, including
all13state outputs. Per-entity cache has1331tensor values,10648bytes float64
excluding weights/container overhead. Equivalent outputs were checked, not
inferred from symbolic shape alone.

An alternating31-repeat same-model CPU measurement found a1.75x median reduction
by skipping unrequested historical forecast heads (25.90ms to14.82ms,B24,T32,
float64). Concurrent training and unpinned CPU affinity limit this observation.
It is neither a rival speed comparison nor an inference SLA, GPU/fusion result,
or constant-time-history accuracy claim. Original raw measurements are in
ME085/EFFICIENCY.json. The entity-cache adapter stores only caller-owned state;
it rejects stale model versions, conflicting elapsed-time retries and gaps.
