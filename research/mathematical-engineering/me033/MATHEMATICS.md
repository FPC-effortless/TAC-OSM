# ME033 — Semantic representation adaptation and composable moments

Let E_R be the R-trained frozen item embedding and E_theta a separately S-trained
copy initialized to E_R. Let q be the known offered item and m_t the3metadata
fields attached to a genuine observed past event. The extra11-dimensional inputs
are

\[
u_H=\left[\frac1n\sum_{t=1}^n E_\theta(i_t),\frac1n\sum_t m_t\right],\qquad
u_Q=\left[E_\theta(q),\frac1n\sum_t m_t\right].
\]

Identical11->56->56->64 MLPs and split RMS map each input to h. The same ME032
centered correction combines h with shared common features C. Query paths retain
past metadata and all of C's genuine history summaries; the treatment removes
only the EXTRA adaptive historical-item-identity projection. The query control
is not a history-free model, and the history control is not an order-sensitive
recurrent architecture.

## What freezing reveals

Common C already contains current E_R(q), pooled historical E_R and pooled
metadata, followed by an invertible coordinatewise R normalization. If E_theta
is frozen at E_R, u_H and u_Q are exact deterministic functions of C. Thus the
frozen extra latent cannot increase ideal conditional information beyond C.
A finite fitted predictor may still improve because its nonlinear function class,
training trajectory and additional S supervision differ. The audit reconstructs
frozen latents from C on real records to verify this mechanism, not merely assume it.

When embeddings adapt, u_H changes the learned projection of historical item
counts. u_Q learns an item-specific representation of the known current offer.
Both use already available original observations; they do not introduce new data
channels. Their gains are finite-model representation effects, not estimates of
conditional mutual information or causal memory utility.

The declared interaction is a loss contrast:

\[
I=(R_{H,A}-R_{Q,A})-(R_{H,F}-R_{Q,F}).
\]

Negative I says adaptation favors the history pathway relative to the query
pathway under this fixed finite learning procedure. Freezing128000coefficients
changes the trainable capacity; the cross-setting interaction is NOT a parameter-
matched architecture ranking. Matched history/query pairs have exactly equal
trainable counts within each setting. Bootstrap intervals condition on fitted
models and3seeds; rotated training roles overlap, and prior FIT design exposure
remains. No information-theoretic identification or independent qualification.

## Moment state algebra

For fixed checkpoint E_theta, define history state

\[
S_n=(\sum_t E_\theta(i_t),\sum_t m_t,n)\in\mathbb R^{12}.
\]

Observe adds `[E_theta(item),metadata,1]`; segment merge is vector addition. The
identity is the zero vector and addition is associative over exact reals. Readout
first divides11sums by count, then applies the learned MLP and RMS. This does NOT
make that nonlinear readout or TAC's nonlinear recurrence an associative scan.
A zero count uses the implementation's denominator floor1; no empty-history
predictive accuracy is studied, and biased MLP output need not be zero.

The query variant only retains metadata sums/count in4coordinates, obtaining
E_theta(q) at readout. Component storage is48bytes history or16bytes query in
float32. The64dimensional latent is a derived activation, not the stored summary.
Native fast/slow state is64float32coordinates (256bytes). These counts exclude
shared common-context/EWMA/prior bookkeeping, model weights and activation batches;
no48byte total forecast runtime claim follows. This study tests only128-event
windows, not unlimited streams.

Floating-point addition is not exactly associative. Bulk prefix means, sequential
updates and segmented sums may differ by rounding; tests compare latents and
predictions under fixed tolerances, not bit-exact real arithmetic. Counts are
float32 too, so precision limits matter on very long streams. No long-stream
accuracy or update qualification is inferred from bounded-window equivalence.

Critically, if optimizer steps alter E_theta, previously cached sums use old
embeddings. An arbitrary new embedding table cannot generally be applied to a
stored8dimensional old sum without replaying event identities. The incremental
API is checkpoint-fixed inference, not cost-free online representation learning.
During training the implementation recomputes genuine prefix means, preserving
ordinary backpropagation to the current embedding weights.

## Credit and native control

All learned extra branches use the centered head from ME032, with zero initial
output/direct coefficients; every model initially predicts the inherited teacher.
The native control retains the exact original fast/slow updates and uses the SAME
R-initialized embedding, without inheriting R recurrent weights. A gain relative
to earlier ME032 therefore cannot be isolated as a warm-start effect: other setup
choices and controls changed, and no historical before/after gate is declared.

Common-only S adaptation tests extra supervised exposure through the original
1205parameter head, without an additional memory or trainable item table. It is
an intentionally unmatched practical reference. It can warn that more elaborate
memory is unnecessary; it cannot certify an equal-capacity architecture winner.
Any future native changes need their own source-frozen, resource-controlled test.
