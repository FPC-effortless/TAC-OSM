# Post hoc reference objective correction — ME036C

Registered after initial frozen test scores revealed extreme ETTh2 long-horizon
errors in the original reference; **not prospective qualification**. No native
architecture, checkpoint, selection, data split or hyperparameter changes.
Original reference, scores, protocol and failed practical behavior remain intact.
This is a fixed mathematical objective correction, not alpha/feature search.

Original ridge regressed z=(x-mu)/sigma plus intercept against q=(y-mu)/sigma
with unweighted least squares. Its deployed prediction is mu+sigma*(zA+b).
Thus its fitted objective is normalized residual error, whereas the reported
train-standardized forecasting objective is sigma^2*(zA+b-q)^2. They differ.

Fit20 corrected references, same datasets/H/stride24 training anchors, same
channel sharing, same97H coefficients and alpha100, intercept unpenalized.
Accumulate the Gram/cross products of a=sigma*[z;1], target=y-mu, in float64,
then deploy coefficients in float32 through the exact same Ridge inference class.
No validation tuning, seed search, replacement native fit or altered acceptance
threshold. Every corrected coefficient file freezes before its descriptive test
scoring. Report both versions. Existing registered gate remains against the
registered matrix; this correction receives a separate comparison with no claim
of new untouched evidence or architectural superiority.
