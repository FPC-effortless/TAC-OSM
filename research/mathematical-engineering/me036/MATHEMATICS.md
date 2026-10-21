# Native patch dynamics and full-horizon readout

For each measured channel, the globally training-standardized history x has
96 observations. Compute input-only mean mu and scale
sigma=sqrt(mean((x-mu)^2)+1e-5), and z=(x-mu)/sigma.
Nonoverlapping ordered vectors p_j=(z_8j,...,z_(8j+7)) retain within-patch order.
Parameters are shared across measured channels; this differs from ME035's
joint seven-channel state. No future observations/covariates enter the model.

For j=1,...,12, f,s in R32:

    g = sigmoid(W_g[p_j;f] + b_g)
    f' = f + g * (tanh(W_e p_j + W_f f + b_e) - f)
    v = sigmoid(W_v[f';s] + b_v)
    r = tanh(W_s[f'-s;s] + b_s)
    s' = s + v * (tanh(s+r) - s)

Gate biases start at -log(tau-1), yielding initial rates1/tau when input-dependent
terms vanish. Fast tau span2..16 patches; slow8..64 patches. These are starting
update scales, not guaranteed half-lives or learned spectral bounds. Slow innovation
starts small and the fast recurrent matrix starts orthogonal with gain.9. The
near-identity slow proposal improves its local path near zero innovation/state;
nonlinear gates still prevent an associative exact scan.

Normalize final [f;s] by sqrt(mean([f;s]^2)+.01). Context c has16coordinates:
12 patch means, last z, last-minus-final-patch-mean, mu, log(sigma).
The correction is a learned low-rank horizon prediction:

    [a;b] = W_mix [c;h] + b_mix,   a,b in R64
    q = SiLU(W_basis (SiLU(a)*b) + b_basis), q in R32
    y_hat = mu + sigma * (A z + b_AR + W_H q + b_H).

A has Hx96 entries, W_H has Hx32 entries; both train jointly from scratch.
This gives full forecasts for every step1..H, not interpolated endpoint scores.
The shared-channel design keeps parameters independent of channel count; runtime
still scales with observed channels and forecast length. Context/GRU controls use
exactly the same AR/readout, replacing h with an active MLP or projected GRU.
No zero/inactive coefficients are added for count matching.

The patch recurrence has12 sequential updates rather than96 hourly updates.
This is an architectural change with no equivalence claim. Rolling mean/variance
could reduce statistic computation; because all retained z change as the window
moves, this implementation re-encodes all12patches per forecast. No constant-time
streaming claim follows from its64-coordinate state.

Metrics use train-only global scaling, then unweighted mean squared/absolute error
across *all* output steps/channels and eligible test anchors. Local normalization
is inverted before those metrics. Different datasets have different physical
cadences: ETT hourly; ETTm1 fifteen minutes; Weather ten minutes; Electricity hourly.
A96-step context therefore has different physical duration across datasets.

For future float32 dot-product verification, let u=eps32/2 and
 gamma_n=n*u/(1-n*u). The prospectively registered absolute error bound is
(gamma_n+2*u)*sum_i(abs(a_i*b_i)), using float64 products/reference. This bound
is for the linear operation only, not a blanket tolerance for nonlinear networks.
Same-CPU checkpoint replay must remain bit exact. ME035B's prior1e-6 initial
ridge-equivalence failure is unchanged and is not covered by a retrospective bound.
