# ME087 — native noise-bounded retrieval and implicit median learning

This new complete-model study follows the frozen ME085 failure. ME086's already
registered second recipe continues unchanged. All attempts are visible; no
passed full gate existed when this prospective one-candidate study was frozen.
No component attribution, independent real qualification or architecture-winner
claim is made. No protected data is used.

## Input-to-output correction

Keep the99731-parameter native elapsed-rate fast/slow/8x8 recurrence, time-decayed
parallel filters, explicit state/cache and shared horizon queries. Let d be the
raw recent-value quiet-center innovation from ME085. Its new innovation is

    u_h = FIT_quiet_noise * sqrt(h)
    d_bounded = d / (1 + abs(d/u_h))
    quiet_center = actual_current + d_bounded.

Then abs(d_bounded)<=u_h at every learned weight, for every coordinate. The
formula does not identify grippers, use future movement labels, select a
posthoc true regime or hardcode a warmup. It prevents a low-noise coordinate
from borrowing a large obsolete movement into its nominal quiet center.
The old absolute moving center is exactly reanchored relative to the new
quiet center. The full mixture can still predict movement; this is not a
physical cap on the total forecast. Original mixture density, CRPS and
quantiles remain coherent.

## Direct proper point objective with implicit-function gradients

A second improvement trains the **actual median of that same distribution**.
Numerically solve F(q;theta)=0.5 with the unchanged40-step float64 inversion,
without retaining root-iteration autograd tapes. Then evaluate

    dq/dtheta = -(partial_theta F(q;theta))/density(q;theta).

The tensor expression `q_detached + (F_detached-F)/density_detached` preserves
the exact forward median while carrying this gradient. Density is floored at
1e-12 for numerical safety. Use a safe piecewise Laplace CDF with the correct
nonzero derivative at zero; differentiating a sign/abs implementation at zero
would incorrectly suppress that derivative. The initial software test caught
an incorrect zero derivative in a clamped draft; no optimizer ran with it.
The draft registration is preserved as PREFLIGHT_DRAFT_V1_NO_FITS.json; v2 is
frozen with the corrected source before any training.

The registered loss is normalized physical CRPS +0.2physical NLL +0.2factual
motion BCE +1.0normalized physical median absolute error, summed over actual
valid targets at1/4/16frames. Absolute error is a proper median score; combining
it with proper distribution scores does not replace the probability model with
an inconsistent surrogate. The implicit derivative is existing mathematics
(Figurnov et al.,2018, arXiv1805.08498), not a new TAC theorem.

All model parameters participate in training. All5 originals/optimizer/RNG are
Git-frozen before8/32-frame REPORT scoring;4096actual updates per seed, no
checkpoints selected by development performance. Original ME081 gates and
ME082 stronger conditions stay fixed. Additional99.1667%clusterCI describes
these6declared attempts only; repeated exposed-DEV tuning remains adaptive,
not independent statistical confirmation. Keep unchanged strict reference
replay at rtol1e-5/atol1e-6 and whole-forward float64 from original learned values.

The first trained original also has derived, serialized B2,T32 and B2,T1 complete
quantile/state executables. Actual FIT profile outputs and round-trip restore
agree under the unchanged tolerance. Alternating31-repeat within-process timing
measured1.87xprefill and1.59xincremental speedups versus the SAME eager model.
Both native training studies were sharing the workspace; these are unpinned CPU
profile observations, not rival comparisons, variable-shape qualification or
production latency guarantees. Independent derived-program verification is
separate from original complete forecast replay. All original weights remain
preserved as the canonical model, without changes from compilation.
