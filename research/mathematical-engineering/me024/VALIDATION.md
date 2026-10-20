# ME024 validation

Before fitting: 897 repository tests passed, including eight new tests covering
original weights/forecast identity, all coupled Jacobian entries against
float64 autograd, block factorization and determinant, RMS derivatives/bounds,
prefix causality, bounded streaming, feature-mean logit preservation and
active gradients/optimizer/standalone round trips. Synthetic inputs serve only
these mathematical tests, never real-world performance claims.

Recorded replay passes for all 12 checkpoints and 12 standalone exports.
Each of the checkpoint, standalone, FIT-feature-mean and feedback-disabled
paths reproduces 71,424 historical development forecasts with **zero** maximum
probability difference. Metrics, feature means, optimizer moments/steps,
reconstructed sampler RNG, parameter counts, exposures, trace steps and compute
ceilings match the frozen record. Streaming state/logit error is exactly zero
on 32 genuine 128-event windows per model (tolerance fixed at 1e-6).

A deliberately corrupted checkpoint is rejected before nonexistent parent-data
paths can be accessed. This verifies the fail-closed ordering of artifact
validation before response loading.

Full float64 chronological Jacobian products use 96 real FIT trajectories:
eight windows × twelve trained models. At depth 128, the mean spectral norm
of the initial-to-terminal hidden-state derivative is:

| Variant | Mean norm | Range across 24 probes |
|---|---:|---:|
| plain | 0.071590 | 0.045979–0.105305 |
| RMS | 0.053135 | 0.045836–0.057729 |
| coupled | 0.066014 | 0.055343–0.088627 |
| coupled + RMS | 0.052811 | 0.045403–0.059847 |

The artifact stores all 64 singular values at each registered depth, along
with local feedback/joint norms and state amplitudes. These are sampled local
sensitivities, not a global contraction proof, a loss/parameter gradient or a
causal explanation of predictive performance. RMS improves prediction despite
a smaller measured initial-state sensitivity; larger product norm is not used
as a success criterion.

Twelve fresh-process Torch-only probes pass with zero streaming/window logit
error on four genuine windows per model. Archive SHA-256 checks and all 70
indexed payload files match after extraction; the restored models undergo the
same full recorded audit. Machine records retain these outcomes. The first fresh-process attempt used Python `-I`, which
hid this environment's user-site Torch installation and failed before loading
a model. The corrected probe uses a temporary working directory and asserts
that neither `tac_osm` nor `scripts` is imported before or after execution.
It changes no model, training source, fit count or numerical tolerance. The
initial failure and correction are retained in `runtime-probe-amendment.json`.

No reserved outcome, additional source cohort or newly fitted comparator was
used for validation. Local replay is artifact verification, not independent
scientific replication. TorchScript upstream deprecation warnings are recorded;
no newer exporter or cross-version runtime has been qualified.
