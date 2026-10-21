# ME025 final readout — primary improvement gate failed

All nine registered neural fits completed in original GitHub workflow
[39521144405](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39521144405),
head `d8c0eb3cd65c6a71bc23fa463f1ee9318b21ef04`. Both original CI jobs passed.
The source ZIP/artifact digest and checkpoint/prediction hashes are preserved.
No fits were repeated in the subsequent audit or ME026 execution study.

The primary kernel-head minus RMS-only mean Brier is **−0.000070352**.
Paired differences: +0.000004244, −0.000104911, −0.000110389. It fails both
all-three-seeds-improve and mean <=−0.001 requirements. Mean ECE worsens by
+0.001605190, within the +0.005 guard, but that cannot rescue the failed gate.
The six-coefficient feature head is not promoted; retain ME024 RMS as the
established native development candidate.

| Model | Seed 14001 Brier | Seed 14002 | Seed 14003 | Mean Brier | Mean ECE | Training CPU s |
|---|---:|---:|---:|---:|---:|---:|
| tac_rms | 0.168493801 | 0.165891335 | 0.165204156 | 0.166529764 | 0.027447297 | 81.354 |
| tac_kernel | 0.168498046 | 0.165786425 | 0.165093767 | 0.166459412 | 0.029052487 | 81.785 |
| tiny_transformer_kernel | 0.167398263 | 0.167659386 | 0.165473389 | 0.166843679 | 0.030079796 | 168.401 |

These Transformer scores are descriptive on reused development data.
Parameter, data/exposure and fixed selection parity are checked; **full
compute parity is unestablished**, so no architectural superiority claim is
allowed. The Transformer used about twice the recorded optimizer CPU of either
native arm. That is not a FLOP/energy audit or a broad Transformer comparison.
The small MLP reference has mean Brier 0.174375374, with unmatched
capacity/representation. Its metrics were retained, but its weights and
probabilities were not saved, so that reference cannot be independently
replayed. No replacement reference was trained to hide this limitation.

The recorded-only audit verifies all nine checkpoint/prediction hashes before
reading any cached responses, then the authenticated ME022 snapshot and its
canonical tensor/entity/time fingerprints. All 53,568 original neural
forecasts replay within 1.192093e-7 probability; all recorded metrics satisfy
the preregistered replay tolerance. Optimizer states/steps, sampler draws,
FIT-only duration/popularity and original exposure counts pass.

Earlier workflow 39520954861 failed repository regression before training;
39521115775 was cancelled during regression before training. Their real-data
jobs have no executed steps. All attempt metadata is retained in the ME026
artifact. The successful run contains the original nine complete fits.

The old causal preflight tested zero-head forecasts, which can hide leakage.
ME026 adds nonzero-head prefix tests and a deliberately unmasked-attention
falsifier, which is detected. This does not identify leakage in the original
masked model or alter any fitted weights, scores or outcomes.

Only 10,776 FIT and 5,952 already-exposed DEV windows from the same authenticated
KuaiRec archive are used. Reserved users/future outcomes remain closed.
See ../me026/REPRODUCE.md and ARTIFACT.md for exact replay and archive access.
