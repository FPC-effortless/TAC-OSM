# ME024 — native feedback rejected; split RMS improves historical development

The registered **slow-feedback primary gate failed**. Coupled+RMS minus RMS-only
Brier is +0.000219208, +0.000005581 and +0.000016667 across seeds 14001–14003;
mean +0.000080485. Every difference is unfavorable. Coupling without RMS also
worsens all three seeds. The added feedback is not promoted.

A prespecified secondary result supports split RMS readout conditioning:
plain native mean Brier 0.167689800 becomes 0.166529764 (0.692% lower), with
improvements in all three paired seeds. Mean ECE improves 0.034814499 to
0.027447295. This is an architecture optimization candidate measured on reused
real development data, not independent validation or a whole-system win.
The normalization adds no parameters and leaves native recurrent dynamics
unchanged. All variants retain causal prefix training from ME023.

| Native variant | Seed 14001 Brier | Seed 14002 | Seed 14003 | Mean Brier | Mean ECE | Feature-mean Brier delta | CPU s |
|---|---:|---:|---:|---:|---:|---:|---:|
| plain | 0.168568 | 0.168159 | 0.166342 | 0.167690 | 0.034814 | +0.000634 | 19.40 |
| rms | 0.168494 | 0.165891 | 0.165204 | 0.166530 | 0.027447 | +0.001189 | 19.82 |
| coupled | 0.169057 | 0.168267 | 0.166423 | 0.167916 | 0.036380 | +0.000517 | 20.40 |
| coupled_rms | 0.168713 | 0.165897 | 0.165221 | 0.166610 | 0.028380 | +0.001139 | 20.42 |

The primary feature-mean guard also fails: coupled+RMS replacement deltas are
−0.000080125, +0.001549354 and +0.001947702, requiring at least +0.0002 for
every seed. The ECE guard passes (mean increase +0.000932731, ceiling +0.002).
The control replaces the post-normalization feature vector by its FIT mean;
it preserves the FIT-average latent logit and avoids the extra intercept shift
from averaging raw state before nonlinear normalization. It remains an OOD
intervention, not causal proof of useful memory. RMS-only replacement increases
Brier in all seeds, but the first delta is just +0.000033753.

Turning off feedback after training also improves average Brier in both
coupled variants (−0.000303880 without RMS, −0.000075948 with RMS). This shares
weights and is not a retrained comparator. It reinforces the negative diagnostic
without establishing a general impossibility of slow-to-fast coupling.

## Execution and provenance

Twelve fits, exactly 300 AdamW updates each, no pilots, extra seeds, early
stopping, checkpoint selection or retuning. Each has 135,050 parameters,
38,400 sampled windows, 4,915,200 historical-event presentations and 153,600
causal supervised-response presentations. Training used 240.129 process-CPU
seconds in total, all fits below the fixed 180-second ceiling. Equal target
exposure and capacity are established; complete FLOP/energy equivalence is not.

The source was committed and pushed at
`969a92e1280cf16248d9247ef5d49532139d9e46` before fitting. Executable training
source, core and machine contract remain unchanged. Subsequent protocol prose
edits only fix spacing; the mathematics additionally clarifies what the initial
state Jacobian does and does not measure. Learning curves show every recorded
checkpoint; only step 300 is used in the declared comparison.

The official authenticated KuaiRec source is reused through its original
ME022 snapshot. Exactly 10,776 FIT windows and 5,952 historical DEV windows
are used, with the original 128 genuine observations and causal cutoff rules.
No new outcomes or reserved cohorts were accessed. The same reused development
cohort and three training seeds limit inference; no test or superiority claim.

## Mathematical and execution result

Native slow-to-fast feedback, split RMS features and their full joint
Jacobian are implemented in the optional CoupledNative core. All 4,096 local
Jacobian entries match float64 autograd; factorization and determinant are
checked. Native state bounds, RMS derivative bound 10, bounded feature norm,
causal prefix forecasts, active gradients, original-model identity and
standalone/streaming equality pass mathematical tests. Full chronological
Jacobian probes and replay results are reported separately in VALIDATION.md.

All raw final weights, optimizer/RNG states, executable exports, probabilities,
feature means, traces and negative outcomes are retained. Nothing is merged,
deployed or promoted. See REPRODUCE.md and ARTIFACT.md for the delivered archive.
The separate exact-discretization/scan design in ARCHITECTURE-DIRECTION.md is
an unfitted proposal, not an ME024 improvement.
