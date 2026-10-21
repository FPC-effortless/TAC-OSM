# ME-030 completed — joint state/query head fails all gates

**Nine original real-data fits completed. No candidate promoted.** Protocol and
training source frozen/pushed at c413369 before fitting. Authentic ME022 KuaiRec
snapshot and source checkpoint/fingerprint checks passed. Exactly 10776 FIT and
5952 already-inspected historical DEV windows; reserved cohorts remain closed.
The additive arm exactly reproduces ME028/029 historical DEV predictions/metrics
under the unchanged recipe. This is not independent replication.

## Fixed300 endpoint (three-seed means)

| Arm | Active params | Brier ↓ | Log loss ↓ | AUC ↑ | ECE10 ↓ | Charged CPU sec/fit | B128 inference ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Existing additive native | 135377 | 0.164198272 | 0.493751638 | 0.811256590 | 0.027659302 | 20.7865 | 12.4848 |
| Joint native | 136145 | 0.167361674 | 0.501369437 | 0.810154624 | 0.049323629 | 20.8278 | 12.5347 |
| Joint order-free pooled | 136681 | 0.165147374 | 0.496372368 | 0.811550572 | 0.042120020 | 2.8002 | 0.4570 |

Charged costs include source verification/FIT preprocessing charged equally per
arm, setup, sampling and optimizer steps. Whole-fit time and lifetime process RSS
are in results.json; monitoring/serialization are excluded from training charge.
Same CPU, single thread, 3 warmups +9 inference measurements, full128 windows.
These are implementation-specific measurements, not portable performance claims.

Joint minus additive Brier by seed14001/2/3:
**+0.004875901, +0.001630526, +0.002983779**. All worse. Mean **+0.003163402**;
user-clustered 95% bootstrap interval **[+0.002220072,+0.004147656]**.

Joint native minus joint pool: **+0.006239153,+0.000932590,-0.000528844**;
mean **+0.002214300**, interval **[+0.001042349,+0.003336818]**. Intervals are
conditional on three training seeds and reused DEV; they do not cure selection
history, dataset reuse, encoder differences or unequal compute.

All four registered gates **FAIL**: gain against additive, gain against joint
pool, calibration, and FIT-mean state diagnostic. Mean replacement helps joint
native in every seed (Brier changes **-0.001802864,-0.001537212,-0.001040672**).
This input-distribution intervention does not identify causal or conditional
information value; it does further challenge this fitted model's state usage.

## Learning curves and interpretation

| Updates | Additive native | Joint native | Joint pool |
|---:|---:|---:|---:|
| 25 | .186946584 | .190276129 | .188632465 |
| 50 | .166551091 | .170205792 | .168893379 |
| 150 | .163654109 | .165414837 | .165209694 |
| 300, primary | .164198272 | .167361674 | .165147374 |

[Full curves](learning-curves.png), per-seed metrics and every update's FIT loss,
gradient and CPU receipt are retained. No earlier checkpoint is substituted for
the registered primary endpoint. Some curves degrade between150 and300; no
post-hoc optimizer change or early-stop selection was made.

This tests a mathematically real limitation: the old additive logit has zero
mixed state/query derivative. The new head permits those interactions but does
not learn a better predictor under this recipe. The new 90-input hidden layer
also uses a different initialization scale from the old26-input layer; this is
not an interaction-only parameter intervention. The additive family is a
subspace of the joint family, so representational inclusion alone cannot predict
finite-budget optimization. Do not infer that all joint readouts fail, that state
contains no information, or that additional latent reasoning is justified.

The next design question is source-only FIT-user cross-fitted conditional
residual prediction with identical nonlinear query interfaces. It remains
**unexecuted** here; it needs its own complete registration. Keep compression,
hierarchy and adaptive reasoning gated. The native recurrence and prior research
candidate are unchanged; ME030 is rejected rather than promoted.

## Parity and audit

Parameters PASS (<1% total gap), source/exposure PASS, selection PASS (no search,
fixed300 all arms), evaluation PASS for identical historical DEV only.
**Actual-compute parity FAIL**: joint pool consumes much less CPU; full FLOPs are
not matched. Profiler forward/backward subtotals for8FIT windows/four targets:
42,965,301 additive;43,114,805 joint native;1,690,789 joint pool. Unsupported/fused
and elementwise work, Adam updates and source preparation are not fully counted;
these are operator-coverage estimates, not full FLOP certificates. No strict
architecture-superiority or generalization claim follows.

All nine checkpoint and nine TorchScript-export paths reproduce **107136** real
historical DEV forecasts bit exactly. Nine mean-state paths also replay. Nine
fresh Torch-only processes each tested4real inputs without package imports;
maximum absolute error5.96e-8. All optimizer counters300, sampler RNGs and300
per-step clocks verified. Deliberately corrupted checkpoint is rejected before
source loading. Full regression suite: **925 passed +12subtests**. See replay.json,
artifact-audit.json and logs. No new reserved outcomes, deployment or merge.
