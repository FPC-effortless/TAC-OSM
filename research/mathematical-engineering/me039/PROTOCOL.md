# ME039 — efficient execution of the new native TAC architecture

Registered before any real-data runtime measurement. This is architecture
engineering, not a diagnostic explanation of why memory works, a new learning
algorithm, a new accuracy fit or a capability/generalization experiment. Preserve
ME038 architecture and weights, its passed power target and failed CPU/legacy
gates. Do not retune on Tetouan/household/closed KuaiRec. Existing selected models
may be replayed for execution correctness, without changing their forecasts.

## Implementation and contract

Compile ONLY ME038's native fast/slow/innovation-memory core with the installed
TorchScript compiler, keeping the surrounding joint AR/context/forecast path.
All9562core weights, parameter names, state equations and128coordinates/channel
remain unchanged. No pruning, extra training data, different recurrence or hidden
parameter capacity. This is a new execution path for the new native architecture,
not a new architectural-superiority or first-in-literature claim. Eager remains
available; no silent change to the old frozen trainer or default package imports.
TorchScript deprecation is recorded; version-dependent behavior is not concealed.

Unit fixtures already established that scripted gradients can round differently.
No bit-exact retraining or unchanged optimizer trajectory claim. Real-data tests
below decide numerical agreement and performance, with no post-result relaxation.

1. All60selected ME038 native checkpoints, including both original lanes, must
   match their original saved two-window validation forecasts BIT EXACTLY after
   core compilation. Original weights/source/file hashes remain immutable.
   Raw-input exports must agree exactly with the same eager physical wrapper;
   physical float32 roundtrip is not compared with separately standardized inputs.
2. Before scoring any new primary accuracy result, require a separate future
   registration and fresh real cohort. This study does NOT rescore final test
   outcomes, conduct another accuracy search or overwrite ME038's failed gates.
3. Numerical derivative/optimizer agreement uses ONLY authentic TRAIN windows
   from ETTh1 and Electricity, four full horizons cases H96/720, both existing
   native seeds38001/38002:8cases. Three repeats, two execution paths,64updates
   per path =3072optimizer steps total. These are numerical/runtime replay
   continuations, not candidate capability models; no validation selection,
   predictive improvement, test qualification or world-model training claim.

## Real-data timing and replay continuations

Each case/repeat starts from the SAME original exposure-lane selected ME038
checkpoint, full model state including anchors. Each arm receives the exact same
64batches of four train-only windows, sampler seed=seed+3000+repeat. All paths
use AdamW lrAR.0001, others.001, wd1e-4, clip1 and original global-standardized
MSE+10anchor+.01applied-correction loss. No learned component disabled for speed.
Alternate path order per case/repeat; one thread, no concurrent local fitting,
scoring, export benchmark or archive upload/hash workload during measurements.
CPU model/quota, versions, peak RSS and actual sample presentations are recorded.

Record construction, checkpoint loading, compilation, optimizer setup and all
step CPU costs. First16updates form the cold segment, including construction;
steps17..64 form the steady48-update segment. Both costs are reported, never
pretending compilation is free. Check/serialization overhead separately recorded.
Record all three repeats; case speed ratio is paired median across repeats,
overall speed geometric mean of eight case ratios. No selection of favorable
trials. Timing targets are >=1.20x steady training and >=1.10x total64-step CPU
including construction. Missing/failed pairs make target FAIL, no replacement.

Save complete trace, optimizer/RNG/model final state and gradient/parameter
snapshots at1/16/64for every attempt. No synthetic task accuracy fitting.
Gradient and parameter agreement is measured, not assumed from mathematical
identity. Set float32 unit roundoff u=2^-24, gamma96=96u/(1-96u), using the
largest96-term forecast projection as nominal accumulation length. Acceptance:
relative L2 gradient deviation<=4gamma96 at snapshots; denominator is max(norm
reference gradient,1e-12). Final parameter relative L2 deviation<=64*4gamma96.
Report absolute max deviation, per-parameter data and any failures. These are
prospective engineering tolerances, NOT rigorous bounds on a conditioned full
nonlinear/autodiff/AdamW trajectory, and not new accuracy equivalence criteria.
Old ME035B precision exception is unrelated and remains failed, unchanged.

## Serving engineering and original source handling

Replay all60frozen native models and compare complete standardized input B1
forward/backward costs only on the8registered cases; 5warmups/30latency samples,
including a separate measured CPU compile/setup cost. Original physical-input
probes and raw-unit exports are checked for all60. Serving latency never overlaps
with other local CPU-heavy work. Preserve every failed equality/performance test.
Full96-step replay remains necessary, state still128float32/channel. No microsecond
streaming, controller/VoI, new accuracy, full FLOP/energy or architectural-superiority
claim. Numerical/compiler speed does not establish useful predictive information.

Freeze original dependency/source/checkpoint hashes and all results. Real-data
sources and train-only normalization are EXACTLY ME038, authenticated originals;
see ME038 DATA.md, sources.json, data.json, split edges/licenses/known caveats.
No new CSV acquisition, reserved cohort, future covariate or label exposure.
Upload original research model/evidence bytes as DRAFT repository-owned assets,
verify API SHA/size then retrieve/replay the compact bundle. This is artifact
availability, not model promotion or public deployment. Refresh README/research
navigation, claims/evidence/master plan and exact delivery links; no merge.
