# ME082–ME084: complete native computational graph and gated development search

Owner direction 2026-10-24: trace input to output, improve the complete model and
pursue the required gate before stopping. This registration permits up to three
ordered, fully recorded capability candidates. It does not guarantee a pass,
permit hidden attempts, alter existing criteria or unlock protected data.

## Specific architecture and computational work

A single 104,539-parameter native model retains fast32/slow32/8×8 delta memory.
The input graph uses actual observed state, prior-known command, observed
velocity/acceleration, **change in the known command**, and **time since an
observed coordinate value changed**. The latter is a prefix-cummax computation;
it is not evidence of true sensor freshness or physical stasis. Features are
causal and derive only from the actual prefix, without future movement labels.

A shared encoder feeds two parallel-capable computational paths: native nonlinear
state evolution and two causal depthwise temporal filters (kernel5, dilations1/4).
A shared horizon-query decoder combines both paths. Batch/time/channel/horizon
axes use vectorized tensor operators. The nonlinear recurrence is still sequential;
there is no automatic conversion into an SSM or an associative native scan.
A separate pure tensor entry point excludes Python value-dependent validation,
and is exported/traced from inputs through every distribution/state output.

The quiet location is a learned bounded multiple of observed velocity. The moving
location combines observed velocity, acceleration and command increments under
learned gates. Absolute state/context can choose coefficients, but cannot alone
produce an arbitrary location correction. If all recent state/control increments
are zero, **both conditional locations equal the last observation for every set
of learned weights**. This is an observed-equilibrium modelling prior, NOT a
universal physical law: delayed/hidden actuation can still cause real movement.
Such failures remain in the actual-moving acceptance gate, never excluded.

A FIT-only 75th-percentile absolute quiet-increment statistic sets noise units,
with a physical 1e-5 floor. Quiet scales are learned; moving scales remain wider
and can express unexpected movement. The mixture median/quantiles derive from
the same CDF, vectorized over probability levels. The legacy distribution key
`current` now denotes the learned quiet **distribution anchor**, while raw
observed state remains separately available for real motion labels/references.
The output fields do not establish a joint trajectory distribution or a policy.

## Input-to-output proof obligations

- Trace real FIT input shapes, all tensor graph operators, dependencies and
  parameter coverage. Input validation remains outside the pure tensor graph.
- Independently compare vectorized label gathering/censor masks with the original
  source loader. Future outcomes are supervision only, never forecast features.
- Test finite nonzero parameter gradients, causal suffix isolation, chunk/step
  equivalence including event clocks and encoded filter history, and the
  zero-increment location invariant with arbitrary learned readout parameters.
- Count the actual whole model before every first optimizer update and verify
  all final checkpoints/2048 updates through the existing live guard.
- Parallelizability and mathematical correctness are engineering facts, not
  improved learned intelligence or hardware throughput by themselves.

## Registered search and unchanged gates

Fixed order: ME082 NLL weight .2, ME083 weight1, ME084 weight.02. All use the
same architecture and CRPS + NLL + .2 factual motion BCE; actual targets only.
Each registration has five cold starts, 2048 AdamW updates, batch24/history32,
final-only weights, no restarts/warm starts or hidden sweeps. Each candidate's
five originals must be Git-frozen before either exposed REPORT readout.

All **ME081 acceptance conditions remain unchanged**: H1>=5%better than ME080;
movingH1<=1.05*ME078; quietH1<=1.10*ME077; H4<=.95*ME079; H16<=1.05*ME080;
coverage>=85% and interval scores<=ME080 at each horizon; no taskH1>1.10*persistence;
>=4/5 H1seed wins; primary95%paired episode CIupper(new-ME080)<0; actual32frame
means beat persistence at each horizon and movingH1 beats clipped velocity.

ADDITIONAL stronger conditions: primary98.3333%paired episode CIupper<0 to
account for three registered candidate attempts; H1>=5%better than ME081; NLL
no worse than ME081 at each horizon; original-model independent strict replay.
No condition is weakened. All candidates, adverse cases and selection counts
are published. This is sequential tuning on already exposed development, so
mechanical gate PASS still does not provide independent population proof.

Stop the registered search at the first FULL pass, marking unused registrations
NOT_STARTED_GATE_PASSED. If all three fail, the family is exhausted: record that
truth rather than manufacture a pass or run an undeclared fourth trial. The
owner's desired outcome is a research goal, not authorization to falsify evidence.
No new fits under completed ME078–ME081 studies.

## Data, numeric contract and preservation

Same authenticated CC-BY-4.0BodenAI/RW-RL-HIL revision
9290f5a96dc91ea58256e81d9ce2dd9621890b51. 120FIT/40exposedREPORT episodes;
FIT-only normalization/noise. Same primary960eight-frame query origins and
904eligible32frame histories. No sourceCAL/TEST/protected tissue-box access.
Source takeover affects labels, not neural inputs. No identified causal control.

Training is float32. The prospective **qualified reference forward is float64**,
using the exact original trained float32 weight values promoted without learning.
CDF inversion/scoring are also float64. The unchanged raw contract remains
rtol1e-5/atol1e-6. This does not edit old study arithmetic or tolerance; FP32
accelerated inference is not silently qualified. Reference learned-weight payload
is twice float32, count unchanged. Do not advertise a microsecond/edge footprint
or fully optimized GPU kernel without actual measurements.

All original weights, optimizer/RNG, source/schedule/registration hashes, actual
update receipts and raw outputs are Git-owned. Independently restore and score
original models on clean checkout and external host. Save every failed attempt.
No architecture-superiority/equal-compute claim is permitted: this is one native
capability-development search, with all training/tuning consumption disclosed.

## Prior art and capability-to-value card

Reuse causal dilated convolutions (Bai et al.2018, arXiv1803.01271), mixture density
networks (Bishop1994), proper scores (Gneiting/Raftery2007, DOI10.1198/016214506000001437),
gated delta memory and conditional probabilistic forecasting. These principles
are established; no category-theory, biological or SSM novelty is invented here.
The unresolved integrated capability is accurate quiet AND moving physical future
state with sharp honest uncertainty and portable original replay.

Operator question: could an advisory model warn a human of actual movement while
avoiding stationary drift? Real source/outcomes yes; operator economics, independent
pilot, customer utility and commercial readiness UNKNOWN/NOT_MEASURED. This work
grants no autonomous action, deployment, merge or protected-data permission.
