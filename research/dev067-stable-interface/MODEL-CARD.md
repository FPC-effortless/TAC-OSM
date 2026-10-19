# DEV067 research artifacts

Intended use: offline, bounded synthetic maintenance research and checkpoint
reproduction. **No promoted or commercial-qualified model.** The scientific
primary is TAC/frozen/log; validation chose GRU/frozen/log before fresh fitting.
Neither passes the registered individual calibration-tail requirement. Do not
pick a different arm/seed from the test results and relabel it the candidate.

All six arms complete every required test trajectory across 20 fresh seeds with
zero final faults. The primary's calibration pass count is 16/20 and the candidate's
9/20, against18/20 required. CASM and frozen interfaces show no completion benefit.
Logarithmic coordinates increase cost in the registered factorial comparison.
Zero final faults does not mean no temporary faults or unnecessary repairs.

Portable artifacts export every arm at initialization96701, selected by protocol
before fitting rather than performance. This first primary checkpoint has p99 KL
.1291 and first GRU checkpoint.1230, both above.1. They are reproducible research
examples, not a quietly selected production release. All20seeds remain archived.

The JSON-lines runtime accepts two public16-vector identity keys,46-field public
events, five19-field action candidates and two public prices. It returns a chosen
candidate index, policy scores, joint12-class health/relation probabilities,
action-outcome probabilities and normalized cost scores. Caller maps the index
to its supplied action. Raw scores are not calibrated action probabilities or an
authorization. Outcome verification remains external to the model.

Runtime requires Python and CPU PyTorch; bundled model modules import no TAC-OSM
harness, simulator, Veritas or NumPy. It runs final EMA in float64/thread1 with the
unchanged deterministic candidate tie rule. Both sequential and optional batched
ingestion preserve the same gate/FIFO semantics. Replay tolerance is 1e-8 plus
exact action equality. Passing a finite set of hosts is not universal hardware
determinism. Event-clock invariance applies to the registered time-independent
simulation, not arbitrary rescaling of physical dynamics.

Training uses8192fresh synthetic public snapshots plus4096hash-verified prior
TRAINING-only student-visit snapshots, exact public posterior/future/cost/policy
targets,4096encoder and 4096 actor updates, and 50%exact-public belief mixing for
actor training. This is posterior-supervised imitation, not new online DAgger,
real repair experience or autonomous discovery. Four TAC heads share their final
encoder. Public identity binding, typed features, gate labels, memory capacity and
health/graph vocabulary are declared priors. The controller itself is neural;
it contains no Bayes filter, Bellman planner or handwritten repair/STOP rule.

CASM here denotes dense learned MLP mixing, not symbolic programs or skipped
operator compute. No scaling, multimodal, unseen industrial-site, profitable
maintenance or foundation-model claim is supported. Neither the ORNL source
inventory nor Malawi repair-source coverage constitutes a trained real model.
Actual cost and independent field-outcome qualification remains unmet.

Project metadata declares Apache-2.0; dataset licences and attribution are
separate. See PROTOCOL.md, RESULTS.md and REPRODUCE.md for exact scope/evidence.
