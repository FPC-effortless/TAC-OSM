
# TAC-OSM Unified Research Run-Gates V2

Status: standing repository-wide protocol for every historical, active, corrective, proposed, and future research lane.

## 1. Non-negotiable rule

A run has two independent eligibility questions:

1. Did the run execute the registered experiment under a secure, uncontaminated
   protocol?
2. Does the resulting evidence support the registered scientific question?

Passing the first does not imply the second.

A failed gate is fail-closed:
- do not emit a publishable measurement;
- record the failed gate;
- retain the failed artifact;
- do not reinterpret the failure as a negative scientific result;
- create a corrected successor experiment only if the protocol is repaired.

## 2. Gate order

Every confirmatory run follows exactly:

G0 Evidence/necessity
G1 Contract/provenance
G2 Security/supply-chain
G3 Benchmark integrity
G4 Leakage/supervision boundary
G5 Representability
G6 Model-state integrity
G7 Degeneracy and discrimination
G8 Oracle/control validity
G9 Baseline/ablation integrity
G10 Statistical readiness
RUN
P0 Post-run provenance and artifact integrity
P1 Post-run leakage audit
P2 Post-run benchmark/invariant audit
P3 Independent metric recomputation
P4 Statistical decision rule
P5 Ablation completeness
P6 Failure/anomaly classification
P7 Scientific disposition and claim ledger update

No lower gate may be skipped because a previous experiment passed it.

## 3. G0 — Evidence and necessity

Before any compute:
- query the evidence register;
- name the exact new claim;
- list inherited evidence and exclusions;
- identify the blocker the run is intended to resolve;
- prove the run is not a redundant re-measurement;
- record the intended evidence layer.

If no new L2/L3 evidence would be created, do not run it as a new experiment.

## 4. G1 — Contract and provenance

The machine-readable contract must be committed before confirmation.

Pin:
- experiment id;
- question;
- hypothesis;
- arms;
- intervention surfaces;
- benchmark version;
- generator commit/hash;
- history levels;
- seeds;
- training schedule;
- evaluation schedule;
- parameter/compute matching rule;
- primary and secondary endpoints;
- materiality threshold;
- decision-rule branches;
- interpretation order.

Never pin the expected outcome.

A changed definition is an amendment, not an overwrite.
A changed protocol requires a new experiment id unless the change is explicitly
amended before any confirmatory run.

## 5. G2 — Security and supply-chain

### Runtime isolation
- CI runners use minimum permissions; contents: read unless a job genuinely
  requires more.
- Network access is disabled during measurement whenever possible.
- External downloads are setup-stage only, with immutable hashes recorded.
- Confirmatory measurement must not depend on mutable remote content.

### Filesystem safety
- Write only to declared artifact/result directories.
- Reject path traversal, absolute paths outside the allowlist, unexpected
  symlinks, and undeclared executable outputs.
- Artifact names and experiment ids are validated.
- No overwrite of an existing immutable result artifact.

### Dependency safety
- Pin the Python/runtime/dependency environment.
- Record dependency lock hash.
- Pin every third-party research checkout to a commit.
- Do not run unreviewed generated installation scripts as part of confirmation.

### Secret safety
- Scan stdout, stderr, configs and artifacts for credentials/API-key patterns.
- Secrets are never valid experimental inputs.
- A detected secret causes the artifact to be quarantined and the run invalidated.

### Agent/tool safety
- Code produced by an automated coding agent is treated as untrusted input until
  reviewed by repository tests and diff inspection.
- Generated code may not silently alter the benchmark, evaluator, contract,
  seed schedule, or artifact destination.
- Measurement code cannot fetch hidden labels at runtime.

## 6. G3 — Benchmark integrity

Before training:
- oracle must reach the registered ceiling;
- gold must be the unique satisfier where uniqueness is claimed;
- candidate placement and shuffling occur before gold derivation;
- generator seeds are independent by stream;
- train/validation/test identities are disjoint;
- no duplicate or near-duplicate tasks cross splits;
- nuisance variables do not change truth;
- intended causal variables do change truth;
- candidate order cannot encode the answer;
- candidate count cannot encode the answer;
- benchmark evaluator cannot read private runtime state;
- exact generated outputs, not teacher-forced loss, are the primary capability
  metric;
- a known cheater must not obtain the claimed capability score.

For history benchmarks specifically:
- paired histories with identical current observations must be generated;
- hidden regime labels cannot be routed into the model;
- history-independent control families are mandatory;
- a current-input-only shortcut must be tested explicitly.

## 7. G4 — Leakage and supervision boundary

There are two boundaries and both must pass.

### Interface leakage

The router/state/executor may not see:
target, answer, outcome, reward, gold, gold_index, true edge, oracle action,
oracle mask, correct action, labels, solution, expected action, future state,
future evidence, or metadata that uniquely encodes them.

The existing TAC-OSM forbidden-field audit remains mandatory.

### Supervision leakage

Also record whether a training signal is merely an interface-safe form of
the answer.

For every intervention document:
- what data can affect gradients;
- what data can affect rewards;
- what labels are available during training;
- whether the same logical information is available to a deployed system;
- whether the teacher can be evaluated on test information;
- whether the intervention makes the benchmark easier rather than the model
  better.

A gold-anchored training signal must never be described as gold-free.

## 8. G5 — Representability

Use the actual feature/state map.

The gate must:
- use one shared weight/state parameterization;
- cover every registered task family;
- run on the real generated episodes;
- test the exact relation under study;
- fail if the relation is in the null space or if margins collapse.

A per-episode oracle vector is diagnostic only.

No learned-arm negative result is interpretable if this gate fails.

## 9. G6 — Model-state integrity

For every learned checkpoint:
- hash the parameters;
- prove the checkpoint is not the initialization;
- record number of updates;
- record configuration hash;
- record training provenance;
- load the exact checkpoint used for evaluation.

For non-learned controls, record explicit "no learned parameters" rather than
leaving the field absent.

Checkpoint identity includes architecture and parameter width.

## 10. G7 — Degeneracy and discrimination

Before any capability endpoint:
- output spread must exceed the registered floor;
- both required classes must be supported;
- satisfying and violating examples must be discriminated;
- no constant-output or constant-action shortcut is allowed;
- programme consistency must hold;
- verifier outputs must be compatible with environment success semantics.

The existing fixed order remains:
representability -> degeneracy -> class support -> discrimination ->
programme consistency.

## 11. G8 — Oracle and control validity

The experiment must include controls sufficient to distinguish:
- broken benchmark;
- broken representation;
- broken optimizer;
- broken routing;
- broken execution;
- broken verification;
- broken state update.

Oracle is a ceiling/control, not a competitor.

A positive oracle with a negative learned arm does not imply the learned
mechanism is causal unless all preceding gates passed.

## 12. G9 — Baseline and ablation integrity

Every confirmatory experiment must identify:
- frozen baseline;
- matched parameter budget;
- matched compute budget or explicitly measured compute difference;
- all registered arms;
- paired instances across arms where possible.

Do not add an arm after seeing results.

For a new mechanism:
- one-component removal;
- mechanism-specific intervention;
- matched-capacity control;
- matched-compute control;
- randomized/shuffled control;
- oracle diagnostic.

## 13. G10 — Statistical readiness

Before RUN:
- declare the primary unit;
- declare seed set;
- declare history levels;
- declare primary endpoint;
- declare materiality threshold;
- declare uncertainty procedure;
- declare multiple-comparison treatment if relevant;
- declare the decision rule;
- declare what each branch licenses and does not license.

The default statistical unit is lane-specific and must be frozen before confirmation. For small-seed research lanes, the seed-level estimate with paired evaluation tasks and bootstrap over seeds is the default unless a different valid method is preregistered.

No threshold may be selected after seeing the confirmatory result.

## 14. RUN — execution requirements

During the run:
- log every gate state;
- record independent RNG streams;
- record checkpoint hash;
- record model/config/benchmark hashes;
- record actual compute counters;
- record state-write and state-read events;
- record intervention identity;
- record all anomalies;
- do not change code/config/benchmark in place.

If a runtime failure changes the registered protocol, stop and classify the run
as invalid rather than silently continuing.

## 15. P0 — Post-run provenance and artifact integrity

After execution:
- recompute artifact hashes;
- verify checkpoint hash against evaluation load;
- verify contract fingerprint;
- verify benchmark/generator hash;
- verify dependency hash;
- verify exact seeds and levels;
- verify all registered arms ran;
- verify no artifact was overwritten.

A discrepancy creates an invalid artifact, not a corrected result.

## 16. P1 — Post-run leakage audit

Audit:
- actual serialized router inputs;
- actual state reads;
- actual evidence packets;
- actual gradients/reward inputs where available;
- verifier inputs;
- post-action state update inputs.

The audit is against observed runtime records, not only source code.

A post-run leakage finding invalidates the affected measurement even when the
source-level interface test passed.

## 17. P2 — Post-run benchmark and invariant audit

Re-check:
- unique satisfier;
- post-shuffle gold correctness;
- train/test independence;
- candidate-order invariance;
- history-pair invariance;
- verifier/success agreement;
- active-node trace bounds;
- state-address correctness;
- output range;
- discrimination;
- class support;
- programme consistency.

For history experiments, also test:
- current-input-only baseline;
- current-input + short-window baseline;
- state-swap intervention;
- state-reset intervention.

## 18. P3 — Independent metric recomputation

Never trust only the training/runner summary.

Recompute primary endpoints from raw per-seed/per-task records in a separate
analysis path.

Verify:
- denominators;
- class balance;
- missing cells;
- NaNs/infinities;
- averaging order;
- paired alignment;
- work-accounting totals;
- no hidden filtering.

A metric that was not independently recomputable is diagnostic, not primary.

## 19. P4 — Statistical decision

Apply the preregistered decision rule exactly.

Report:
- point estimate;
- seed values;
- confidence interval;
- baseline;
- absolute and relative effect where registered;
- compute/capacity tradeoff;
- failures/invalid cells.

Do not convert a CI into a qualitative claim that was not preregistered.

## 20. P5 — Ablation completeness

Before promoting a result:
- all registered arms exist;
- every arm used the same protocol except the registered intervention;
- no arm has a hidden benchmark, optimizer, schedule or evaluator change;
- missing arms are explicit.

A result with incomplete ablation is marked partial, not promoted.

## 21. P6 — Failure and anomaly classification

Use mutually exclusive primary failure classes:
- protocol mismatch;
- security violation;
- leakage;
- benchmark invalid;
- representability failure;
- model-state failure;
- degenerate model/output;
- statistical underpowering;
- implementation defect;
- scientific null;
- scientific positive.

Implementation and benchmark defects are not scientific negatives.

## 22. P7 — Scientific disposition

Only after P0-P6:
- assign evidence layer;
- update CLAIMS.md;
- update EVIDENCE_REGISTER.md;
- update EVIDENCE_MAP.md if the blocker moved;
- archive the artifact;
- write a concise decision record.

Every claim must retain its exclusions.

## 23. Special rules for any mechanism that uses memory, persistence, or adaptation

Do not use biological terminology as hidden target labels.

Do not build the correct latent regime directly into the state vector.

Do not give the model a handcrafted memory feature that is unavailable to the
control.

Do not compare MTSK against a parameter-starved control and call the result an
architecture advantage.

Do not count state storage as free while charging only terminal execution.

Do not call a successful current-input-only policy a memory result.

The decisive causal test is intervention on the persistent state under identical
current observations.

## 24. Invalid-result policy

Invalidated runs remain visible.

Required record:
- experiment id;
- run id;
- commit;
- artifact hash;
- failure gate;
- exact defect;
- why the number cannot support the registered claim;
- successor experiment if one exists.

The successor must not silently inherit the invalid number.

## 25. Required machine-readable result envelope

Every new confirmatory runner must emit at least:

experiment_id
run_id
contract_fingerprint
repository
commit
benchmark_version
generator_hash
dependency_lock_hash
configuration_hash
checkpoint_hash
seed
gate_results
primary_endpoint
secondary_endpoints
work_accounting
intervention
anomalies
artifact_hash

Status is assigned by scientific disposition, not by the measurement
instrument itself.

## 26. Anti-drift requirement

Any future research script must be either:
- registered with a machine-readable contract and wired to this gate protocol;
- explicitly exempted with a named reason.

A new measurement script with neither is a test failure.

The same rule applies to every research track.


## 27. Portfolio continuity

Before a new lane is implemented, consult docs/RESEARCH_GOVERNANCE.md and docs/RESEARCH_LANE_REGISTRY.md. New ideas may add or branch research, but may not silently close unfinished lanes. Historical results remain visible and carry an explicit legacy-audit state until checked against the current governance.
