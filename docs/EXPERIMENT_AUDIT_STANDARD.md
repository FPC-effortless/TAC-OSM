# TAC-OSM Experimental Audit Standard v0.1

## Purpose

Every TAC-OSM experiment must pass an independent methodological review before its result can support a scientific claim. The review is separate from implementation/CI review.

The required sequence is:

1. benchmark validity
2. leakage audit
3. protocol/preregistration audit
4. implementation-to-contract audit
5. measurement/instrument audit
6. statistical analysis audit
7. provenance/reproducibility audit
8. claim-boundary review

A green CI run is evidence of software integrity, not scientific validity.

## Disposition vocabulary

- **PASS** — no identified blocker in the reviewed scope.
- **CONDITIONAL** — usable only with explicitly documented restrictions.
- **BLOCKED** — a flaw can change the interpretation of the endpoint.
- **VOID** — measurement was run but cannot answer the registered question.
- **UNREVIEWED** — insufficient evidence has been inspected.

A result may not be upgraded from VOID/BLOCKED to scientific evidence merely because the code was fixed afterward. The corrected protocol requires a new confirmatory run unless the defect is demonstrated to be non-material.

## A. Benchmark validity gate

Review all of the following before running confirmatory measurement:

- What population does the benchmark represent?
- Is the unit of analysis explicit?
- Are train/validation/test or adaptation/evaluation boundaries explicit?
- Are duplicates, near-duplicates, shared latent instances, and correlated generated samples excluded across boundaries where relevant?
- Is the target/action uniquely defined?
- Can the benchmark succeed by construction without learning the tested capability?
- Is there an oracle/reference ceiling?
- Does the reference actually measure the capability claimed?
- Is difficulty changing with the scaling axis?
- Are distractor density and acceptable-action density controlled?
- Is the benchmark balanced across families/classes where the claim requires it?
- Are test instances frozen before model selection?
- Are hidden targets or test outcomes inaccessible to model selection?
- Does the metric remain meaningful when the reference fails?
- Is there a trivial/base-rate/lookup/index shortcut?

**Hard rule:** a benchmark that makes the primary endpoint attainable by construction is an instrument defect, not a positive result.

## B. Leakage gate

Audit information flow, not just dataset splits.

Check:

1. test/evaluation samples into training;
2. test/evaluation targets into feature construction;
3. future outcomes into current representations;
4. hidden wiring/structure into public descriptors;
5. oracle labels into routing or repair;
6. validation outcomes into representation proposal;
7. test outcomes into hyperparameter selection;
8. benchmark-generation code for target-dependent construction;
9. duplicate or near-duplicate instances across boundaries;
10. provenance metadata that accidentally identifies the target;
11. cached artifacts/checkpoints carrying evaluation information;
12. adaptive stopping or retry logic conditioned on held-out results;
13. leakage through candidate ordering/index construction;
14. leakage through normalization/calibration/feature selection fitted on evaluation data;
15. leakage through post-hoc pair or action selection.

For each suspected channel record: source, destination, whether information is available before the registered decision point, and disposition.

## C. Representation-specific leakage

For VRS and later PLM experiments, audit separately:

- representation proposal time;
- proposer inputs;
- prompt/model/version;
- anchor construction;
- calibration data;
- pair construction;
- action schedules;
- counterexample generation;
- verifier inputs;
- refinement inputs;
- hidden evaluation outcomes.

A representation may not be called frozen if any of its dimensions, anchors, scaling, feature selection, or version was changed after seeing confirmatory outcomes.

## D. Protocol integrity gate

Before measurement verify:

- machine-readable contract exists;
- contract hash/version is recorded;
- primary endpoint and threshold were registered before confirmatory measurement;
- all scaling levels and seeds are fixed;
- exclusions and stopping rules are fixed;
- amendments are explicit, dated, and preserve superseded definitions;
- no result-dependent amendment occurred;
- randomization/seeds are recorded;
- benchmark generator version is pinned;
- external dependency commits are pinned;
- environment/dependency versions are recorded;
- exact command/configuration is recoverable.

A contract-compliant run is not automatically scientifically valid.

## E. Implementation-to-contract audit

Independently compare:

contract -> benchmark generator -> model/router -> executor -> verifier -> metrics -> artifact.

For each primary endpoint identify the exact code path that computes it.

Required checks:

- the code measures the registered quantity, not a proxy;
- work counters count actual operations;
- capability and work are measured independently;
- exhaustive/reference and selective arms use equivalent task instances;
- reference failure is surfaced, never silently treated as success/base-rate;
- hidden information is not passed through convenience APIs;
- checkpoint loaded for evaluation is the trained checkpoint;
- integrity hash is recorded;
- no dead code or fallback path can satisfy the endpoint;
- failed gates terminate measurement rather than merely logging warnings.

## F. Instrument validity gate

Every new benchmark or metric must have adversarial tests for:

- all-zero/untrained model;
- random model;
- oracle model;
- constant-output model;
- always-first/always-last action;
- base-rate strategy;
- empty candidate set;
- duplicated candidates;
- shuffled candidates;
- permuted labels;
- corrupted state;
- hidden-field removal;
- exhaustive-reference failure;
- maximal-work and minimal-work cases.

A metric that gives a strong score to an invalid control is not ready for confirmatory use.

## G. Statistical analysis gate

Before viewing confirmatory results, register:

- estimand;
- unit of analysis;
- aggregation rule;
- seed handling;
- confidence/uncertainty method;
- paired/unpaired comparison;
- multiple-comparison policy;
- materiality threshold;
- missing/failed-run handling;
- exclusion policy.

Report raw per-seed values, not only pooled means.

For scaling studies distinguish:

- descriptive scaling over the tested range;
- fitted model assumptions;
- uncertainty;
- extrapolation.

Finite-range data must not be converted into asymptotic O(1), O(log H), or sublinear claims without a justified model and uncertainty analysis.

## H. Reproducibility/provenance gate

Every result artifact must identify:

- repository;
- branch;
- commit SHA;
- contract SHA/hash;
- benchmark/generator SHA;
- external dependency SHAs;
- environment;
- hardware;
- software versions;
- seeds;
- command/config;
- checkpoint hash;
- raw artifact hash.

The result artifact must be immutable after disposition. Corrections create a new artifact/version and preserve the original.

## I. Claim-boundary gate

Every result receives a sentence-level disposition:

- what was tested;
- what was observed;
- what mechanism is supported;
- what was not tested;
- what cannot be inferred.

The following are never inferred without direct evidence:

- benchmark success -> general intelligence;
- persistence -> semantic memory;
- retrieval reduction -> computational speedup;
- fixed execution work -> scaling;
- routing accuracy -> useful execution;
- representability -> learned optimization;
- contract compliance -> scientific validity;
- one synthetic benchmark -> universal PLM capability.

## J. Independent review rule

The author of an experiment must not be the sole scientific reviewer of its primary result.

The reviewer receives:

1. preregistration/contract;
2. benchmark generator;
3. experiment code;
4. raw results;
5. artifact/provenance manifest;
6. claimed interpretation.

The reviewer does not receive an expected conclusion as an input to the audit.

The reviewer may mark a result PASS, CONDITIONAL, BLOCKED, or VOID and must cite concrete evidence.

## K. Confirmatory-run rule

If a material defect is discovered before the confirmatory run, amend the contract and rerun from the corrected boundary.

If a material defect is discovered after measurement:

- preserve the original run;
- mark its affected result VOID or BLOCKED;
- record the exact defect;
- fix the instrument/protocol;
- run a new confirmatory measurement.

Do not retroactively relabel the old measurement as if it had used the corrected protocol.

## L. Required audit record

Each experiment must have:

- `AUDIT-<experiment-id>.md`
- benchmark audit
- leakage/data-flow audit
- protocol audit
- implementation audit
- instrument/adversarial-control audit
- statistical plan audit
- provenance audit
- final disposition
- unresolved limitations

This standard is methodological infrastructure. It does not itself establish any scientific claim.
