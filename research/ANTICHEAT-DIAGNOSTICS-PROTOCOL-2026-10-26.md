# TAC-OSM anti-cheat and failure-localization protocol

**Effective: 2026-10-26. Status: implemented engineering diagnostic tools and
physical-robot adapter; per-variant end-to-end pipeline adoption remains OPEN.**
No training, checkpoint modification, model superiority claim or replacement of
original negative results is authorized by this change.

## Threat model

A neural network can appear to learn without learning the intended dynamics if
its data or computation carries an unintended shortcut: future labels,
post-action rewards, IDs encoding outcomes, image stripes, sample order,
batch layout, padding, global normalizer statistics, teacher forcing,
evaluation oracle state, cached latent state, or an incorrect pre/post
observation boundary. Conversely, a weak score can be a dead gradient, missing
trained checkpoint, wrong gold label, state-reset error, target mismatch, or
a genuine model capability failure.

The existing source contracts, router-leakage checks and checkpoint integrity
gates remain necessary. This protocol adds an independent fail-closed
diagnostics layer around them.

### Gate status semantics

- PASS: the named diagnostic ran and its specific invariant held.
- FAIL: a concrete violation was detected. No clean integrity claim.
- INCONCLUSIVE: absent, invalid, nondeterministic or unsupported test; do NOT
  convert this into PASS.

AuditLedger requires named checks and throws on FAIL or INCONCLUSIVE.
All reports explicitly say engineering_only_not_capability_evidence.
No general-purpose test can prove the absence of every possible hidden
information channel. Source authenticity, capability and scientific fairness
require additional independent evidence.

## Executable components

| Surface | API | What it can detect |
|---|---|---|
| Schema and nested objects | audit_input_boundary | Unexpected root features, nested future/label/oracle keys, opaque objects |
| Source target contamination | probe_future_invariance | Future-only target poisoning changes any predecision model input |
| Checkpoint inference channels | probe_prediction_invariance | Changing only forbidden material changes deterministic predictions |
| Partition membership | audit_partitions | Exact source ID duplicates or FIT/DEV/TEST intersections |
| Fitted preprocessing | audit_train_only_fit | Transform fitted using a row outside registered FIT |
| Longitudinal split | audit_forward_time | Entity time reversal and insufficient embargo |
| Learning graph | torch_gradient_diagnostics | Disconnected, zero and nonfinite parameter gradients by module |
| Runtime data path | trace_torch_modules | Per-layer output shape, maximum magnitude and NaN/Inf origin |
| Hidden graph influence | trace_torch_noninterference | Fixed-checkpoint, paired source poison: first divergent leaf module or untraced functional path |
| Actual physical causal interface | audit_robot_windows | Every RealLoggedWindow input allowlist, chronological order and future mutation |

The hash comparisons are content fingerprints of values passed into the
diagnostic, **not proof of publisher provenance**. Tensor payloads are opaque
to key scans: poison every permitted tensor producer where it could encode
denied information, and compare actual intermediate outputs.

## Run commands

    python -m pytest -q tests/test_research_anticheat.py tests/test_robot_anticheat.py
    python scripts/run_robot_anticheat.py --fixture

The --fixture smoke check uses constructed records ONLY, with zero capability
evidence. The physical adapter can receive true original, source-verified
frames directly:

    from tac_osm.real_robot_transition_contract import real_windows_from_original_episode
    from tac_osm.robot_anticheat import audit_robot_windows
    windows = real_windows_from_original_episode(table, episode_index, history=256)
    ledger = audit_robot_windows(windows, run_id="ME-XXX-source-gate")
    ledger.assert_clean()
    report = ledger.report()

Only a source-authorized FIT/DEV original episode is appropriate for
development diagnosis. No repeated sealed TEST queries for tuning. The
separate CLI also accepts a JSONL export of genuine RealLoggedWindow records:

    python scripts/run_robot_anticheat.py --windows-jsonl windows.jsonl --expected-sha256 PINNED_DIGEST --run-id SOURCE-AUDIT --output report.json

Pin the file hash before evaluation. A matching supplied hash **does not**
authenticate the publisher; retain independent publisher proof, licence,
original file digest, episode IDs and source versions.

### Required source-to-decision wiring

Every future TAC-OSM training or evaluation runner must explicitly instrument
and preregister these checkpoints:

1. **Data and provenance:** original real source hash, licence, publisher,
   sealed label policy, source group and time, FIT/DEV/TEST assignment,
   duplicate/near-duplicate and group-boundary audit, observed action and
   reward availability. Only real outcomes establish primary capability.
2. **Source-to-feature:** an allowlist at the ACTUAL encoder invocation,
   fit-on-FIT normalization, data availability timestamps and masks;
   independent future-label poisoning to check joins and cached state.
   Passing the RealLoggedWindow producer test does not certify every
   downstream tensor consumer.
3. **Graph and learning:** instrument observation, action, fast/slow memory,
   associative/structured memory, retention, CED/decoder, planner and
   verifier boundaries as applicable. Check value provenance, state resets,
   module gradients, per-head label denominators, activations, finite state,
   optimizer updates and checkpoint hashes. Make predictions BEFORE feedback.
4. **Shortcut falsification on FIT/DEV only:** compare fully blinded IDs,
   ID-only, shuffled-label, shuffled action, reverse-time, memory reset,
   nuisance-feature-only and null-input controls. Register expected
   responses before opening results. High score in a denied-channel-only
   control is a shortcut warning requiring investigation, not evidence of
   intentional cheating.
5. **Scientific qualification:** independently replay checkpoint/loading,
   genuine chronological data, matching sample order, 100k size band,
   matched parameters/training budgets, heldout test sealing,
   preregistered baselines, subgroup and calibration scores. Preserve failed
   gates and report unknowns. No synthetic fixture may count as capability.

### Failure localization guide

| Observed evidence | Diagnosis to investigate |
|---|---|
| A future-only edit changes inputs or deterministic predictions | Leaked future or answer-bearing path; inspect first changed producer/module |
| Normalizer fitted on DEV/TEST or source ID overlaps | Data contamination |
| ID-only/metadata-only control predicts targets | Shortcut or hidden source correlation |
| Expected head gradient absent, optimizer steps recorded | Disconnected computation graph or wrong loss wiring |
| First nonfinite activation or nonfinite gradient block | Numerical or loss-scaling failure |
| Checkpoint hash still initial/unloaded | Training/load/evaluation harness failure |
| Memory reset does not alter required DEV outcomes | Memory unneeded, unused or overwritten; test retention and delayed effects |
| Correct local fit but poor out-of-site/out-of-time performance | Domain shift, source assumptions or generalization failure |
| Prediction high but action results weak | Proxy objective vs real action outcome mismatch; no causal claim |
| Required test missing or nondeterministic | INCONCLUSIVE: cannot issue integrity clearance |

### Shared program across six variants

System 1, Enterprise, Temporal, Physical, Edge-Lite and Counterfactual all
inherit the same evidence vocabulary and source-to-output gate. The source
tensor graph, legal information edges and time cut must be registered per
variant. System 1 and Edge-Lite additionally audit online cache/latency;
Enterprise audits entity/site joins; Temporal audits retention; Physical
audits executed vs candidate actions and human takeover timing;
Counterfactual audits logged support/propensities and never invents alternate
outcomes. PNDS / Lean mathematical noninterference is a distinct proof goal:
operator proofs do not automatically verify arbitrary actual PyTorch source
and training data paths.

## Limitations / OPEN

Current code checks the real-robot causal window adapter, general predecision
schemas, paired target poison, split/normalizer IDs and optional model hooks.
It has NOT yet enforced every runtime tensor producer/consumer edge for all
six variants, proved arbitrary-code information-flow isolation, established
the truth of producer-provided IDs, or trained and qualified a model.
Every uninstrumented boundary must remain UNKNOWN/INCONCLUSIVE in claims.
Never infer "not cheating" from a passed fixture suite.
