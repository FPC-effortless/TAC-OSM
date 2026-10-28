# Binding offline customer inference contract (2026-10-28)

Status: PROPOSED implementation contract; supersedes conflicting *prospective customer deployment* guidance once reviewed. Historical experiment records and prior decisions remain unchanged.

## Non-negotiable deployment boundary
Every company-built customer model in System 1, Enterprise, Temporal, Physical, Edge-Lite and Counterfactual must perform the entire inference path on the user-controlled device without connectivity: input preprocessing, feature encoding/embeddings, forward execution, recurrent state updates, uncertainty/calibration, typed output, abstention and any speech processing. No remote inference endpoint, remote embeddings, remote state update, or hidden network fallback. Network failures must not change the ability to infer from available local observations. Cloud training, CI, source tooling and signed model distribution are permitted; none may be required for inference.

## Initial product candidate
Evaluate the existing real-data Model01 energy-forecasting checkpoint for a simple offline prediction demo. Separately evaluate a frozen TAC-OSM checkpoint with explicit recurrent state input/output. Do not treat Model01 as TAC superiority evidence.

Proposed physical targets (access unverified): Samsung Galaxy A15 4G (Android), iPhone 13 (iOS), Raspberry Pi 5 8GB (embedded). Record exact OS, CPU/accelerator, memory and runtime before qualification.

Local boundary: authorized timestamped numeric observations, masks and provenance -> local preprocessing and persistent model state -> typed prediction, uncertainty, abstention and version metadata. No externally controlled actions are authorized by a prediction.

## Export and device qualification
For each checkpoint capture source commit and checkpoint hash, supported/unsupported operators, dynamic-shape and state serialization behavior, runtime/backend, numerical agreement with a preregistered tolerance, and fallback behavior. Measure on physical hardware: cold/warm p50/p95 latency, peak RAM, energy with measurement method, offline airplane-mode operation, restart/state continuity and no-network inference trace. Every missing measurement is NOT RUN or BLOCKED, never PASS. CI and hashes prove artifact integrity, not physical-device scientific qualification.

## Research integrity
Maintain six families and preserve all historical variants, results, negative outcomes and checkpoints. Prospective TAC neural experiments use 95k–105k active trained/deployed parameters; comparator counts within 1%, equal real data/splits, matched training and tuning compute and registered evaluation. CPU wall time is measured, not required to match. Synthetic tasks are diagnostics only and cannot support real-world claims. The 500k language studies are preserved as separately registered exploratory work, not a general parameter-rule change.

## Release gate
Do not call a model offline-qualified until complete on-device inference, numeric comparison and physical hardware airplane-mode, latency, memory and energy tests are recorded. Veritas qualification is independent; Woyengi may supply authorized observations and consume advisory typed outputs but remains authoritative for actions. Forge/TrainLab may use cloud infrastructure for data, training and evidence; no hosted per-inference business model.

## Execution checklist
- [ ] Reconcile prospective deployment language in master plan and company strategy via reviewed changes
- [ ] Inventory immutable candidate checkpoint and reference implementation
- [ ] Run export/operator and numeric agreement tests
- [ ] Implement minimal local SDK and no-network test
- [ ] Execute physical-device matrix, record failures and raw measurements
- [ ] Publish developer evidence receipt and independent Veritas qualification
