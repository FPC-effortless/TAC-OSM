# Woyengi ↔ TAC-OSM ↔ Veritas: governed predictive-state handoff (prototype v1)

**Status:** transport-neutral contract + tests in TAC-OSM; NOT a deployed Woyengi adapter, Veritas plugin, authenticated service, or validated predictor. Aligned with Woyengi `CONSTITUTION.md`, bitemporal `validAt/recordedAt` reconstruction and proposal-only writes. Woyengi retains authoritative state, security and execution, including any absorbed KoreFrame/SafeEdge functionality. Veritas remains a distinct offline training/verification environment. No model confidence ever confers authority.

## End-to-end sequence

1. **Woyengi authenticates and authorizes** the requesting principal for a limited entity/workspace and available action set. It reconstructs the permitted view at explicit `valid_at` (world-valid time) and `recorded_at` (knowledge cutoff), records reconstruction/permission receipt IDs and immutable state revision, excludes hidden/private facts and already-late events.
2. A **domain-specific, versioned feature adapter** maps authorized Woyengi state/history into exactly the model's expected tensor schema. ME-061/062 use 14 robot-state and 14 action channels; Woyengi's arbitrary claims, semantic entities and security attributes do not magically become robot vectors. If no validated mapping exists, return `NO_FEATURE_ADAPTER` and skip the model.
3. Send a **read-only forecast request** with last up to 65 real observed state rows, past *executed* actions, provenance, authority/takeover flags, and 1–5 **proposed** candidate action sequences (each 1–16 horizon steps). Past and future actions must remain distinct. Model state starts from versioned checkpoint and selected authorized history, not from an untrusted reused private memory blob.
4. TAC validates only **syntax, temporal cutoffs, shape and trace binding**. An authorization-receipt string is not a validated permission: the Woyengi service authenticates and verifies it out-of-band. TAC forecasts the candidates and emits means, positive uncertainty scales, checkpoint and replay commit, state revision, input fingerprint and mandatory `review_required=true`.
5. **Woyengi rechecks** identity, allowable actions, approval/quotas, time-to-live, current canonical state revision and policy before attempting any actual action. A changed state revision or permission invalidates predictions. The forecasting response is not a state write. Without a qualified calibration/cost gate, block autonomous action and present human review or informational forecasts only.
6. Woyengi records proposal and independent verification/approval, executes via its governed runtime only if authorized, then appends actual action, receipt and independently observed outcome as new records. A fresh observation can update TAC's bounded belief, but never edits canonical history.
7. For learning, export only licensed, minimized and authorized time-cut event/action/outcome/cost traces to Veritas under an explicit training contract. Veritas can simulate, replay and score experimental policies, but generated counterfactual success is *not* evidence that a different action would have worked in production. Store only replay-verifiable TAC model checkpoints and corresponding lawful dataset acquisition manifests per repository rule.

## Input contract: `tac.woyengi.forecast.v1`

All fields below are required for the current physical14 prototype. The example is **fabricated for interface testing, not real training data**.

```json
{
  "schema_version": "tac.woyengi.forecast.v1",
  "request_id": "r-demo-1",
  "trace_id": "trace-demo-1",
  "principal_id": "principal:example",
  "workspace_id": "workspace:example",
  "entity_id": "robot:example",
  "feature_schema_id": "physical14:v1",
  "model_id": "me062:untrained",
  "checkpoint_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "authorization_receipt_id": "woyengi:authorization:demo",
  "reconstruction_receipt_id": "woyengi:reconstruction:demo",
  "state_revision": "state-event-head:42",
  "valid_at": "2026-10-22T12:00:01Z",
  "recorded_at": "2026-10-22T12:00:02Z",
  "observed_history": [
    {
      "valid_at": "2026-10-22T12:00:00Z",
      "recorded_at": "2026-10-22T12:00:00Z",
      "provenance_ref": "event:1",
      "state": [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
      "executed_action": null,
      "human_takeover": false
    },
    {
      "valid_at": "2026-10-22T12:00:01Z",
      "recorded_at": "2026-10-22T12:00:02Z",
      "provenance_ref": "event:2",
      "state": [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
      "executed_action": [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
      "human_takeover": false
    }
  ],
  "candidate_scenarios": [
    {
      "scenario_id": "hold",
      "proposed_actions": [[0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    }
  ]
}
```

The validator `src/tac_osm/woyengi_forecast_contract.py` checks finite vectors, explicit timezones, strict valid-time ordering, no observation after either requested cutoff, expected dimension, distinct scenario IDs, horizon limits and a canonical SHA-256 input fingerprint. It does **not** verify the signature, principal scope or validity of a receipt; those are Woyengi responsibilities. A guessed receipt ID must never authorize reads or writes.

## Output contract

TAC is permitted to return only scenario-bound finite 14-channel `predicted_states`, positive `predicted_std`, and `review_required=true` until the forecast itself qualifies on authentic unseen real data. It returns `status=forecast_only_unqualified`, `authority=none`, `execute=false`, immutable source model/checkpoint/replay references, and the input fingerprint/revision. Arbitrary tool calls/approval instructions in model output are discarded rather than forwarded.

The initial uncertainty values are **not calibrated intervals**: physical uncertainty must be calibrated on independent, appropriately exchangeable episodes with a declared tolerance and model checkpoint. Until qualified, the forecasting API MUST NOT automatically choose the smallest model-predicted risk as an executable command.

## Cross-repository mapping

- **Woyengi:** TypeScript/HTTP or SDK adapter in its Platform/Compute boundary, with scoped bitemporal state reconstruction and two independent control checks (on input and before any execution). Integration code belongs in Woyengi after its architecture/production review; this TAC branch does not change the Woyengi repository.
- **TAC-OSM:** Pure model + validator, with model-state version, feature contract hash, training provenance and prediction-risk measurements. Predict without secrets or mutation authority.
- **Veritas:** Offline trainer and independent verifier receiving replay bundles and factual outcome receipts, not credentials or mutable authoritative state. Simulated tasks stay identified as simulated, even where native artifacts are executable.

## Hard failure outcomes

`DENIED` for Woyengi authorization failure; `STALE_REVISION` for mismatched authoritative state; `NO_FEATURE_ADAPTER` for missing domain encoding; `INSUFFICIENT_HISTORY` for unsupported inputs; `PREDICTION_UNQUALIFIED` for unknown calibration/risk; `OUT_OF_SUPPORT` for unsupported actions; `REPLAY_UNAVAILABLE` for missing model checkpoint bytes or legal reconstruction material. None of these is overridden by TAC's numerical trust gate.

**Causal limitation:** Requesting five candidate actions and comparing five model-produced trajectories is hypothetical model-based planning, not a verified counterfactual or policy improvement. Genuine action-value qualification requires logged or randomized action support, measured costs and externally verified outcomes.
