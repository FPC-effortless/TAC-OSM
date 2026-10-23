# Enterprise adapter boundary

ME069 introduces a configurable categorical-event adapter, elapsed-time native
state updates, next-event regime probabilities and a waiting-time distribution.
It is trained on BPI2012 loan-process records, not on Woyengi customer data.
The native core retains fast, slow and associative state; it is not replaced
with a Transformer, LLM or SSM. Standard process-prediction and probability
methods are reused with attribution in PROTOCOL.md.

## Prediction and authority

A caller supplies only authorized events already observed at its query time.
Activity/lifecycle, resource and event time use the frozen FIT vocabulary and
feature mapping. Unknown categories map to zero input embeddings. A caller
must retain its own stable entity key and authoritative event history; case IDs
never enter the learned representation. Do not merge belief states across
entities or revise a historical event from a forecast.

Forecasting reads belief state without updating it. Only assimilation of a
subsequent actual event updates that state. A forecast contains a next-event
distribution, a positive-wait probability and positive lognormal parameters.
The long-wait probability uses a FIT-derived threshold. It is **not** a timeout,
failure, contractual SLA or authorization probability. Raw interval coverage
must be reported; the intervals have no guaranteed calibration under shift.

Any future Woyengi integration should attach source/version, event-time and
observation-time cutoffs, model/checkpoint identity, belief version and
prediction horizon. It must mark unknown permissions, costs and outcomes as
unknown. Woyengi retains canonical state and action authorization. A TAC
mixture coefficient cannot authorize execution, certify evidence or replace a
verified outcome receipt. No runtime integration is claimed by this study.

## Evaluation scope

The registered experiment resets state at each bounded, at-most-32-event
history window. Its scores do not qualify uninterrupted lifetime state,
long-history retention, multi-step process simulation, causal action effects
or useful adaptive reasoning. Future experiments must measure those directly.
Resource labels are anonymized actors/resources, not independently verified
customer identity. The event timestamp is assumed available when recorded;
the source does not separately supply Woyengi-style observation/ack times.

Veritas can test the interface and replay behavior using labeled engineering
fixtures. Such simulated results cannot establish enterprise predictive value
or commercial readiness. No edits or deployment to Woyengi or Veritas are part
of this TAC-OSM research branch.
