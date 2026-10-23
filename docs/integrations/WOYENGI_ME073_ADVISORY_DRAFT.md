# Draft: Woyengi consumption of TAC trajectory forecasts

Research integration specification, no deployment or external execution.
TAC remains an observational predictor; Woyengi retains authoritative state,
authorization and execution. No verified SLA-risk label, calibrated customer
risk, actionable cost or intervention benefit exists in BPI2012 evidence.

## Read-only forecast request

Pass schema/model/source versions, dependency hashes, authorized entity key,
valid/recorded-as-of cutoffs, immutable observed event/resource/time prefix,
and permitted output horizons. Entity key routes state; it is not an embedded
feature. Reject stale/contradictory versions and unavailable-time facts.
Predicted events never enter the authoritative observed event stream.

## Forecast response

Return checkpoint identity and source provenance; per-horizon activity/EOR
probabilities, conditional time quantiles and representative imagined paths;
finite-particle count, evidence/exposure tier and abstention reasons. A source
record boundary is not a completed or stalled customer workflow. Include
conditional survival support and clear time units.

The model can estimate `P(delay to next recorded event >24h | event exists)`.
This is **not** `P(SLA breach)` and omits cases with no subsequently observed
event. A threshold such as>.80 is an illustrative unvalidated review trigger,
not an approved operating point. Forecast reliability and actual SLA/censoring
labels must be qualified on authorized independent real cases first.

## Advisory handoff and eventual controlled routing

A forecast may produce a read-only review proposal with explicit probability,
uncertainty, case evidence, model hash, expiry and suggested operator check.
It cannot reroute, approve, cancel, mutate facts or issue a customer commitment.
If future authorized policy enables automatic rerouting, Woyengi must verify
real SLA semantics, calibration/support, current dependencies, action permission,
idempotency and audit trail; execute outside TAC; then record the actual action
and independently verified outcome. That future automation is not implemented
or authorized by this draft. No notification or API call is sent here.

A pilot would prospectively measure false reviews, missed delays, lead time,
operator effort and actual benefits/costs. Internal simulated fixtures may test
interface behavior only. No funding, revenue, deployment or live-client claim.
