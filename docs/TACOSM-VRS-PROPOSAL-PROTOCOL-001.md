# VRS-001 Proposal Protocol

The proposer is an external LLM process. The TAC-OSM runner never calls an LLM
and cannot adapt a proposal after seeing benchmark outcomes.

The proposer receives the frozen domain description plus the public state/task
descriptions needed to materialize its representation coordinates. It must not
receive action outcomes, post-action states, target identities, truth tables,
verifier outputs, or benchmark results. It returns one frozen JSON artifact
containing proposer/model provenance, prompt/domain hashes, feature definitions,
calibration anchors, complete scored state rows and task rows, and declared
construction cost.

Provenance field `outcome_visibility` must equal `none`. Proposal generation
precedes confirmatory evaluation; the complete artifact is hashed before any
outcome is inspected.

Unregistered answer-like feature fields are rejected. Numeric filtering is
fixed by the VRS contract. The checked-in `proposal_smoke.json` is an
expression fixture for software tests only and is rejected from confirmatory use.

Dynamic cases and action schedules are generated before proposal scoring.
The representation may classify fixed pairs as near, but cannot construct the
pair set. Zero near pairs or zero checked transitions is instrument-invalid.

No smoke output is scientific evidence.


## Independent calibration set

Calibration is population-independent and is not sampled from any evaluation population.
The frozen runner defines 15 calibration states `cal:0` through `cal:14`. Each
feature must declare exactly three anchors named `low`, `mid`, and `high`, with
values 0.0, 0.5, and 1.0 respectively. Anchor state IDs must belong to the frozen
calibration set. In scored mode the proposal must include complete scored rows for
all calibration states in `scored_calibration_states`, separately from evaluation
state coverage. This prevents calibration from becoming an outcome- or
population-dependent operation.


## A4 input-visibility and neighbor-metric amendment

The scored artifact interface is now explicit: state/task values needed to
materialize representation coordinates are public inputs, while all outcome,
target, verifier, and benchmark information is forbidden during proposal
generation. The dynamic neighbor criterion uses normalized Euclidean distance,
computed as Euclidean distance divided by the square root of retained
representation dimension, with the preregistered threshold 0.35.

The one-step action space is finite (four actions) and is therefore tested
exhaustively. Multi-step checks use the representation-independent seeded
action schedules already fixed by the contract. This amendment predates all
confirmatory VRS measurement.
