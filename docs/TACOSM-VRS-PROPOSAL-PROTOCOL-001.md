# VRS-001 Proposal Protocol

The proposer is an external LLM process. The TAC-OSM runner never calls an LLM
and cannot adapt a proposal after seeing benchmark outcomes.

The proposer receives only the frozen domain description and public task
specification. It returns one frozen JSON artifact containing proposer/model
provenance, prompt/domain hashes, feature definitions, calibration anchors,
complete scored state rows and task rows, and declared construction cost.

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
