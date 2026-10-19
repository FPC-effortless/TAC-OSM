# DEV065 research model card

Status: NO PROMOTION; not commercially qualified. The primary isolated/mixed
recipe across20seeds has78.05%completion, zero primary final faults but20higher-noise fault
bits, and fails p95/p99belief calibration. Do not treat a healthy-probability
estimate as independent verification. See RESULTS.md and all seed tables.

Each exported arm uses the predeclared first seed96501, not a best seed. Weights
are final float64 EMA from2048updates, with raw/pilot/all-seed evidence retained.
All decisions, including STOP, originate in trained weights. The model has no
repair heuristic, Bayes filter, transition law or scripted action selector.

Scope: posterior-supervised synthetic two-entity/five-action tasks. Typed46-field
public observations, two public16-vector identity keys, FIFO64, supplied12-class
health×graph schema, known sensor-error/price inputs and an event-gap clock remain
structural priors. Learned admission has explicit source-type labels. Neural CASM
is a dense three-MLP mixture; CDL-style retrieval scores two bounded banks.
No symbolic discovery, autonomous physics learning, scalable C5 index, industrial
sensor transfer or actual repair economics is demonstrated.

Application modules import Torch and standard-library code. Portable tests block
TAC-OSM/Veritas/absolute NumPy imports after Torch startup; this is not a test of a
NumPy-uninstalled Python environment. runtime.py is the reference JSON-lines
adapter; runtime-batched.py is optional typed ingestion with the same weights.
See runtime-profile.json for the one-host CPU profile and audit.json for parity.
All raw records are still read; no DRAM/FLOPs/energy or production SLA claim.

No live equipment execution, action authorization, independent post-repair
verification or deployment acceptance is supplied by the learned completion head.
Commercial and generalization gates remain open. Code project metadata declares
Apache-2.0; training observations here are synthetic. External maintenance source
licences/records are separate and are not included in these weights.
