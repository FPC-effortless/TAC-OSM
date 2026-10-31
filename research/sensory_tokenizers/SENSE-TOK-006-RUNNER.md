# SENSE-TOK-006 training worker continuation

Status: implemented worker/source path; **zero fits, no real corpus qualification**.
The actual positive training path remains unexecuted until external gates pass.
Original S004/S005 and all six comparison candidate graphs are unchanged.

## Implemented path

`src/tac_osm/adaptive_token_training.py` verifies FIT/DEV file bytes against
frozen hashes and sizes, rejects duplicate IDs/content, escaped paths and source
groups crossing splits. It reads TEST metadata only: no TEST path resolution,
stat or byte access. Exact/near-duplicate corpus qualification beyond file
identity remains an external audit. Segments never cross source files; incomplete
tails are excluded and counted explicitly, not padded or silently scored.

Batches use independent deterministic (seed, step) schedules shared across arms.
The sample-order digest is computed before fitting and checked against the
canonical registration. Inputs are BOS plus preceding bytes. Random-control
uniforms are independent of targets and generated after sample selection.

The worker implements the proposed AdamW/warmup/cosine objective, finite-loss
and gradient guards, predeclared DEV checkpoints and earliest-best selection.
Checkpoints include model, optimizer, step, configuration, RNG and sample-order
state. Worker receipts are explicitly UNQUALIFIED and do not invent actual
FLOP measurements. Exact trained checkpoint reload/replay remains untested.

`scripts/run_sense_tok_training.py` defaults to preflight only. `--execute`
requires the complete reviewed six-arm/five-seed comparison, canonical resource
and signature validation, a separately signed protected execution-bundle digest,
closed math/probe/custody/source/near-duplicate/prefit-profile receipts, protected
GPU enablement and exactly two observed T4 devices. Independent processes receive
alternating seed shards; both devices perform actual work. Workers inherit an
allowlisted environment without evaluator/signing/GitHub credentials. Actual
filesystem isolation is an externally audited requirement, not established by
this environment filtering.

## External execution-bundle contract

The exact bundle file SHA256 must equal the independently signed prefit
`protected_assets_sha256`. This runner uses that field as the protected bundle
commitment; the independent evaluator must explicitly adopt this convention.
The bundle has exactly three keys: `qualified_protocol`, `corpus_contract`,
`full_graph_math_and_probe_receipt`. Each value contains an absolute protected
`path` and `sha256`. These files must be outside actor write permissions.

The qualified protocol needs actual reviewed numeric optimizer/schedule/batch/
history settings, `checkpoint_steps`, calibrated `random_byte_probability`,
`training_authorized: true`, and `no_new_model_fits: false`. No existing draft is
changed to supply these approvals. The corpus contract has original `files`
records with source/group IDs, FIT/DEV/TEST split, path, size and SHA256. Its
canonical sorted compact JSON digest is the registered real source hash.

The independently protected receipt names this study and has PASS for
`full_graph_source_correspondence`, `actual_source_probe`, `sealed_test_custody`,
`source_qualification`, `full_flop_prefit_profile`, `near_duplicate_audit`.
Those statuses are trusted only through the external signature binding the
protected bundle; an agent-created receipt cannot qualify a run.

Execution remains under the source-controlled phone planner/protected GPU
supervisor. Independent actual resource collection and the canonical final
parity/checkpoint/outcome attestation remain necessary before qualification;
this worker does not replace `scripts/run_qualified_research.py`. Its supervised
execution integration, positive full-budget run and postfit replay are OPEN.

## Checks actually executed

Eight new no-fit cases cover source tampering, source-group and duplicate
contamination, path escape, withheld TEST access, batch/target/order replay,
schedule endpoints and refusal of the existing no-fit manifest before any
optimizer. They use constructed temporary source fixtures, not real training
or independent source authentication. Full-suite results and remote CI are
reported in draft PR #675. No TEST contents or GPU quota were used.
