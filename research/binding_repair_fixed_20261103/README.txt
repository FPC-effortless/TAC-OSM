Proposed single repair-learning check — NOT RELEASED

The source defect is repaired and no-fit tests pass. Existing weights still
produce the original failed answers because an adjoint repair changes learning,
not stored weights. This proposal measures whether the repair helps acquisition.
The old one-off synthetic fitting exception was consumed by run 41623750060.
It does not authorize this new fit; the owner approval record is explicitly false.

Exactly two repaired models: phase and context, 99,971 active parameters each.
Seed 18001, 128 optimizer updates per model, 256 total. Exact old initializer
parameter hashes must match BEFORE optimization. Identical data, case schedule,
full 256-byte objective, AdamW LR .003/betas .9,.95/weight decay .1/epsilon 1e-8,
and clip norm 1. No learning-rate, architecture, gate-bias or epsilon search.

Checkpoints 0,32,64,112,128; source/reference/causal tests before fit, final
source/reference check, full optimizer/RNG artifacts and bitwise serialization
replay. First-character and complete-answer success separately require all eight
FIT cases at BOTH late checkpoints. The 16 reused synthetic DEV cases are already
exposed; no independent-generalization claim. No early-success selection or
failure-only extension. Compare with the preserved old phase/context results;
no new GRU fit and no architecture-superiority claim.

Caps: 360 aggregate CPU seconds, 600 wall seconds, with the same unchanged five
reviewed wrapper sources and 345-second child CPU hard limit. Keep all failures
and partial update counters. No retry, varying stage, GPU, paid compute, protected
TEST, independent qualification or merge. Full-model formal/IEEE/AD completion
remains OPEN; scalar theorem checks do not establish capability.

Source model repair: 41af8c423eceeb2bb4f1d95ae391ee807779f603
Independent repaired reference: 7b637554765267b81e41ee38fe9f060767d2a19d
Kernel diagnosis: https://github.com/FPC-effortless/TAC-OSM/actions/runs/41628953759
FP32/FP64 reference audit: https://github.com/FPC-effortless/TAC-OSM/actions/runs/41630918959
No-fit checks are technical evidence, not a new synthetic-fit exception.

REQUEST_TEMPLATE.json is disabled. There is deliberately no RUN_REQUEST.json.
Only a fresh direct owner approval of this exact scope can enable the scoped
admission record, refresh administrative hashes, and release the frozen job.
Code and scientific scope must remain unchanged when doing so. No new training
has been executed by this proposal or its source/bootstrap checks.
