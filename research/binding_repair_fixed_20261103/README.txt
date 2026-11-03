Approved single repair-learning check — ONE EXACT RELEASE

The source defect is repaired and no-fit tests pass. Existing weights still
produce the original failed answers because an adjoint repair changes learning,
not stored weights. This proposal measures whether the repair helps acquisition.
The old one-off synthetic fitting exception was consumed by run 41623750060.
It does not transfer. The new direct owner response "Proceed" authorizes only
the exact context control/repair pair documented below.

Exactly two arms in the same job: original context and repaired context,
99,971 active parameters each. Seed 18001, 128 optimizer updates per arm, 256
total. Both initial parameter hashes must equal the original context initializer
BEFORE any optimizer construction. Identical data, case schedule,
full 256-byte objective, AdamW LR .003/betas .9,.95/weight decay .1/epsilon 1e-8,
and clip norm 1. No learning-rate, architecture, gate-bias or epsilon search.

Checkpoints 0,32,64,112,128; source/reference/causal tests before fit, final
source/reference check, full optimizer/RNG artifacts and bitwise serialization
replay. First-character and complete-answer success separately require all eight
FIT cases at BOTH late checkpoints. The 16 reused synthetic DEV cases are already
exposed; no independent-generalization claim. No early-success selection or
failure-only extension. The primary control is the unchanged context model
trained alongside the repair with alternating arm order on identical batches.
Historical failures remain preserved. One seed cannot establish generalization
or architecture superiority. No new phase or GRU fit is proposed.

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

REQUEST_TEMPLATE.json remains a disabled example. RUN_REQUEST.json records the
new direct owner approval and releases the exact frozen job once. Scientific
scope, source, code and resource-wrapper manifests are unchanged. Administrative
approval/protocol/review hashes are refreshed. No new training had executed at
release. Preserve all failures and stop after this allocation; no retry.
