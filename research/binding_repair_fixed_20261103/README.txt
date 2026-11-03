Completed single repair-learning check — ALLOCATION CONSUMED

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

Completed outcome

Workflow 41643877841, attempt 1, execution b3b500c76203561375e966594b5701c52a0ae238:
128 updates per arm completed; all 1,992 implementation comparisons passed.
Both arms scored 1/8 first characters and 0/8 complete answers at BOTH updates
112 and 128. Both frozen FIT recall gates FAIL. On the 16 already exposed DEV
queries, each arm scored 2/16 first characters and 0/16 complete answers.
The original arm returned aeee for all eight FIT histories; the repair returned
aaaa for all eight. Final parameter hashes differ, and every AdamW state reached
128, so the repair changed training without rescuing conditional retrieval.

Captured experiment CPU was 173.301491521 seconds; wall before final receipt
was 176.083676252 seconds. The outer final write/exit tail is not included in
that receipt; runtime provisioning is outside this process accounting.
All ten checkpoint byte hashes, optimizer/RNG serialization and source/replay
records are retained. No additional fits or retries were launched.
CLOSURE_RECEIPT.json records consumption. RUN_REQUEST.json is the historical
single-use grant, not a new or transferable allocation. Whole-model proof,
independent generalization and an effective attention replacement remain OPEN.

Task-interpretation correction

This was eight one-record examples with the SAME key aa: delayed copying.
It did not test choosing among multiple stored facts or varying query keys.
Even a perfect FIT pass alone would not establish in-context recall, because
fixed mappings could be memorized in the weights. No binding claim rests on
this screen. TASK_INTERPRETATION.json records the correction; all original
bytes, scores, failure records, checkpoints and allocation closure stay intact.
Earlier capacity, absence-rejection and speed percentages: earlier experiments;
not re-verified; different configurations. Unexecuted drafts are proposals,
not empirical evidence. No additional run or update is released here.
