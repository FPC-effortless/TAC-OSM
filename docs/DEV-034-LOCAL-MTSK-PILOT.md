# DEV-034: banked MTSK factorial engineering pilot

Registered before pilot outcomes. This implements the preceding proposal's
locality/retention candidate, not full-system Test6 or a proven fusion benefit.
Original SwiLAMTSK, VerifiedMTSK, masked-local TAC and schema41 code is unchanged.

The new banked adaptation has eight public entity banks and eight rank-three
experts per bank. The persistent matrix has576 scalars and the persistent
responsibility prior192:768 total,3072 bytes in FP32. Expert count/rank differ
from the original six-expert MTSK. Input and query width12 and head width32
stay independent of memory rank. The unused inherited read-key parameters remain
frozen and are separately counted in total versus trainable parameters.

The four corners cross global/local **writing** with tied/independent retention
parameters. All four retain supplied entity-indexed reads and learned bank
embeddings. Consequently the global corner is a new global-writer banked control,
not native MTSK or an unstructured retrieval baseline. Locality switches which
banks receive the entire proposed transition, including responsibility updates
and decay. Tied retention shares decay, input-decay and beta functions across
experts; varying responsibilities can still yield differing effective timescales.
Independent experts start from3 fast,3 medium and2 slow priors. These names do
not establish semantic retention. Input-conditioned decay modifies the base priors;
the initialized effective retention is not simply0.70/0.92/0.99.

Both local/global arms of each retention configuration have identical initial
parameters. Common event encoder, query encoder, prediction head and read adapter
are byte-identical across the factorial arms. Tied versus independent gates have
different parameter counts, explicitly reported. State equality is not parameter,
FLOP or compute equality: global writing updates eight banks per event; local
writing updates one. Report actual timing and complete search cost before any
efficiency claim. Baselines are the existing masked-local TAC and learned schema41.
Original native MTSK remains an untested external reference in this pilot.

Boolean commit rejection masks the complete augmented state; an all-rejected batch
skips proposal execution. Explicit commits reject non-finite proposals. Ordinary
observation updates instead raise on a non-finite proposal, preserving a numerical
failure rather than silently repairing training. This kernel-level admission API
is not PNDS four-gate enforcement or a trained integrated verifier.

Ten integrity tests cover complete-state budget,503 unrelated events, mixed
batch addresses, rejection, non-finite transitions, state-gradient isolation,
kernel relabeling, global interference, common heads and reset/read-only behavior.
Relabeling changes state/address/learned bank embeddings together; it is not a
claim of full-predictor permutation generalization or entity discovery.

One common limited recipe: seed311, initialization34311,216 AdamW updates,
LR0.002, weight decay0.0001, gradient clipping1.0 with non-finite rejection,
B2/H512. Loss averages pre-receipt and one seeded public inspection post-receipt
cross-entropy over eight entities and three action queries. No action teacher,
active sampling, global relational memory or temporal-channel module is added.

Namespaces are dev034-v1-train and dev034-v1-validation; the known synthetic
generator is reused adaptively. Validation uses H128/H512, three delays, four
roles, three deadline placements, two repetitions and two paired partners:
144 cases per history,96 mature primary and48 cancellation diagnostics.
These small validation sets cannot establish robust superiority. The primary
endpoint is all-role mature accuracy; cancellation is separate. Report a descriptive
locality-retention difference-in-differences, without seed confidence or synergy
claims from one initialization. Physical failures, services, unnecessary services,
inspections, absolute costs and selected-minus-model-only costs remain visible.
Zero failures through over-servicing is not useful predictive-memory success.

Smoke runs use2 updates/H32 and eight cases only. They are never promoted to
trained pilot results. Freeze a dedicated clean checkout before execution;
bound CPU concurrency to2 and preserve every failure/timeout. Require all six
arms before aggregate audit. The audit independently reconstructs the condition
census, split fingerprints, oracle labels, public receipts and physical costs;
it also performs exact separate-process replay of model logits and action records
using the same evaluator. Thus it is not an independent implementation of the
learned evaluator or an independent scientific replication.

The original DEV-029 full tuning remains incomplete. No recipe selection,
twenty-seed confirmation, scientific test access, generalization to new entities,
architectural superiority, calibrated verifier, or trained complete-system claim
is supported by this pilot. Scientific validation still requires adequately tuned
controls, fresh seeds and independently frozen larger evaluation.

## Execution checkpoint

Frozen source: `ec0ff80b058308dad8de57eab7d223ec9a81dcba`. The six-arm smoke
audit passed: [raw smoke audit](DEV-034-SMOKE-AUDIT.json). Actual smoke records
confirm identical initial parameter hashes within both local/global pairs.
All1161 local tests passed. The216-update pilot runs separately in a clean
detached checkout with two CPU processes. Source/model/protocol hashes and
all failures remain recorded. Training outcomes are not yet claimed.
