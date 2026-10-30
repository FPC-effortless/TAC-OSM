# LANGACQUIRE097: teach acquisition and later use

This is an internal research experiment, not a capability result. It follows
LANGPERSIST096's negative lexical-reuse and general-capability findings without
changing that experiment's questions, outputs or checkpoints.

The intervention changes training loss placement, not TAC's inference graph.
Four genuine byte models have 100,521 (TAC), 100,661 (GRU), 100,404 (selective
SSM) and 100,484 (Transformer) active/deployed parameters. Each is trained from
scratch under full next-byte loss and query-target-only loss, with identical
initialization, chronological examples, batches and updates within a seed.
Query gradients pass through earlier acquisition. No trained Transformer is
borrowed from a different budget.

The frozen data contains 128 real-code FIT episodes and 24 ARC demonstration
tasks, plus 64 code DEV episodes and eight task-disjoint ARC DEV tasks. Code
files are split by complete original source owner. Actor IDs are opaque;
source identities and future answers belong to preparation/scoring records.
ARC is a human-authored abstract benchmark and a secondary mechanism diagnostic,
not primary real-world or general-intelligence evidence. Public DEV has prior
research exposure and is not an independently blind TEST.

There are 40 fixed fits: four families, five seeds and two objectives, each
with 256 updates at batch four and 272 dense compute slots. Padding occurs
only after a real episode ends and receives no loss. Authentic input bytes,
supervised targets and padding work are counted separately. The original
512-update draft was reduced before any real-input model operations because
constructed complete-operator profiling showed it exceeded the CPU job limit.

Execution has two stages. First, source-authenticated FIT-only profiles record
all ATen forward/backward operator signatures, shapes and counts with zero
optimizer steps. Their original receipts and the mathematical companion are
reviewed and pinned. `REGISTER.json` keeps fitting disabled until those gates
are complete. Then five CPU jobs each seal eight fits. The downstream evaluator
requires every original checkpoint before making any DEV predictions.

Carry/reset/donor/corruption test lexical state use. ARC also has a format-
matched shuffle of observed demonstration outputs; all eight shuffled controls
preserve acquisition length. Five unmatched-clock ARC donor controls remain
inconclusive. A fixed 1,928-byte raw-history Witten--Bell reference and a copy-
last-demonstration ARC reference provide practical quality hurdles. Predictions
are replayed before target scores are computed; answers never enter query actors.

Each family has a strict matched-objective resource manifest. Complete operator
ledgers establish the registered comparison within a family; absolute hardware
FLOPs remain unknown. Cross-family architectural superiority and a separate
compute-matched architecture benchmark remain unqualified. Full tensor/IEEE/AD
formal certification, autonomous verification/repair and intelligence claims
remain open.

The seven phone proposals expose the full upper bound: 15 CPU minutes for
preflight, five fit jobs of at most 40 minutes, and 40 evaluation minutes,
255 aggregate CPU-job minutes. No GPU, protected TEST, promotion or merge is
authorized by these receipts. Report-only commits run software checks without
repeating scientific fitting. Failures, all seeds and negative outcomes remain
part of the evidence.

See `PREREGISTRATION.md`, `REGISTER.json`, `FROZEN_INPUTS.json`, the four
`FAIR_COMPARISON_*.json` files and `FORMULATION.md` for the complete contracts.
