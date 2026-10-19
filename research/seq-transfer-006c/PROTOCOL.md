# SEQ-TRANSFER-006C — source-frozen 20-seed long-sequence confirmation

**Registration status:** PRE-TEST; source-frozen pilot predecessor in draft PR232.
Do not equate the scratch TAC-inspired recurrent agent with pretrained DEV061.5.
New source branch is stacked on the original six-pilot branch; no prior weights
or reserved real-maintenance data are modified.

## Independent research hypothesis
Does entity-local **learned state updating** transfer beyond the training
lengths of 32/64 events to hundreds of events and from four entity IDs to
six? In the original two-seed validation pilot, TAC-inspired gated and
simple routed GRU achieved 100% on late overwrite, repeated toggle, pair XOR
at L64/E4 and L128/E4; global GRU was near chance. These preliminary results
only motivate, not substitute for, independent held-out confirmation.

## Fixed model/training contract
Use unmodified `scripts/run_seq_transfer_006.py` training generator, architectures,
joint intermediate symbolic-state teacher loss, and 400-update AdamW recipe
from the parent research branch. The three models are scratch TAC-inspired
gated entity-local recurrence, independently trained routed GRU, and global
GRU. There is no pretrained model and no symbolic state machine inside learned
inference.

**20 NEW training initialization seeds per arm:** 690701..690720, no pilot
seed reuse. Every arm sees the same locked 3,000 training examples, and
600 validation examples each at L64/E4 and L128/E4, with same optimizer
settings, training schedule, per-sample routing and task definitions.
No best-seed or epoch selection. Save all 60 final checkpoints, failed
cells, weights/hashes, final losses and time.

## Locked held-out source and strata
The completely reserved TEST namespaces are:
- L256/E4: 690603, 600 examples.
- L512/E4: 690613, 600 examples.
- L512/E6: 690623, 600 examples.

Exactly 200 examples/task/stratum, labels balanced 100 positive/100 negative
per task. These datasets are constructed using the immutable original
SEQ006 event generator and the same operation vocabulary, never the
completed DEV068 or REAL-MAINT datasets. Each queried target has an
information age >=L/3 after its last informative update.
The public identity mapping is randomly permuted per example.

TEST generation is denied until **all 60 fresh fits** finish, their complete
training manifest and checkpoint hashes are saved, then a permanent
TEST-EXPOSURE.json containing the study identity/source SHA and locked
test namespaces is written. If fitting fails, do NOT generate TEST.
Test outcomes cannot change model choice, loss weights, seed set or budget.

## Registered gates
1. The exact symbolic event-state reference must reconstruct all held-out
   targets, event times and label balance without inconsistencies.
2. For each model independently, at least 18 of 20 fresh training seeds
   must achieve >=95% correct final predictions in **EACH of three tasks
   in EACH of three held-out strata**. Report each seed's individual nine
   cells, macro mean and any failures. Report exact-task confidence intervals
   clustered by held-out example, not pseudo-independent model seeds.
3. CASM-/TAC-specific superiority requires paired statistically justified
   improvement over the **independently retrained routed GRU**. Ties do not
   count, and model parameter counts/CPU must be disclosed. Stronger Mamba,
   full-context Transformer and fully pretrained DEV061.5 attribution remain
   separate research, not established by this experiment.
4. Positive synthetic result ≠ industrial maintenance economy, autonomous
   state discovery, or free-form relational reasoning: the event vocabulary,
   public IDs, relation query types, deterministic simulator, and intermediate
   symbolic-state supervision remain given.

Any failed gate closes this registration. After exposure, changes require a
new study ID and fresh, genuinely independent test data. No broad claims
based on best seeds, split rewrites or post-hoc favorable strata.

## Implementation and reproducibility
`python scripts/confirm_seq_transfer_006c.py --output results/seq006c`
Training 60 fits can use GitHub Actions CPU. No CI test refitting after
original archive is published; independent replay should consume the
original recorded checkpoint hashes and fixed source. The confirmatory
runner must preserve a strict pre-exposure condition and append a recorded
failure if any stage cannot complete.
