# DEV-031: bounded mixed-device continuation of DEV-029

Registered before training outcomes. DEV-030's all-CUDA rule failed and remains
failed. Its timing-only evidence supports a new explicit placement: GRU768 on
CUDA, other five arms on CPU. This is a separate runtime cohort, not a silent
replacement of earlier CPU experiments or a rescue of the failed global rule.
All arms use the same Kaggle image/torch build; CPU/CUDA numerics can change
long training trajectories and placement must accompany every reported result.

[Registration](../research/dev031/protocol.json): one DEV-029 tuning seed311,
LR0.002, fixed H512/B2, all six arms to 648 updates, saving 216 and 648. No
optimizer, examples, labels, architecture, metric or selection criterion changes.
Three CPU subprocesses plus one GPU subprocess overlap independent trials;
CPU dispatch order is frozen from engineering timing, not validation. A fixed
30-minute wall limit retains failures/partial checkpoints without promoting them.

Each checkpoint is reloaded in a separate process on its recorded runtime.
The complete validation records, predictions, selected actions and physical
costs must match strictly. Replay failure remains a failure; this stage does
not relax numeric criteria to claim success. Source, checkpoint and protocol
hashes are checked. This is repeat-execution integrity, not an independent
full-matrix scientific audit.

This is only six of 72 tuning trial slots and two of three registered checkpoint
budgets. The complete matrix, independent audit, validation-only recipe freeze,
untouched test and twenty-seed reliability remain pending. No recipe selector
or test evaluator is invoked. The original all-role selection metric and legacy
cancellation metric must continue to be reported separately. No abstract
architecture superiority, incapacity of unpartitioned memory, or complete-system
capability follows from this bounded tranche.

The private Kaggle notebook is source-pinned, preserves logs and model/optimizer
checkpoints outside the git checkout, and contains no launch credentials. It
allocates a GPU for the fast GRU path while CPU workers execute the other arms.
Kaggle quota measures allocation duration, including intervals where that GPU
is idle. This tranche is capped; it does not automatically launch the full grid.
