# DEV068 learned agent card

Status: validation rejected. Four exported models use first-declared fresh
initialization96801, not best seeds. They run but are NOT promoted/qualified.

Inputs: public typed events46fields, two public entity vectors, shuffled5action
descriptors, sensor error/inspection price and remaining10-action budget fraction.
Outputs: neural action scores,12paired-measurement/graph probabilities,21future
measurement/terminal-verification probabilities and normalized physical-cost
predictions. These are unqualified estimates; no neural output verifies success.

Learned: supervised admission gate, recurrent history encoding, neural attention/
CASM for TAC, factual state and action-consequence models, Q-based action/stop
selection. Primary arms never train against analytical posterior/action/transition
targets. `tac_teacher_belief` alone is an explicitly labeled teacher control.

Supplied: event/identity/action schemas, cosine binding, FIFO64retention, two
entities, probability-class meanings, cost/budget objective and verifier. Three
neural CASM operators run as a dense soft mixture. No claim of symbolic discovery,
learned indexing, arbitrary entity counts, C5scaling, efficiency advantage or
teacher-free ontology discovery. Physical simulation is independent code;
deployment never imports it. Every selected action comes from trained Q weights
with generic epsilon tie selection, without repair/stop/retry scripts.

Training data is synthetic. Extra exact inspection labels are paid and withheld
from the student's history, assuming non-invasive inspections/static intervening
state. Actual operation, pricing, sensor-law drift, richer action vocabularies,
resources, real maintenance endpoints and commercial economics remain unverified.

20fresh initializations per arm, plus2pilots: only3/20TAC,7/20GRU,5/20no-CASM
and8/20teacher controls meet the validation reliability gate. Reserved tests were
not opened. Do not use pooled means as evidence of robust training convergence.
False stopping, healthy exhaustion and collateral faults are recorded separately.

Portability: Python/PyTorch only, float64/thread1, eager deterministic mode.
Actual archived weights/source/public requests and all saved predictions replay;
cross-hardware claims require successful remote checks. Outputs are simulation
scores, not real prices. Independent execution authorization and physical outcome
verification remain required. No live equipment deployment is authorized here.
