# TACOSM-PLM-PERSISTENCE-002: actual intervening memory writes

**Status: pre-registered; not yet measured.** This is a new scientific experiment
based on [issue #149](https://github.com/FPC-effortless/TAC-OSM/issues/149).
The previous `TACOSM-PLM-PERSISTENCE-001` result, its contract, and its
artifact remain unchanged. The frozen machine-readable specification is
`contracts/TACOSM-PLM-PERSISTENCE-002.json`.

## Motivation and hypothesis

PERSISTENCE-001 reported 100% accuracy with retained state and 51.41% with
reset at delay 4. Its delay steps, however, did not write state, and its exact
same-present/opposite-past control used a separately generated delay-zero
sample. This experiment tests whether the causal signal survives **actual**
intervening distractor writes in the **same paired evaluation population**.

At t0 a model receives the synthetic latent bit `m` through a small text,
image and audio observation. During each of `k` intervening steps it receives
independent distractor observations and updates its memory with a learned
`GRUCell`. At q2 it receives current bit `c` in a separate observation and
predicts `m XOR c`. The model sees no hidden label, future target, or oracle
state; all roles are identified by normal observable phase markers.

**Each evaluation pair** shares byte-identical later observations (including
all k distractor modalities and the current observation), but has `m=0` in
one past and `m=1` in the other. The correct q2 labels are opposite. A reset
control therefore produces identical predictions on each pair and must
score exactly 0.5. This exact paired ceiling is checked rather than assumed.

The model carries a 64-dimensional learned state. The GRUCell is called once
for t0 and **once per intervening distractor**, with hook-verified call counts
and per-update state-change magnitudes. This is not just iterating an identity
`retain(state)` function. The no-write diagnostic uses the same trained
weights and only the t0 state; it does not replace the main treatment.

## Frozen experimental constants

- Seeds: 0, 1, 2, 3, 4; delay k in {1, 2, 4, 8, 16}
- Training: 600 fixed steps/seed, batch size 64, cyclic delay schedule,
  AdamW lr 0.002, weight decay 0.0001, gradient clipping 1.0; no model selection
- Evaluation: 300 exact pairs (=600 q2 decisions) **per seed and delay**
- Separate train/eval RNG namespaces: 1,000,000 and 7,000,000
- Primary endpoint: q2 accuracy at k=4, averaged across the five seeds
- Registered mean criterion >= 0.90; worst seed >= 0.75; mean carry-minus-reset
  gap >= 0.30; seed-bootstrap 95% lower bound strictly >0; mean opposite-past
  prediction disagreement >= 0.90
- Exact reset pair accuracy = 0.5; oracle = 1.0, mandatory for each condition
- All observed state writes = 1+k; average intervening update magnitude >0.0001

No thresholds may be changed in response to a run. A negative result is a
valid scientific measurement, not grounds to change or remove an evaluation
arm. If an identity, gradient, leakage, provenance, write, or completeness gate
fails, the instrument must fail closed without a promoted capability result.

## Every evaluation arm

1. Persistent state: trained weights, t0 write and k real distractor writes
2. Reset: same weights and exact current input, but state zeroed just before q2
3. Swapped history: paired histories exchange final recurrent state
4. Oracle memory: evaluation-only `m XOR c`, not a model input
5. No intervening write: reuse the state after t0, bypass distractor writes

The audit must also report per-delay capability, pairwise opposite-past
prediction disagreement, state-write magnitude, observed call counts,
representation-call counts, fingerprints and the deterministic seed-bootstrap
gap interval. This is a count of state work, **not** evidence of sublinear
history retrieval or C5.

## Executable workflow

- `tests/test_persistence_002.py`: paired-collision, leakage, gradient,
  actual update, oracle and exact reset preflight
- `scripts/run_persistence_002.py`: contract-enforced frozen train/eval runner
- `scripts/validate_persistence_002.py`: independently recomputes registered
  arm coverage, integrity and success-criterion status from saved data
- `.github/workflows/persistence-002.yml`: runs the gates first, then
  measurement on its own branch head, uploads an immutable experiment artifact

An integrity/CI success does not prove the scientific hypothesis. A positive
outcome only supports **bounded synthetic causal state retention through
intervening writes**. It does not establish semantic long-term memory,
real-world learning, general intelligence, or selective compute scaling.
