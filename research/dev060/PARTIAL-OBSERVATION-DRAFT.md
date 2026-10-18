# DEV-060 draft: public-information competence and privileged teaching

Status: FULL LEARNED-POLICY STUDY DRAFT, NOT STARTED. Small two-entity
world/reference foundations are implemented under `protocol-foundations.json`;
see `docs/DEV-060-FOUNDATIONS.md`. These are not learned-policy results. Freeze implementation, models, tuning/training
budgets and source before main training. DEV-059 remains an exact-public-history
study. Keep MTSK novelty retirement in force.

Question: can public-only learned policies maintain calibrated hidden-state
beliefs and acquire information cost-effectively? Does private-state imitation
help, or omit the inspections a public student needs?

## References and students

- Privileged teacher: sees current hidden fault state, chooses safe progress.
  It has extra information and is not automatically an achievable student target.
- Public belief reference: sees exactly the student's records/actions/receipts,
  tracks a finite-state posterior and explicitly evaluates inspection, repair
  and stop. Name its finite planning depth; prove optimality separately before
  calling it a global oracle or a pure information-theoretic regret floor.
- Students: compare privileged-policy imitation with public-reference imitation,
  using identical architecture, public training episodes and sample/compute
  budgets. Include both full learned and conventional GRU controllers, ten
  initialization seeds each. No final-test selection of favorable seeds.
- External simulator truth scores physical repair and valid completion. A
  learned completion head does not certify external success.

## Sensor/structure specification to freeze

Initial status missing independently with probability .5. Emitted initial
statuses and repair receipts have known bit-error probability .1. Public
relations remain exact. Paid inspection yields an exact status. Receipt schema
must explicitly identify the previous action kind/provenance so a student can
distinguish inspection from a noisy repair response; no hidden-state fields.

Use random per-world continuous eight-dimensional opaque identity vectors,
consistently supplied in records, queries and candidate descriptors, instead of
eight memorized one-hot labels. The learned path may not use a public-ID-to-slot
dictionary. Train N=2/4/6 with disjoint pairs. Separately test N=8, longer disclosed
chains, longer histories and .95-to-.7 repair drift. Keep sensor uncertainty and
each OOD shift distinct. Capacity overflow is a measured limit, not OOD success.

Declare generic memory capacity (e.g.64slots), parameter/state budgets, raw
attention capacity, full scans and acquisition charges before training. This is
not a C5 indexed-retrieval or equal-bits claim.

## Information mismatch check

Preserve a fixture with identical public histories but hidden states
[faulty,healthy] and [healthy,faulty]. Privileged teachers choose different first
repairs. A deterministic public-only student cannot match both. DEV-059 already
has a regression test constructing this counterexample. A paid inspection can
resolve the ambiguity, but a private teacher that never inspects supplies no
direct imitation examples of that acquisition decision.

Report student-minus-public-reference regret and public-reference-minus-private-
teacher gap separately. The latter also includes approximation error in a
bounded public planner; it is not automatically a pure information gap.
Report observation/inspection cost, repairs, residual faults, premature stop,
missing stop, physical completion and calibrated belief/outcome scores. Logical
write charges remain separate from measured CPU or physical memory traffic.

## Gates

- Ten fresh training initializations, disjoint world splits, frozen validation
  tuning, all learning curves, failed worlds, checkpoint/source hashes.
- Exactly matched public payloads and action availability; private labels only
  in training/evaluation. Standalone/import audits and hidden-state non-access
  tests remain mandatory.
- Retrained conventional and information-matched public controls, not only a
  private oracle or broken versions of one fitted model.
- Reliability noninferiority before accepting lower cost, separately for each
  sensor/structure/OOD family. Report planner depth, compute and stronger oracle
  checks wherever feasible.
- Veritas remains shared-synthetic-physics execution verification. Industrial
  transfer, original CDL distillation, general symbolic CASM discovery and
  learned scalable indexed acquisition require separate evidence.
