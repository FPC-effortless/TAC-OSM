# SEQ-TRANSFER-007 — symbolic intermediate-state supervision ablation

**Status: registered exploratory validation-only study, no held-out claims.**
Uses original SEQ006 training/validation events, not DEV068 or any
TEST-seed dataset. Run independent of frozen SEQ006C confirmation.

## Question
Is the 100% L64/L128 validation accuracy of scratch entity-local models
enabled primarily by public identity routing and final-outcome supervision,
or does it require the additional *exact intermediate entity-state teacher*?

## Factorial
Three scratch backends:
- TAC-inspired gated entity-local recurrence (not pretrained TAC-OSM).
- Shared entity-routed GRU.
- Global GRU baseline.

Train each with:
- FINAL_ONLY: final binary outcome BCE, auxiliary state-head weight 0.
- DENSE_TEACHER: the exact original SEQ006 loss,
  final outcome BCE +0.25 intermediate-state BCE.

All arms use the SAME original training source 690601 (3,000 sequences L32/64,
4 entities), validation 690602 L64/E4 and 690612 L128/E4 (600 examples
each), optimizer AdamW at LR0.003, batch48, 400 updates, no selection.
Pilot seeds 690810 and 690811 are new and distinct from SEQ006/006C.
12 complete training fits; retain each failure, output distribution, seeds,
final loss, and checkpoint hash. No best model substitution.

## Explicit interpretation
A strong FINAL_ONLY result would mean the exact state teacher is not
essential in these synthetic tasks, but labels still come from a
deterministic symbolic simulator, public entity IDs remain supplied,
and the model has no real-world belief distribution or independent
maintenance cost. This is NOT unsupervised/posterior-free world
discovery, because final outcome labels are supervised.

If dense training dominates, quantify precisely which task and
horizon; do not conflate rule learning, information retention, and
relational XOR. Model parameter counts are identical for each
same-architecture teacher/no-teacher comparison. Comparing across
architectures remains capacity-confounded.

No long-held-out L256/512 or E6 data may be opened in this exploratory
ablation. Any promotion requires fresh source-frozen seeds/data,
>=20 training seeds, stronger controls and a preregistered gate.

## Success decision
This study is descriptive, not confirmatory. Report paired seed accuracy
differences on all six validation task/length cells and teacher consumption
at training and inference. Keep a negative result as evidence.
