# TACOSM PLM P2-P4 Results

Authoritative execution used for the P2-P4 integrated run:
GitHub Actions run 36872710966, branch research/plm-unified-phase-001.

Repository tests: 834 passed.
PLM phase tests: 5 passed.
P2, P3 and P4 runners completed successfully and uploaded the three JSON
artifacts.

## P2 — state stability

Protocol:
- 5 seeds
- 256 transition attempts per cell
- false-acceptance rates 0.00, 0.01, 0.05, 0.10, 0.20, 1.00
- no-verifier control

The strict zero-false-acceptance arm produced:
- 128 verified writes per seed
- 0 wrong verified writes
- 0.0 wrong-write rate

The deterministic no-verifier control produced:
- 256 verified writes per seed
- 128 wrong verified writes
- 0.5 wrong-write rate

Across seeds, the injected false-acceptance dose response was:

| false-accept rate | mean verified writes | mean wrong verified writes | pooled wrong-write rate |
|---:|---:|---:|---:|
| 0.00 | 128.0 | 0.0 | 0.0000 |
| 0.01 | 128.8 | 0.8 | 0.0062 |
| 0.05 | 137.0 | 9.0 | 0.0657 |
| 0.10 | 142.2 | 14.2 | 0.0999 |
| 0.20 | 152.2 | 24.2 | 0.1590 |
| 1.00 | 256.0 | 128.0 | 0.5000 |

Interpretation: the state layer is mechanically verifier-sensitive, as intended.
The experiment does not estimate real-world verifier error rates and does not
establish long-horizon stability beyond the 256-step window.

## P3 — operator scaling

Protocol:
- E = 4, 16, 64, 256, 1024, 4096, 16384
- 5 seeds
- key dimension 16
- query noise 0.05
- admission cap 8
- index levels 4, 8, 12

Mean indexed cost proxy versus full scan:

| E | indexed cost | full scan | target admission |
|---:|---:|---:|---:|
| 4 | 15.0 | 4.0 | 1.00 |
| 16 | 15.0 | 16.0 | 1.00 |
| 64 | 15.8 | 64.0 | 1.00 |
| 256 | 20.2 | 256.0 | 0.80 |
| 1024 | 21.4 | 1024.0 | 1.00 |
| 4096 | 32.6 | 4096.0 | 0.80 |
| 16384 | 73.4 | 16384.0 | 0.80 |

Interpretation: within this exact synthetic address relation, indexed query work
beats the scan proxy from E=16 onward on the measured grid. The result is not
an asymptotic claim, and the 0.80 admission rate at E=16384 shows that lower
address work can coexist with lost target recall.

## P4 — continual plasticity

Protocol:
- train domains 0,1,2
- domain 3 held out
- 5 seeds
- 200 examples/domain
- 8-dimensional binary features

Mean results:

| arm | final retention | early adaptation error | held-out accuracy |
|---|---:|---:|---:|
| global_fast | 0.5367 | 0.4567 | 0.5150 |
| slow_shared_fast_specialist | 0.5477 | 0.4467 | 0.5290 |
| specialist_fast | 0.5640 | 0.4867 | 0.5940 |

The specialist-only arm has the highest measured held-out accuracy and mean
retention in this small synthetic protocol, while the combined slow/fast arm
has slightly lower early adaptation error than the other two.

This is mechanism evidence only. The task family is deliberately small and
the domain identifier is observed. No general continual-learning or AGI claim
is promoted.

## Research consequence

P2 establishes the verifier-state interface as a measurable stability axis.

P3 establishes computation addressing as a separate scaling axis and exposes
an empirical recall/cost tradeoff.

P4 provides a plausible but non-promoted plasticity signal.

P5 therefore combines the already isolated state and computation addressers
under independent M and E growth rather than re-running any of these phases.
