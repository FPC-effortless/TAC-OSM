# TACOSM-C5-FULL-PHASE-001 — Measured Result

**Status: MEASURED — integration pass, capability claim bounded by benchmark scope**

## Run

- GitHub Actions run: 36829950787
- Artifact: TACOSM-C5-FULL-PHASE-001
- Artifact ID: 11147106908
- Artifact SHA-256: fd676ca8e2863bc4eafda417596d742c15c1548f2384b8d6cf0f83b60762ed40
- Repository commit measured: 8cf00124b0eb861f23deb8001cb45e7def502cd5
- Seeds: 0–4
- Training: 600 log-uniform-M steps over M=8..256
- Evaluation: 64 calibration + 64 held-out trials per seed/M
- M levels: 8, 16, 32, 64, 128, 256, 512, 1024
- Online loop: 96 trials/seed at M=128
- Repository gate: **800 tests passed**

## Architecture exercised

The run executes one integrated path:

```
persistent state
      ↓
CDL dense Q/K student
      ↓
OR-over-tables random-hyperplane LSH
      ↓
calibrated K shortlist
      ↓
exact CASM equality circuit
      ↓
independent relation verifier
      ↓
verified experience: write
      ↓
execution-derived CDL update
      ↓
next trial
```

The router receives the public state address and reads the corresponding world value. The target index and acceptable-action set are never passed to the router. World writes are delayed by one causal step; verified learning writes use the `experience:` namespace.

## Pooled headline results

| Endpoint | Result |
|---|---:|
| Target-rank scaling exponent `gamma` | 0.000 |
| Query-routing work exponent `beta` | 0.255 |
| Oracle persistent success, M=8 | 100.0% |
| Oracle reset success, M=8 | 14.06% |
| Max calibrated K90 | 1 |
| Mean online first-attempt success, M=128 | 81.04% |
| Mean online final success, M=128 | 81.04% |
| Mean verified experience-write rate, M=128 | 81.04% |
| Mean M=32 Top-1, analytic initialization | 100.0% |
| Mean M=32 Top-1, random initialization | 90.625% |

## Scaling result

The exhaustive router is effectively Top-1 perfect on this workload:

| M | Mean target rank | P90 rank | K90 | Held-out coverage @ K90 | Sparse admission recall | Sparse final success | Mean rerank/M |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1.000 | 1.0 | 1 | 100.0% | 100.0% | 100.0% | 30.0% |
| 16 | 1.000 | 1.0 | 1 | 100.0% | 95.31% | 95.31% | 17.5% |
| 32 | 1.000 | 1.0 | 1 | 100.0% | 94.69% | 94.69% | 11.2% |
| 64 | 1.000 | 1.0 | 1 | 100.0% | 90.94% | 90.94% | 5.66% |
| 128 | 1.003 | 1.0 | 1 | 99.69% | 90.31% | 90.31% | 3.50% |
| 256 | 1.000 | 1.0 | 1 | 100.0% | 87.50% | 87.50% | 2.22% |
| 512 | 1.009 | 1.0 | 1 | 99.38% | 85.63% | 85.63% | 1.31% |
| 1024 | 1.016 | 1.0 | 1 | 98.13% | 79.38% | 79.38% | 0.64% |

At M=1024 the 95%-calibrated K is 1.2 when averaged across seeds, reflecting one or more held-out rank-2 cases in the calibration streams; K90 remains 1.

The LSH geometry is stable across the tested range:

- p1 ≈ 0.931–0.960
- p2 ≈ 0.481–0.512
- rho ≈ 0.060–0.103
- requested LSH table count is 1–2
- the 128-table cap never binds

Query routing work grows with an observed log-log exponent of approximately 0.255. This is an empirical arithmetic-count result for this implementation and workload; it is not a theorem and is not a hardware latency claim.

## Selection versus admission

Conditional selection is 100% at every M in this phase.

Therefore:

[
P(mathrm{success})
=
P(mathrm{target admitted})
]

for the tested workload.

This cleanly isolates the observed sparse failure at larger M as an admission/LSH miss rather than a CASM or verifier failure.

CASM execution count remains bounded by the admitted shortlist. Because K90=1 for almost all registered M, mean executed-candidate count stays approximately one; this is a particularly strong bounded-computation result, but it follows from the unusually easy rank distribution.

## Persistence control

The oracle control gives:

- persistent state: 100.0% at M=8
- reset state: 14.06% at M=8

This establishes that the state-addressing task used in this phase contains information that the reset control does not have. It does **not** establish that a persistent memory mechanism improves the learned router, because the learned router in this phase is given the exact state value through the public address.

## Online closed-loop result

Across five seeds at M=128:

| Seed | Admission/final success | Verified experience writes |
|---:|---:|---:|
| 0 | 97.92% | 94 |
| 1 | 72.92% | 70 |
| 2 | 80.21% | 77 |
| 3 | 78.13% | 75 |
| 4 | 76.04% | 73 |
| **Mean** | **81.04%** | **77.8** |

The run performed approximately 59–77 verified online updates per seed after the 600-step dense-outcome training phase, ending with 659–677 router updates.

The online result is descriptive: it demonstrates that the complete execution→verification→experience→learning loop runs end-to-end, not that the online update is causally responsible for the observed performance.

## Analytic initialization

At M=32 over three diagnostic seeds:

- random initialization: Top-1 = 90.625%, mean rank = 1.09375
- analytic initialization: Top-1 = 100.0%, mean rank = 1.0

This supports the narrower hypothesis that analytic initialization can stabilize this specific student on this task. It does not establish the result on the original noisy-recovery benchmark.

## Main scientific limitation

This phase does **not** reproduce the earlier CDL precision bottleneck.

The reason is structural: the query contains a public state address, and the router reads the exact target descriptor from persistent state before ranking candidates. The ranking problem therefore becomes nearly:

[
	ext{exact state value} ightarrow 	ext{matching candidate descriptor}
]

rather than:

[
	ext{noisy/ambiguous observation} ightarrow 	ext{recover relevant candidate}.
]

That is why the target-rank exponent is `gamma=0` and K90 remains 1.

Consequently, the measured `beta=0.255` and the 79.38% sparse success at M=1024 demonstrate that the integrated implementation can perform bounded approximate admission plus exact execution, but they do **not** establish that OR-LSH solves the original CDL admission bottleneck.

## Research decision

The full-loop implementation is operational and the major failure planes are now instrumented separately.

The next scientifically meaningful phase must preserve the original noisy-recovery difficulty while keeping:

[
Mightarrow CDLightarrow Kightarrow CASM_Kightarrow Verifier
]

unchanged.

Specifically, the next benchmark should:

1. generate fresh noisy query observations independently of the hidden target;
2. prevent the CDL rank measurement from reading the exact target descriptor from persistent state;
3. retain persistent state only in the explicitly registered persistence/control arms;
4. measure target ranks under the noisy scorer before selecting an index;
5. rerun conformal K and OR-LSH p1/p2/rho under that harder reference distribution.

That correction is required before treating the C5 admission/scaling hypothesis as supported or falsified.
