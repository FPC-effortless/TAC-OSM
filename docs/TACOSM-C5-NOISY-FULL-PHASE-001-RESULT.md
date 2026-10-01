# TACOSM-C5-NOISY-FULL-PHASE-001 — Measured Result

**Status: MEASURED**

## Run provenance

- GitHub Actions run: 36832999287
- Successful job: 110273686597
- Repository gate: **806 passed**
- Measured source commit: 85c79886a51c24ccc1625f79e6dd3ae484682d15
- Artifact: TACOSM-C5-NOISY-FULL-PHASE-001
- Artifact ID: 11148458464
- Artifact SHA-256: 6140e8be378103d74b598cfc1932d763b27379d56eec60bdc57195b4593b49ef
- Seeds: 0–4
- Training: 800 exhaustive outcome-supervised steps/seed
- Calibration: 64 trials/seed/M
- Held-out evaluation: 64 trials/seed/M
- Online loop: 96 trials/seed at M=128
- M: 64, 128, 256, 512, 1024
- Query corruption: exactly one flipped bit
- Candidate population: 64 semantic classes, equally repeated across M
- Codebook: 16-bit binary, minimum Hamming distance 3

## Information boundary

The main noisy arm uses:

```
noisy public query
      ↓
CDL dense Q/K
      ↓
OR-LSH admission
      ↓
calibrated K
      ↓
CASM execution
      ↓
environment action outcome
      ↓
verifier
      ↓
repair / next admitted candidate
      ↓
verified experience update
```

The main router receives no state address and no clean target descriptor.

CASM also receives only the noisy query and the selected candidate. The hidden valid-action set exists only inside the environment and is exposed as the action outcome after execution.

Persistent state is used only in the separately registered clean-target control.

## Scaling results

| M | CDL mean best-valid rank | CDL P90 rank | CDL Top-1 | K90 | K95 | Held-out coverage @ K90 | Sparse admission recall | Sparse final success | First attempt | Mean CASM executions | Mean repairs | Rerank/M |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1.278 | 2 | 74.69% | 2 | 2.4 | 97.5% | 70.00% | 70.00% | 64.06% | 1.33 | 0.06 | 7.62% |
| 128 | 1.556 | 3 | 74.69% | 3 | 3.8 | 97.5% | 73.44% | 73.44% | 69.38% | 1.52 | 0.08 | 5.86% |
| 256 | 2.113 | 5 | 74.69% | 5 | 6.6 | 97.5% | 65.94% | 65.94% | 63.44% | 2.18 | 0.10 | 4.49% |
| 512 | 3.225 | 9 | 74.69% | 9 | 12.2 | 97.5% | 67.50% | 67.50% | 65.31% | 3.07 | 0.18 | 3.56% |
| 1024 | 5.450 | 17 | 74.69% | 17 | 23.4 | 97.5% | 65.31% | 65.31% | 61.88% | 5.23 | 0.55 | 2.72% |

The fitted P90 dense-CDL rank exponent is:

[
gamma = 0.776.
]

The fitted sparse query-routing arithmetic exponent is:

[
eta = 0.465.
]

No LSH table cap bound at any registered M.

## Baseline separation

The public-query Hamming reference is exactly correct across the registered M levels:

- mean best-valid rank = 1.0
- Top-1 valid = 100%

Dense noisy CDL is materially weaker:

- Top-1 valid = 74.69% across all registered M
- mean best-valid rank increases from 1.278 at M=64 to 5.450 at M=1024
- P90 rank increases from 2 to 17
- K90 increases from 2 to 17

This identifies the learned representation as a genuine bottleneck under the anti-leakage benchmark. The perfect Hamming reference also shows that the synthetic task remains algorithmically recoverable from the noisy observation; the problem is not information-theoretic impossibility.

## LSH admission

Measured collision geometry:

- p1 ≈ 0.855–0.871
- p2 ≈ 0.489–0.501
- rho ≈ 0.193–0.223
- requested tables: approximately 3–5
- table cap never binds

The sparse funnel reranks a shrinking fraction of M:

- 7.62% at M=64
- 5.86% at M=128
- 4.49% at M=256
- 3.56% at M=512
- 2.72% at M=1024

But admission recall remains only 65–73%. Therefore the LSH stage does reduce the candidate work substantially while also becoming a significant source of false negatives.

## CASM and verifier/repair

Conditional on the admitted set being sufficient, CASM and verification can recover the correct action.

The clearest evidence is the gap between first-attempt success and final verified success:

| M | First attempt | Final success | Gain from repair |
|---:|---:|---:|---:|
| 64 | 64.06% | 70.00% | +5.94 pp |
| 128 | 69.38% | 73.44% | +4.06 pp |
| 256 | 63.44% | 65.94% | +2.50 pp |
| 512 | 65.31% | 67.50% | +2.19 pp |
| 1024 | 61.88% | 65.31% | +3.44 pp |

This is the first phase in which the verifier/repair path is doing observable work rather than merely passing a deterministic execution.

The mean number of CASM executions rises with M because calibrated K rises and the first ranked candidate is not always accepted. At M=1024, the system executes 5.23 candidates on average under K90=17.

## Persistence control

At M=128:

- clean persistent-state control = **100%**
- reset control = **0%**

This confirms the control can recover the clean class when the clean value is intentionally stored behind an opaque state address, while the reset arm has no access to that information.

The control does not imply that persistent state improves the learned noisy CDL arm; the main arm is deliberately forbidden from reading the clean target.

## Online verified-learning loop

Across five seeds at M=128:

- first-attempt success = **53.96%**
- final verified success = **58.96%**
- admission recall = **58.96%**
- mean executions = **1.70**
- mean repairs = **0.79**

Verified updates are applied only after a successful environment action and successful verification. The online result is descriptive rather than causal.

## Scientific interpretation

This phase changes the state of the C5 research question.

### Supported observations

**1. The clean-target leak is removed.**  
The main router operates solely from the noisy public query and candidate descriptors.

**2. The noisy CDL rank distribution is nontrivial and degrades with population size.**  
P90 valid rank rises 2 → 17 from M=64 → 1024, producing a fitted `gamma=0.776`.

**3. Calibrated K tracks that degradation.**  
K90 rises 2 → 17 rather than remaining constant.

**4. Sparse admission reduces reranking work but loses valid candidates.**  
Rerank fraction falls to 2.72% at M=1024, while admission recall is 65.31%.

**5. CASM + verifier + repair recovers a measurable portion of first-attempt failures.**

### Not supported

The phase does **not** support a claim that CDL+OR-LSH currently preserves high capability as M scales. The sparse final success is 65.31% at M=1024 on this workload.

It also does not support a general sublinear end-to-end complexity claim. `beta=0.465` is only an empirical arithmetic routing fit for this synthetic workload, and the result excludes index build cost from the exponent.

### Bottleneck localization

The failure decomposition is now:

```
noisy observation
     ↓
CDL representation   ← primary learned bottleneck
     ↓
OR-LSH admission      ← secondary bottleneck
     ↓
K-limited CASM
     ↓
verifier / repair     ← recovers some first-attempt errors
     ↓
verified outcome
```

The Hamming reference remaining at 100% is important: it means the next improvement should be directed at **learning a robust relevance representation**, not at adding more recurrence, verifier complexity, or memory machinery to compensate for a weak signal.

## Required next phase

The next experiment should therefore hold the learned CDL architecture fixed and change the **representation objective**, not the execution substrate.

The primary comparison should be:

```
current noisy CDL objective
vs.
multi-view positive-noise objective
vs.
higher-negative-coverage objective
vs.
cosine-normalized objective
```

with:

- identical noisy task stream
- identical candidate populations
- identical CASM execution
- identical verifier/repair path
- identical M scaling
- identical calibration split
- identical seeds

The decision endpoint should be the change in **P90 best-valid rank and K90 at M=1024**, followed by sparse admission recall under the unchanged LSH layer.

The purpose is to determine whether the current 0.776 rank-growth exponent is a representation-learning limitation that can be reduced before further investment in the LSH/index layer.
