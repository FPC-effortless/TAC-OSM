# TACOSM-C5-ADMISSION-SCALING-AUDIT-001 — Measured Result

**Status: MEASURED**

## Run provenance

- GitHub Actions run: **36841250040**
- Successful job: **110300313510**
- Repository gate: **813 passed**
- Source commit: `842edc6d56728df4c3b5ec251adbec8b652a8a1f`
- Artifact: `TACOSM-C5-ADMISSION-SCALING-AUDIT-001`
- Artifact ID: **11151453812**
- Artifact digest: **sha256:ba98834b524590fc55b61172b1ad158514a31ea2260fa299b941737f9a42fcce**
- Seeds: 0–4
- Training: 800 steps/seed, unchanged from C5 noisy phase
- Calibration: 64 trials/seed at M=1024
- Held-out LSH evaluation: 64 trials/seed
- Dense scaling M: 64, 128, 256, 512, 1024, 2048, 4096, 8192
- LSH sweep M: 1024
- LSH tables: 1, 2, 4, 8, 10, 12, 16, 24, 32, 48, 64
- LSH cap: 128
- Query corruption: exactly one flipped bit
- Main arm: no persistent-state input and no hidden-target input

## 1. Dense-CDL scaling

The pooled P90 best-valid rank is:

| M | P90 best-valid rank | Mean best-valid rank | Top-1 valid |
|---:|---:|---:|---:|
| 64 | 2 | 1.256 | 76.25% |
| 128 | 3 | 1.513 | 76.25% |
| 256 | 5 | 2.025 | 76.25% |
| 512 | 9 | 3.050 | 76.25% |
| 1024 | 17 | 5.100 | 76.25% |
| 2048 | 33 | 9.200 | 76.25% |
| 4096 | 65 | 17.400 | 76.25% |
| 8192 | 129 | 33.800 | 76.25% |

The full-range fitted P90 exponent is:

`gamma = 0.8732`.

More importantly, the consecutive-doubling local exponents are:

| M → 2M | Local P90 exponent |
|---|---:|
| 64 → 128 | 0.585 |
| 128 → 256 | 0.737 |
| 256 → 512 | 0.848 |
| 512 → 1024 | 0.918 |
| 1024 → 2048 | 0.957 |
| 2048 → 4096 | 0.978 |
| 4096 → 8192 | 0.989 |

The observed P90 sequence therefore approaches proportional growth in M. The full-range `gamma=0.8732` should not be read as an asymptotic sublinear law.

## 2. Explicit LSH admission sweep

At M=1024 the dense calibration procedure gives:

- K90 = **17** for every seed
- b = **10** hash bits/table
- mean p1 = **0.861875**
- mean p2 = **0.510938**
- iid diagnostic L90 = **9.2 tables** on average across seeds

The iid L90 expression is:

`ceil(log(0.10) / log(1 - p1^b))`.

It is a diagnostic only; it assumes independent table events and does not account for candidate-set truncation by top-K reranking.

### Pooled admission/work curve

| Tables L | Admission recall | Rerank fraction | Mean routing ops |
|---:|---:|---:|---:|
| 1 | 35.94% | 1.025% | 328 |
| 2 | 49.69% | 1.558% | 575 |
| 4 | 63.44% | 2.412% | 1,035 |
| 8 | 76.56% | 3.970% | 1,930 |
| 10 | 80.00% | 4.619% | 2,357 |
| 12 | 81.56% | 5.288% | 2,786 |
| 16 | 86.88% | 6.528% | 3,630 |
| 24 | 93.75% | 8.799% | 5,282 |
| 32 | 94.69% | 10.850% | 6,898 |
| 48 | 97.19% | 14.170% | 10,066 |
| 64 | 97.50% | 16.997% | 13,025 |

All requested table counts are below the cap of 128, so the sweep is unbound by the cap.

## 3. What changed relative to C5 noisy phase

The previous registered operating rule at M=1024 derived approximately five tables from `m^rho`. In the explicit sweep, four tables gives 63.44% admission and eight gives 76.56%; this is consistent with the previously observed ~65% admission under the derived-table rule.

More importantly, admission is not intrinsically limited to the ~65–73% regime. With the same learned representation and the same K90=17:

- L=16 gives 86.88% pooled admission;
- L=24 gives 93.75%;
- L=48 gives 97.19%;
- L=64 gives 97.50%.

At L=64, sparse admission reaches the same pooled level as the prior dense K90 held-out coverage (97.5%), but reranking work rises from 2.72% in the prior operating point to about 17.0%.

The sweep therefore exposes a recall/work frontier that the single `m^rho` operating rule hid.

## 4. Interpretation

### Supported

**Dense rank growth is close to linear by M=8192.**
The local P90 exponent increases from 0.585 at 64→128 to 0.989 at 4096→8192.

**LSH admission is a major independent loss at the previous operating point.**
The current representation can support materially higher admission by increasing L, but only by accepting substantially more reranking work.

**The iid table-count diagnostic is optimistic on this workload.**
The mean theoretical L90 is 9.2, while the first registered pooled table count above 90% admission is L=24. This gap is a workload-specific empirical observation, not evidence against the formula in general; it indicates that the independence approximation and/or the finite-K selection layer are not sufficient to predict this measured frontier.

### Not supported

- No learned semantic-retrieval claim: the benchmark remains a one-bit binary-code routing task with an exact Hamming shortcut.
- No asymptotic complexity theorem from gamma or beta-style arithmetic fits.
- No persistence claim for the main noisy funnel: persistent state is not an input to this arm.
- No claim that L=24, 48, or 64 is an optimal operating point; the sweep reports the full registered curve and does not select on held-out outcomes.

## 5. Consequence for the next research phase

This audit changes the priority order.

1. The dense rank problem is real and becomes near-linear at high M.
2. The previous LSH rule was also materially under-admitting valid targets.
3. Therefore the two issues should not be conflated into a single "CDL bottleneck".
4. Before another representation-objective ablation, the next benchmark should remove the exact Hamming shortcut and introduce a relevance relation that cannot be solved from bit distance alone.
5. A separate persistence-in-the-funnel experiment is also required before attributing any C5 result to persistent state.

The previous #28/#29/#31 objective interventions remain historical diagnostics; this audit does not rerun or reinterpret them as new evidence.
