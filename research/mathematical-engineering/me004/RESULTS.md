# ME-004: First real-data trained native TAC retention-model results

**Status: ORIGINAL REAL-DATA EXPLORATORY TRAINING COMPLETED; no architecture promotion or merge.**

- Frozen successful original run: https://github.com/FPC-effortless/TAC-OSM/actions/runs/39357593422
- Original model/source SHA: `954baafe63d1450e4968cc1b49dac02a5e8f6f1d`
- Original evidence artifact: `12235023980` (`me004-original-real-tac-models`)
- Original failed software-only preflight: preserved. Its check squared the deliberately zero-initialized output, giving zero loss gradient. A `+ y.mean()` probe fixed **only** this test before genuine fitting; recurrence/data/protocol unchanged.
- Source: real UCI MetroPT-3; official ZIP SHA256 `aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a`. Opened 190,443 real source rows, from 2020-02-01 to 2020-02-25 23:59:57, *not* the later reserved data. 12 physical signals, 32 past observations, 12- and 60-step forecast horizons.
- Data cohort: 12,412 training and 3,235 chronological exploratory validation windows. Training days 0–20, validation days 20–25. Historical ME003 already inspected some aggregate features within the first 30 days; THIS is not pristine independent external qualification.
- Two fixed seeds (240401 and 240402) × three learned models = six original PyTorch CPU fits, 450 AdamW updates each. Original archive contains six SHA256-indexed weights; **six of six checkpoint hashes checked against ledger using independent artifact inspection**. Training workflow and self-test passed.

## Genuine real validation — macro absolute error in TRAIN-normalized sensor units (lower better)

| Model | Seed 240401 | Seed 240402 | Mean over seeds | Trainable parameters |
|---|---:|---:|---:|---:|
| Persistence | 0.482884773 | 0.482884773 | **0.482884773** | 0 |
| GRU | 0.280572372 | 0.282888887 | 0.281730629 | 6,360 |
| Native TAC identity-first | 0.279958090 | 0.281991832 | 0.280974961 | 4,440 |
| Native TAC two-timescale | **0.268731096** | **0.271425002** | **0.270078049** | 9,368 |

All learned arms beat persistence in both original seeds. Two-timescale TAC is 44.07% lower MAE than persistence, and 4.14% lower than the GRU on this *exploratory* validation split; it wins vs GRU in both seeds. Identity-first advantage is only ~0.27% and may be noise.

## Horizon-specific / rare-change observations

- Persistence at 12- and 60-step horizons: **0.353250762** and **0.612518784** MAE.
- GRU mean: **0.229293389** and **0.334167870**.
- Native TAC identity-first mean: **0.228232940** and **0.333716982**.
- Native TAC two-timescale mean: **0.214431179** and **0.325724519**.
- There are **3,861 selected large true future changes** in the validation comparison: **2,521 positive and 1,340 negative**. This avoids the all-positive selected direction degeneracy discovered in ME001B, but direction is still assessed on *hindsight-identified actual changes*, not an action-available change detector.
- Average changed-value magnitude MAE: persistence **3.435252**, GRU **1.927725**, identity **1.934428**, two-timescale **1.818627** (normalized feature units).
- Original duration for 450 training updates: GRU **3.91/4.04s**, identity **4.70/4.76s**, two-timescale **8.63/8.70s**. Two-timescale roughly **2.18×** GRU CPU training time and 47.3% more parameters. That prevents an efficiency or fair-architecture superiority claim.
- Numerical gradients/self-test passed, but these metrics do not identify which component or capability caused the gains. No CDL/CASM/role–filler or semantic entity discovery was tested.

## Decisions

**Keep** the native TAC two-timescale checkpoint family as a runnable real-data *research candidate*, with its original checkpoints and complete results. **Do not claim** state-of-the-art performance, causal action quality, general-purpose reasoning, real repair benefit, or statistically qualified superiority. No architecture promotion, no merge.

**Fastest next validation:** independent recorded-only replay of the SIX exact weights on official original source, without refits; assess a compute/parameter matched GRU (e.g., GRU48), a tuned boosted-tree or linear drift baseline, and fresh time-shifted evaluation after freezing design and model selection. Only continue elaborate memory mathematics if these direct real-data controls expose a specific failure; avoid repeating synthetic memory rediscovery.

TAC-OSM remains its own predictive executable world-model architecture, **not an SSM conversion**.
