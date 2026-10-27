# MODEL01-TAC-CAL-ONLY-UNCERTAINTY-001 — genuine original CAL-only interval results

**Evidence:** original immutable trained native TAC UCI374 three-seed checkpoint (103,252 parameters), source-pinned ZIP, original CAL and previously exposed DEV. **Scientific decision: `DEVELOPMENT_MIXED_TRADEOFF_COVERAGE_GATE_FAILED`.** No trained weights changed, no protected future TEST opened, no independent household/model generalization.

## Reproducibility

- [Original frozen CAL-only source CI #40493248005](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40493248005): **PASS eight software/security/unit tests and actual source computation**, with original result committed as `research/source-evidence/MODEL01-TAC-CAL-ONLY-UNCERTAINTY-001-ORIGINAL.json` (source Git blob SHA `bfd4c5249b3d441c44c42c82b7197dda2cffcc6d`).
- Original trained Git checkpoints and independent no-fit replay are in parent [PR #502](https://github.com/FPC-effortless/TAC-OSM/pull/502), source [training #40488233306](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40488233306), and [fresh checkout #40488762280](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40488762280).
- Source study frozen before this run: `research/registrations/MODEL01-TAC-CAL-ONLY-UNCERTAINTY-001.md`. Only first 60% CAL used to fit interval offsets, middle 20% selected exactly once with >=1% interval-score improvement and >=88% coverage, final 20% CAL independently *within the already exposed household* audits but cannot change the candidate. Same original 1,968 DEV cases across every method and seed.

## Original paired 90%-nominal DEV results

| Fixed original model | Original symmetric interval score ↓ | Selected phase-asymmetric interval score ↓ | Original coverage | New coverage | Original mean width | New mean width |
|---|---:|---:|---:|---:|---:|---:|
| Seed 1701 | 447.2049 | 333.0416 | 91.21% | **88.26%** | 146.43 Wh | 219.76 Wh |
| Seed 1702 | 440.3443 | 340.7412 | 90.55% | **89.74%** | 153.39 Wh | 211.63 Wh |
| Seed 1703 | 422.9428 | 341.3217 | 90.40% | **89.63%** | 157.84 Wh | 216.27 Wh |
| Mean (three original checkpoints, not three independent households) | **436.8307** | **338.3682** | **90.72%** | **89.21%** | **152.55 Wh** | **215.89 Wh** |

Average DEV interval-score improvement **22.54%**, mean interval width **41.52% higher**, mean coverage **1.51 percentage points lower**. Scores improved largely because the original symmetric intervals missed high outcomes disproportionately (overcounts 166/161/152, undercounts 7/25/37), while phase-conditional asymmetric bounds reduced costly miss magnitudes but became much wider (new overcounts 108/99/100 and undercounts 123/103/104). The original point forecasts and source MAE remain exactly unchanged.

## The prespecified veto

All three **actual selected scheme** CAL-AUDIT coverage values fall below nominal 90%: seed 1701 **88.85%**, 1702 **88.59%**, 1703 **88.97%**. All three exposed DEV coverage values also fall below 90%, so every seed **FAILS** the registered development reliability condition even though all interval scores improve. This is **not** `INDEPENDENT_GAIN`, `RELIABLE_MODEL`, `CALIBRATED_90`, new TAC mathematical memory breakthrough or deployment approval. There is no independent data: original DEV had already influenced past model work, and CAL windows are temporally dependent.

### What this teaches

At equal candidate forecast centers, original symmetric CAL intervals can allocate tail risk badly. Asymmetric time-group intervals may meaningfully lower proper interval score while risking undercoverage. One should evaluate **joint interval score + expected coverage + actual width + lower/upper misses + delayed-response utility**, and keep a predeclared coverage veto. The evidence supports further *calibration* work but not changing the output silently for customers.

### Subsequent research — distinctly post-hoc after observing this DEV

Future CAL-only bias/coverage repairs must be labeled a **new retrospective DEV study**, frozen before any new measurements, not a fresh independent test. For commercial reliability, acquire a genuine new source-authorized same-target customer household with authenticated observation `available_at`, fixed train/calibration/never-exposed evaluation and task-appropriate baselines. Preserve all original 100k neural weights and non-replayable-gate constraints. No action, revenue, physical control, production cloud or new REFIT target equivalence is demonstrated.
