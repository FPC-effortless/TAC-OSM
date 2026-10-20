# ME-007 and ME-007F — matched-capacity GRU falsifies earlier native TAC forecasting advantage

**Study:** 2026-10-20. **TAC-OSM remains a native predictive-executable world model, not an SSM/GRU conversion.** Negative evidence preserved. No checkpoint/architecture promotion or merge.

## Original authentic training
[GitHub Actions run 39382074165](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39382074165), artifact `12244849855` (`me007-original-gru41-controls`), initial protocol commit `5bfd80577f53c718c7c456ea9559952d8400021f`.

- Official UCI MetroPT-3 TRAIN-only source, same publisher SHA-256 as ME004/ME006; first 20 days TRAIN, days20–25 exploratory validation (historically opened), 12 named authentic sensor channels, 32 historical rows, horizons 12 and 60 future readouts.
- GRU41 **9,249 parameters** vs original native TAC two-timescale **9,368 parameters**, 1.27% difference, but direct adapter adds **1,082** trainable parameters for approximately **10,450 total** in the final TAC system.
- Three predetermined seeds 240401, 240402, 240403. Snapshots at fixed training 450 and 1,000 updates, no selection based on validation, six exact trained checkpoint SHA256 digests **independently verified** against the archived original score ledger.
- Original CPU training cumulative time: 450 steps 3.01/3.15/3.07 seconds; 1,000 steps 6.86/7.01/7.10 seconds per seed. Original TAC+adapter pipeline was approximately 8.6–8.7 + 1.7–1.8 seconds at 450+160 steps. Although capacity is close, **training compute, total parameters and learning schedule remain not identical**.
- Reused exploratory validation means: GRU41 at 450 steps ~0.2777 MAE; GRU41 at 1,000 steps ~0.2579 MAE; original native TAC+direct adapter ~0.2646 MAE. No scientific advantage claims from this reused validation.

## One-time original frozen future-week test

[GitHub Actions run 39382346906](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39382346906), artifact `12246585064` (`me007f-original-frozen-fair-baselines`), **original first-exposure commit `829f25ee957e180411ac9a8f1cc783befa30e85d`**. Week **March 16–23, 2020**, after all ME003/ME004/ME006 earlier source and evaluation windows; **55,339 recorded physical-sensor observations**, **4,566** valid forecast windows, **zero refits or calibrations**. Complete input and original/adapter/control checkpoint hashes required before opening source; no observations at/after March23 used. 10 separately loaded frozen models: original TAC (2), TAC+direct (2), GRU41-450 (3), GRU41-1000 (3), plus persistence.

| Model | Seed 1 | Seed 2 | Seed 3 | Mean macro MAE |
|---|---:|---:|---:|---:|
| Persistence | — | — | — | 0.648523384 |
| Original TAC | 0.344442527 | 0.358409132 | — | 0.351425829 |
| TAC + direct future-trained adapter | 0.335046467 | 0.348707885 | — | 0.341877176 |
| GRU41, 450 fixed steps | 0.367647001 | 0.361542979 | 0.357348296 | 0.362179425 |
| **GRU41, 1,000 fixed steps** | **0.333661542** | **0.333046417** | **0.326377931** | **0.331028630** |

**Registered result:** GRU41-1000 beats TAC+direct on BOTH paired original random seeds and achieves **3.17% lower mean MAE** than TAC+direct across the available seeds. Against original TAC alone it improves ~5.81%. Training more than 450 steps changed the outcome substantially: GRU41-450 lost to TAC+direct. This strongly implicates training budget/optimization rather than uniquely effective TAC memory mathematics.

### Significant genuine changes
**10,060** selected real large target changes, **5,949 positive and 4,111 negative** (not the earlier degenerate all-positive SCANIA direction metric). Mean MAE conditional on these observed true changes:
- Original TAC: **1.92916**
- TAC+direct: **1.86255**
- GRU41-450: **1.97264**
- **GRU41-1000: 1.79843**

Mean direction accuracy conditional on actual observed changes (NOT a deployable event detector):
- Original TAC: **88.27%**
- TAC+direct: **88.73%**
- GRU41-450: **87.72%**
- **GRU41-1000: 91.89%**

### Scientific decision

- **FAIL** the previous hypothesis that native TAC's two-timescale recurrent forecasting head currently has a distinctive improvement over a sufficiently trained, nearly capacity-matched GRU on this authentic compressor stream.
- **KEEP** authentic trained TAC predictive and direct adapter components, with all original favorable and negative evidence, for future joint action-conditioned study. No transition to GRU/SSM as TAC architecture identity.
- **DO NOT PROMOTE** CDL-PHR associative memory, neural-assembly discovery, oscillatory gates, Lean-verified network behavior, System 2 or real-world decision-cost benefit from any of these results.
- **NEXT:** Train **the unchanged native two-timescale TAC recurrence for 1,000 steps**, with exactly the same optimizer/input/seed budget as GRU41, then compare frozen checkpoints on an entirely later preregistered week. If it closes the gap, optimization was the bottleneck; if not, investigate loss design and update interference, not more architectural components.
- Evaluate on independent compressor/repair domain before commercial claims, and require logged actions, outcomes, utility/cost data before judging neural CASM.
- Preserve GitHub original run artifacts and freeze no-refit future protocol. Both PRs remain research draft and unmerged.
