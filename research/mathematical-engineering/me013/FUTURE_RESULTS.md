# ME-013F — Original frozen real future-time and unseen-site result (2026-10-20)

**Scientific decision:** Native TAC and GRU all beat real PM2.5 persistence at 24h on both known and entirely untrained air-quality monitoring sites. **FAIL** the registered hypothesis that anonymous learned context routing improves native TAC over strong controls. The ordinary GRU is stronger on mean physical MAE at both time and new-site transfer and has a lower recorded CPU cost.

## Immutable first-exposure provenance

- **Successful original evaluation:** [GitHub Actions run 39391688618](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39391688618), original source commit `2df52eee20cc7dd721a9eb7717fb1d454860141d`. Original evidence artifact `12249832054`, `me013f-first-frozen-real-site-time-results`. Test re-evaluation disabled after first exposure.
- **Original nine real fitted models:** [run 39391203298](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39391203298), commit `fc563c13c30b69a9b265c67812276754e2333659`, archive `12249154188`. All nine exact checkpoint SHA256 digests independently verified on the original archive and again within the future evaluator BEFORE opening test-site member files.
- Official UCI Beijing Multi-Site Air Quality archive SHA256 `d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8`, **7,959,991 source bytes**. Twelve real monitoring stations from 2013-03-01 to 2017-02-28; missing input values explicitly masked; no replacement by future values. Only ten original stations/dates before July 2016 were parsed for fitting/diagnostics. Never-trained station members Shunyi and Wanshouxigong were opened ONLY after hash verification and checkpoint freeze.
- Future real cohort: 2016-07-01 inclusive through 2017-03-01 exclusive, but scoring requires fully observed PM2.5 at anchor and +24h; 48 causal past hourly observed sensor rows, eleven physical numeric measurements plus present/past masks and known clock features, shared train-only normalization. Exactly **4,635** eligible future windows from ten known stations plus **917** from two never-trained sites (Shunyi 452, Wanshouxigong 465). Test optimizer updates **0**; no after-cutoff source timestamps parsed.
- Three original seeds 13001/13002/13003, 500 original AdamW fits steps each, nine frozen checkpoints. Models receive no explicit station label. Native TAC plain 8001 params, routed TAC 9361 params and GRU44 9549 params. Matched optimizer steps, **NOT matched CPU time** (GRU ~6.73 sec, both TAC ~13 seconds train).

## Registered true future outcomes: MAE in physical micrograms/m³ (lower better)

| Frozen candidate (3-seed mean) | Ten known stations, later time (4635 windows) | Two never-trained stations (917 windows) |
|---|---:|---:|
| **Persistence, last PM2.5** | 62.827 | 69.024 |
| Historical trailing 24h observed mean | 69.549 | 75.329 |
| Last-day observed motion extrapolation | 101.183 | 111.682 |
| Native TAC plain | 52.489 | 57.864 |
| Native TAC + 4 expert routes | 52.547 | 58.090 |
| **GRU44** | **51.994** | **57.312** |

Individual native TAC plain known-future scores 52.823 / 51.834 / 52.811; routed 52.772 / 51.549 / 53.319; GRU 51.985 / 51.603 / 52.394. Individual heldout-site scores plain 58.040 / 57.343 / 58.209; routed 58.267 / 57.260 / 58.742; GRU 57.387 / 57.072 / 57.478.

- Known-time real source: GRU44 improves over persistence by about **17.2%** relative MAE, native plain **16.5%**, routed TAC **16.4%**. GRU wins across three-seed mean. Neither native TAC nor GRU demonstrates deployable pollution-alert calibration, causality or safety from this numeric task.
- Never-trained-site source: GRU improves over persistence by approximately **17.0%** and plain TAC approximately **16.2%**. GRU best on mean and the routed TAC does not beat plain TAC.
- True large PM2.5 changes: 1267 known-site windows, 278 unseen-site windows (absolute change at least original TRAIN-derived one PM2.5 sigma). Large-change conditional MAE known: plain TAC 108.75, routed 108.96, GRU 108.35 µg/m³; unseen: plain 116.58, routed 116.89, GRU 116.13. This is a retrospective conditional metric, **not** an operationally available alert signal.
- Route mean entropy roughly **1.22 nats** (maximum ln4=1.386), so route usage is nonuniform; however between-site standard deviations of average expert allocation were only about **0.005–0.012**. This does NOT substantiate learned entity identity, independent specialized functions, useful intervention pathways or emergent System 2. Causal ablation, diverse environments and appropriate real outcome tasks are still needed.

## Falsification and next engineering

**REJECT the current four-expert router for architecture promotion:** it has positive dev-only accuracy but fails both first-exposure cohorts against ordinary native recurrent TAC and GRU. Maintain exact original unfavourable scores and weights. **KEEP** real multi-site demonstration that native TAC represents some physically useful historical evidence and generalizes above naive persistence across stations, but it is not better than the conventional GRU here, and its compute cost is higher.

The unified master research plan retains mathematical recurrence contracts, neural CASM, CDL-PHR, executable memory, BDH plasticity, emergent entities, oscillations and PNDS Lean formalization. Future work should explore causally ordered real decision consequences and long-history state verification where real action/outcome evidence supports it. For pure forecasting, high-confidence context/reliability features must improve beyond GRU and persistence under independently registered domains before adding further neural modules.

**No model merge, no production safety claim, no TAC-OSM to SSM conversion.**
