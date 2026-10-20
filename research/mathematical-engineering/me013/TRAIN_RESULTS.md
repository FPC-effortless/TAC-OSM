# ME-013 — Original authentic multi-station training results

**Source and original run:** successful [GitHub Actions run 39391203298](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39391203298), original source commit `fc563c13c30b69a9b265c67812276754e2333659`, original artifact `12249154188` (`me013-original-ten-site-real-models`). All **nine** checkpoint SHA-256 values verified independently against the original artifact bytes.

**Source attribution:** UCI Beijing Multi-Site Air Quality (Song Chen, doi:10.24432/C5RK5G), authentic observations of 12 real air-quality monitoring stations, four years hourly. Original UCI legacy ZIP URL recorded in `PROTOCOL.md`, observed original archive SHA256 `d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8`. Original trainer downloaded the publisher ZIP bytes but **parsed only ten registered station CSVs up to 2016-07-01 exclusive**; the two heldout station members were not opened and no later sample rows were parsed. Exactly 29,232 observed hourly time stamps per parsed station.

Global 11-physical-feature mean/std from real FIT stations before 2016-01-01 ONLY. Authentic PM2.5 observed 24h future change target, 48 causal measured history hours plus missingness masks/observed clock features. Fit **19,782** eligible windows; historical development Jan–Jun 2016 **3,461** eligible windows. No synthetic sample used for scientific fit. Each model 500 fixed optimizer updates, batch64, same standard Huber/AdamW/clip, same 3 seeds 13001/13002/13003. Neither full pipeline nor learned operator stack was modified.

## Genuine but previously used development metrics (lower better)

| Learned forecast | Seed 13001 | 13002 | 13003 | Mean MAE µg/m³ | Params | Mean CPU training seconds |
|---|---:|---:|---:|---:|---:|---:|
| Native two-timescale TAC, plain | 44.99844 | 46.18085 | 45.79262 | **45.65731** | 8,001 | 12.72 |
| Native TAC with four anonymous learned routers | **44.58525** | **45.38217** | **44.89799** | **44.95514** | 9,361 | 13.07 |
| GRU44 recurrent control | 45.08358 | 45.42857 | 46.11913 | **45.54376** | 9,549 | 6.73 |

The routed TAC improves against plain TAC on this **development** split in all three seeds; it also beats GRU on mean but the difference is small relative to absolute error and 3 seeds. The GRU is faster and has comparable parameters to routed TAC (9549 vs 9361, ~2% size difference). No test-site or final-time measurements were consulted before checkpoint freeze. **This is NOT proof of discovered entity identities or architecture superiority.**

Learned router mean route entropy on historical dev is ~**1.21 nats** vs max ln(4)=1.386. This shows nonuniform but still mixed functional selection, NOT station specialization or concept emergence. Each of the four computed expert routes is anonymous; no station label was input. Causal ablation or route/station MI with heldout real data is needed before functional semantics.

## Independent future source already reserved
A separate stacked branch/PR scores all frozen models and physical baselines, after verifying original SHA256 digests, on (a) genuine later-time records for ten previously seen stations and (b) genuine records from two never-trained stations Shunyi and Wanshouxigong. The future period is 2016-07-01 through 2017-02-28; later or heldout station rows were never opened in original model fitting. Do not treat reused historical dev as untouched evaluation.

**No model promotion or merge.** Preserve all prior ME001–ME012 failures, native TAC, CDL-PHR, executable memory, BDH plasticity, PNDS Lean and CASM action/outcome hypotheses in master plan. These air-quality observational forecasts cannot verify causal intervention or downstream commercial decision cost.
