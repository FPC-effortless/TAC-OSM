# S1TEMP007 amendment 001 — replacement source selected BEFORE ANY training or first DEV

October 26, 2026. Preserves original S1TEMP007-PROTOCOL.md and unsuccessful initial fork-pick-up source-only audit run #40295647598. That dataset had no compatible historical LeRobot v2 per-episode schema. No optimizer step or heldout source record had been accessed.

A metadata-only audit run #40296253263 found the NEW genuine **LeRobot ALOHA Mobile Chair** data source, a different mobile bimanual robot task; supported 55 actual episodes x 2,000 physical state/action frames per episode, original 50Hz. Newest historical per-episode v2 candidate publisher SHA **7249d8a935db04f846f9e669b788f26c3a19b645**, with MIT license. This must now be validated against actual publisher API, historical/current license and all file hashes before FIT. No data rows whatsoever were inspected in selecting it.

**This amendment changes only preregistered data eligibility/splits/horizon feasibility, before any model FIT and DEV score**:
- FIT **episode0..29** (30 original complete 2,000-frame recordings), 10 DEV **episode30..39**, TEST **episode40..54** quarantine.
- H128, H512 and **H1024** actual observed-time contexts, same target frame for all arms/horizons.
- Pre-registered evaluation at **8 source sample indices per DEV episode: [1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800]**, 80 real one-frame targets/arm/seed across ten DEV clusters; tiny outcome prevalence limits event inference.
- Training 192 identical steps/arm/seed with H32/H128/H512/H1024, **48** updates for each horizon, batch4, new FIT-only physics normalization/labels. Match actual original sample plan and computational exposure across all three arms; full FLOP parity is still unverified and must be disclosed.
- First source-only CI MUST lock **30** original Parquet SHA256 values and source timestamps before model fit. DEV source FILES not retrieved until all **9** complete model/optimizer/RNG checkpoint SHA frozen; TEST never downloaded.
- Source historical revision must exactly equal the metadata-only selected SHA, fail closed on different history/row dimensions/clock/licence.
- Registered science gate: across every one of three seeds, multiscale H1024 physical MSE >=1% better (i.e. <=.99×) than BOTH original additive H1024 and mono slow16 H1024 AND <= .99 × multiscale own H128; no >5% worse high-motion MSE than constant velocity; measured high-motion AUPRC retained >=90% of best same-horizon control where positive cases exist. H512 comparisons and meaningful loss attribution also mandatory. If no positivity or strict equal training FLOPs, comparative scientific qualification FAIL or INCONCLUSIVE.
- H1024 is now authentically feasible but **not yet executed or qualified**. The source is a different mobile robot task but no independent site or safety qualification can be claimed from metadata alone.

All old S1TEMP006 and S1MOTION006 trained negative outcomes and original gates are unchanged. Four-timescale mathematical model/rates/params/state unchanged; no hindsight experiment selection based on this source's unseen labels.
