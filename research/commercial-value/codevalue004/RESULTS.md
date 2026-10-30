# CODEVALUE004 verified results

**Primary internal generalization gate: PASS. All four TAC utility screens: FAIL. Commercial viability: not established.**

All24 models completed5120 updates on two physically observed distinct TeslaT4s. All120 checkpoints and24 DEV-only selections were sealed before new transfer scoring. The remote run completed successfully in34.17 minutes (68.34 allocated GPU-minute upper bound). No report recovery or extra allocation was needed.

| Predictor | Corpus/schedule | Transfer NLL↓ | Next-byte accuracy↑ | Exact4 /5120 |
| --- | --- | ---: | ---: | ---: |
| TAC | small_constant | 2.3380 | 48.85% | 812.67 |
| TAC | small_cosine | 2.3315 | 48.99% | 820.67 |
| TAC | broad_constant | 2.0369 | 52.47% | 896.00 |
| TAC | broad_cosine | 2.0279 | 52.67% | 888.00 |
| GRU | small_constant | 2.1281 | 51.72% | 940.33 |
| GRU | small_cosine | 2.1201 | 52.20% | 958.67 |
| GRU | broad_constant | 1.8634 | 55.30% | 1028.00 |
| GRU | broad_cosine | 1.8373 | 55.42% | 1015.00 |
| P3 | small | 2.4052 | 47.00% | 748.00 |
| P3 | broad | 2.1783 | 49.09% | 815.00 |
| WB6 | small | 2.7182 | 52.59% | 958.00 |
| WB6 | broad | 2.4094 | 56.01% | 1050.00 |

Neural rows average all3 seeds; count baselines are deterministic and refitted to their corresponding FIT corpus. Each neural fit used only exposed selectionDEV to choose one of five predeclared checkpoints; no transfer-selected winner was substituted. These NLL values concern the new CODEVALUE004 cohort and cannot be directly compared with CODEVALUE003 scores.

## What changed

TAC broad+cosine beats TAC small+constant by0.3101 nats/byte: paired seed/publisher/file95% CI0.2048 to0.4231, positive in all5 publishers and all3 seeds. The main corpus-expansion gain is0.3024 (CI0.2018 to0.4097). The cosine schedule main effect is only0.0078 (CI−0.0014 to0.0177), so a positive TAC schedule effect remains uncertain. Corpus expansion changed both size and content; this is not a pure diversity-effect estimate.

GRU also benefits strongly from expansion and remains the stronger neural predictor on this protocol. Broader data improves TAC, but all four TAC cells fail the existing count-baseline/rejection-cost utility screen. No measured developer benefit, revenue, Android performance or general English quality follows from this result.

## Verification and preservation

VERIFICATION.json confirms all120 original checkpoint hashes, all24 selections, all280 original score files and143360 recomputed saved suggestions against original publisher bytes. There were zero new optimizer updates, new predictions or additional GPU minutes during verification.

The Git replay bundle preserves all24 selected full original model/optimizer/RNG/sampler checkpoints, all raw score files and selection curves. All120 candidate checkpoint hashes are in the selection seal; all120 actual files were downloaded and verified locally and remain in the original private Kaggle version1 outputs. ARCHIVE.json documents this split, avoiding another720MB of candidate duplicates in Git.

Full training FLOPs remain unmatched and architecture superiority is ineligible. The five-project transfer cohort is internal DEV, not an independently protected blind TEST. Three seeds/five projects limit bootstrap precision. Original CODEVALUE003 evidence is unchanged.

The result supports investing in a broader authenticated corpus for the separate10M program. It does not authorize replacing a formal10M training registration, matched controls, adequate unique data, replay, or an actual no-update memory/throughput profile.
