# ME070: original ME069 paired-score audit — **retrospective, exposed TEST**

**Real observational data, no training/reselection.** Source: original Git-owned ME069 `research/replay-bundles/ME069/REAL_EVALUATION.json`; read-only audited by `scripts/me070_audit_me069.py` in GitHub Actions [run 39920681790](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39920681790) (seven fixture tests passed, measured original scores recomputed as posthoc diagnostics). This document **cannot support new prospective TEST or TAC memory-superiority claims**.

## Original paired TEST error overlap and 90% predictive interval diagnostics

All five seeds score the **same** 2,618 later-starting BPI2012 cases. McNemar is per paired seed; NEVER pool repeats as independent observations.

| Paired seed | Both correct | Both wrong | TAC-only right | GRU-only right | Exact McNemar p | TAC median interval (seconds) | TAC p90 width (hours) | TAC p99 width (hours) | TAC maximum width (hours) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 69001 | 2471 | 130 | 7 | 10 | 0.629 | 10.63 | 118.97 | 4605.4 | 33637.5 |
| 69002 | 2464 | 139 | 6 | 9 | 0.607 | 8.07 | 197.25 | 3090.3 | 21087.1 |
| 69003 | 2465 | 117 | 18 | 18 | 1.000 | 6.45 | 108.95 | 3714.0 | 22940.9 |
| 69004 | 2479 | 128 | 8 | 3 | 0.227 | 4.78 | 154.48 | 2134.2 | 26006.9 |
| 69005 | 2481 | 126 | 6 | 5 | 1.000 | 6.68 | 162.63 | 4184.0 | 32341.4 |

- **Accuracy:** at each paired seed only 11–36 of 2,618 cases have a correct-vs-incorrect disagreement. Per-seed McNemar p-values are 0.227–1.00; these provide **no accuracy evidence that TAC beats GRU**. Both may still predict *different wrong classes* on both-wrong cases, so discordance shown is a lower bound on label prediction disagreement.
- **Waiting-time uncertainty:** narrow medians of 4.78–10.63 seconds coexist with 90th-percentile widths ~109–197 hours, 99th-percentile widths ~2,134–4,605 hours (89–192 days), and per-seed extreme maximum widths ~21,087–33,638 hours (2.4–3.8 years). Those extremes explain why the mean 90% width was around 230h. The result **is not uniform overconservatism**; it is radically uneven forecast dispersion.
- **Positive-gap targets:** each TEST evaluation has positive waiting time for 97.86% of cases. Native conditional positive-gap nominal 90% interval coverage ranges 89.89–93.13%; reporting mean coverage alone does not establish well-calibrated predictive intervals.
- **Missing independent endpoints:** original Git-owned per-case scores have widths and coverage but not the complete lower/upper/target timing tuple or next-target categorical label. Do not invent Winkler scores, pinball, lifecycle START→COMPLETE prevalence or process-transition confusion from the saved summary; require original source-licensed scored replay to recompute them, or obtain them on a fresh registered real cohort.
- **Preservation:** This is the Git-tracked after-the-fact diagnostic of the **already-exposed ME069 TEST**. Do not revise ME069 training, use these results to choose a purported ME069 TEST winner, or count this as ME070 fitted real-world performance. The original repo-owned checkpoints remain authoritative.

## Next gate

Qualify authentic 64/128/256-event *actual* history support on the candidate BPI2018 source before model fitting. Continue only after source and new `FAIR_COMPARISON.json` pass; the registered **64-event kill criterion** and stronger controls are in [PROTOCOL.md](PROTOCOL.md). `ME070_FITS=0`, `SOURCE_256_SUPPORT=UNKNOWN`, `MEMORY_SUPERIORITY=NOT_ESTABLISHED`.
