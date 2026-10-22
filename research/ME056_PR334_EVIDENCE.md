# ME-056 PR #334 — evidence update (2026-10-22)

Source: https://github.com/FPC-effortless/TAC-OSM/pull/334

Status: **implemented, execution qualification failed, comparative training not started**. Do not label this trained or predictively improved.

Architecture: slow-memory consolidation at fixed 128-frame anchors; dense batched physical-transition supervision. Reported corrected ME-055 memory: **8-dimensional relevance trace, not 8x8 matrix**. This is distinct from ME-054 associative 8x8 matrix; do not conflate.

| Gate | TAC | GRU | Gap | Registered outcome |
|---|---:|---:|---:|---|
| Trainable parameters | 38,266 | 38,295 | 0.076% | Pass |
| Profiled FLOPs/update | 42,257,505 | 42,546,619 | 0.680% | Pass |
| CPU/update | 0.038523 s | 0.035837 s | 8.007% | Fail under PR's 5% CPU rule |

Robust paired CPU gap improved from 16.17% (remeasured ME-055 layout) to 8.01% (~50.5% relative reduction). Paired 95% bootstrap CI: 6.35–10.00%, 41 paired samples. Timing gap not explained by shared 16-step rollout. Tensor scan, whole-history compilation, larger batches and prediction-time refinement were tested/rejected in this optimization line.

**Research-rule discrepancy:** user compulsory scientific parity requires matched parameters, training budgets, data and evaluation; CPU time is to be reported, not necessarily matched. PR #334 explicitly enforces a 5% CPU qualification and blocks training. Treat the gate as an existing PR contract requiring explicit governance change, not as scientific evidence that training is unfair. Do not bypass the repository gate silently.

Data: TAC and GRU receive identical source observations, actions, authority information and servo-prior parameters. Factual pre-takeover evaluation cannot establish control-policy performance. Adjacent episodes grouped in blocks of four; operator/session independence cannot be confirmed from publisher metadata.

Validation: 960 repository tests reported passed. ME-049 1,062 tests and ME-055 951 tests arose on divergent branches and must not be interpreted as a monotonic test-count trend. PR's own contract CI passed per report; broader inherited checks were not yet confirmed when authentication expired.

ME-054 training workflow reportedly completed but inaccessible artifacts prevent promoting or auditing its results.

**Next action:** decide explicitly whether to revise CPU qualification as an engineering target rather than a hard comparative-training gate. Preserve 5% failure in evidence either way. Then perform matched real-data training with persistence and GRU controls and artifact accessibility checks; report forecast accuracy, uncertainty and evaluation limitations.
