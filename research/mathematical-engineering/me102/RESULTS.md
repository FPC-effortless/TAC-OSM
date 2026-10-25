# ME102 — masked state reconstruction and non-cancellable uncertainty floors

Completed fixed fifteen-fit panel. **No recipe passes the full five-seed gate.**
Original model/optimizer/RNG and live budget receipts were frozen in commit
4a0e75a9 before REPORT scoring. All original results and predictions are retained.

The integrated native architecture adds a shared-address associative retrieval,
current-state probabilistic reconstruction and a FIT-derived staleness floor
applied with `max(modeled_variance, floor)`. Training includes clean histories,
single-channel gaps and jointly masked pollutant groups. Input-only affine memory
schedules retain the parallel scan; belief decoding occurs after it.

Each recipe used the same 98,495-parameter graph, five shared seeds102001–102005,
1,536updates, batch16, and contexts8/64/256. Initial weights, actual augmented
input hashes and sampled orders match within each seed. Dense forward/backward
FLOP estimates and full profiled ATen graph signatures match. Pointwise,
optimizer and hardware FLOPs are not fully measured; CPU is an efficiency outcome.
All29 parameter tensors received finite nonzero gradients in the live audit.

## Original exposed-development results

Five-seed means, full256-hour history; lower errors/scores are better.
Coverage is nominal90%, with registered acceptable range90–98% **per seed**.

| Recipe | Gap PM2.5 MAE | Gap PM2.5 coverage | Gap six-channel NLL | Gap PM2.5 interval score | Ordinary PM2.5 MAE | Full-gate seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original ME101 reference | 37.7470 | 68.67% | 6.2345 | 301.2206 | 32.7709 | historical |
| light_floor | 38.7814 | 88.53% | 5.1446 | 229.2109 | 34.3075 | 2/5 |
| strong_reconstruction | 39.7432 | 96.08% | 5.1327 | 316.7059 | 33.8963 | 2/5 |
| physical_score | 44.1222 | 93.43% | 5.1779 | 221.6744 | 33.3507 | 1/5 |

These recipes improve coverage and likelihood while worsening mean point error.
The strong reconstruction recipe also worsens mean interval score. No blanket
calibration, memory utility, superiority or safety claim follows. ME101 is a
historical reference with a different training recipe/budget; matched comparisons
apply to the three ME102 recipes, not a controlled architecture win over ME101.

There are143 original natural-gap queries in21 calendar-week clusters and4,288
ordinary queries in53 clusters. Five fitted seeds do not multiply independent
observations. Posthoc week-cluster PM2.5 MAE difference intervals are recorded in
PANEL_SUMMARY.json; all include zero and are conditional on these fitted models,
not independent confirmation or multiplicity-adjusted superiority tests.

All fifteen models are included. Gates are the unchanged coverage90–98%, gapMAE
within5% of mapped ME101, ordinaryMAE within5% of LOCF, gapNLL no worse than
ME101mean, and gapintervalscore within5% of mapped ME101. Five individual seeds
pass everything; each recipe fails its every-seed acceptance requirement.

## Data, engineering and limitations

Training uses authentic UCI501 measurements from2013–2014. REPORT2015 was already
exposed; protected2016–2017 numerical outcomes remain closed. Artificial masks
are training augmentation, not new observations or the primary capability test.
The original publisher archive SHA256, raw-row label/eligibility replay, fitted
source/gradient audits and full per-channel forecasts are repository artifacts.

All15 fitted networks pass actual-source versus independent serial reference
checks in FP32 and FP64, including carried chunks, state, reads, outputs and
gradients. These numerical checks do not prove statistical calibration.
PNDS PR14 supplies exact-real scalar variance/age/read-only kernel proofs and a
source-linked Python reference. Full neural/IEEE Lean refinement remains OPEN.
The608-byte float core excludes integer clocks, external source-age metadata,
model weights, buffers and runtime overhead; no embedded target latency was run.
No model is promoted, merged or qualified for authoritative product decisions.

Original artifacts: `research/replay-bundles/ME102/`. Replay command:
`PYTHONPATH=src:. python scripts/me102_reconstruction_panel.py replay`.
Run `me102_panel_summary.py` and `me102_source_receipt.py` for saved-score and
publisher-row audits. These commands do not train or overwrite original scores.

## Original artifact restoration

A separate clean detached checkout of93ecebd41122b15be5957f87d4d72d37e71d0e59
reloaded every original model, recomputed all fifteen model scores and gates,
replayed the panel summary and publisher-row labels, and repeated all30dtype
fitted graph checks successfully. No retraining or checkpoint changes.
[Independent GitHub original15 score replay40189256081](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40189256081)
passed. Expanded graph/summary/source workflow40189416852 was still running when
this receipt was written; its final status is independently inspectable.
`SHA256SUMS` covers all67payload files including15original checkpoints.
The capability scorecard declares TRAINED_REPLAY_VERIFIED with mixed trade-offs
and a failed full gate; its structural validator is not a scientific-gain test.
