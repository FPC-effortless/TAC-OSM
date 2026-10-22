# ME-057 — source-backed transition to scientifically matched real robot training

**Status:** original FIT data qualification and machine-checkable prospective parity governance. This phase **must not fit a model or claim prediction improvement**.

## Directly observed historical engineering evidence
Native anchored TAC of draft PR #334 has 38,266 active parameters against 38,295 for its GRU control, profiled update FLOPs 42,257,505 vs 42,546,619, and paired CPU overhead 8.007% (95% bootstrap interval 6.353–9.999%). The **5% CPU engineering gate failed** under historical parity-v2. We preserve that exact historical failure.

### Scientific governance correction
The repository's existing `AGENTS.md` and `research/REAL-WORLD-EVIDENCE-RULES.md` state that CPU time is not compulsory equality; parameters, identical real data/order, training budget/compute and evaluation conditions are. ME-057 adds prospective schema `tac-osm.parity.v3` for NEW studies only. Historical v2 retains the 5% rule in the unchanged old receipt. Neither version grants superiority on the basis of profiled operator FLOPs alone. The validator still requires total registered and actual training-compute parity, real sample order, and signed checkpoint hashes. If profiling undercounts operations, explicitly disclose incomplete compute measurement and withhold a fully matched scientific win.

## Source-only qualification before any full fit
Use the already outcome-blind selected independent robot task `clean_stains_off_tabletop`, official BodenAI/RW-RL-HIL-Dataset pinned at revision `9290f5a96dc91ea58256e81d9ce2dd9621890b51`. Metadata-only reservation establishes 228 FIT / 76 DEV / 80 TEST episodes, four-adjacent-episode hash blocks, without verified cross-block operator or session independence.

ME-057 script:
- verifies both publisher metadata byte hashes against `research/me055-independent-robot-task-source.json` and recomputes exact registered episode-ID membership;
- downloads and hashes **all 228 FIT original episode files**, verifies original 14D state/action, real authority flags, exact frame counts and finite values;
- counts actual FIT takeover onsets and robot-controlled candidate origins;
- writes immutable per-episode original parquet hashes and a canonical aggregate SHA into an audit receipt;
- does **not** download DEV or TEST trajectory bytes, fit any network, or read heldout outcomes.

The GitHub Actions job uploads only JSON hashes and logs, never the raw publisher data. Partial download or changed source must fail with no success receipt. Failed source qualification prohibits future comparative training. The FIT audit is a source integrity and access test, not a model capability experiment.

## Next registered experiment (not yet authorized)
After source audit, freeze in version control a NEW genuine `architecture_comparison` v3 manifest: all original source/FIT and split hashes, real input/target and censoring contracts, actual model parameter and profile receipts for the exact training objective, >=2 paired seeds, fixed training updates and schedule, equal exposure and measured training-compute budgets, original evaluation/selection policy, untouched TEST. The present ME-057 diagnostic manifest does **not** meet these conditions.

Then implement and audit a genuine FIT training runner for 128-anchor TAC and a GRU with the SAME authentic source inputs and access to dense physical transition targets. Verify no one-arm target privilege, chronological state alignment, post-takeover censoring for claimed physical outcomes, finite gradients, equal optimizer settings, and exact sample-window schedule. Compare physical persistence, GRU and TAC on factual before-takeover trajectories, retain every seed and negative result, and verify source/compute/log/checkpoint receipts before any architectural superiority claim.

Previously open earlier robot TEST and sample-dependent exploratory design results require labeling developmental; the independent selected task offers new heldout task episodes but does **not** guarantee independent operator/session splits. Neither factual logged action forecasts nor takeover labels establish counterfactual policy performance.

## Existing prior art
- R2D2: recurrent burn-in/replay, https://openreview.net/forum?id=r1lyTjAqYX
- DreamerV3: learned world-model dynamics, https://doi.org/10.1038/s41586-025-08744-2
- TD-MPC2: action-conditioned multihorizon dynamics, https://arxiv.org/abs/2310.16828

**Unresolved question:** Does ME-056 anchored fast/slow relevance state improve *real physical forecasting beyond persistence and a matched GRU*, rather than simply enabling a stable implementation? ME-057 qualifies data and governance only; it does not answer this yet.

## Distinct variant naming
The anchored native **ME-056 PR #334** is NOT the concept-swarm/CED **ME-056 PR #332**. Both are documented as separate lineages. Results must not be conflated.
