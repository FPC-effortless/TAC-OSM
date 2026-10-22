# ME-057 research result and running status

As of 2026-10-22 this branch contains prospective parity-v3 governance code and original FIT source-audit tooling. **No result of the new authentic FIT audit, no comparative model fit and no predictive improvement are yet verified.**

## Verified from GitHub PR #334 prior to this branch
- Anchored native TAC / GRU active parameters: 38,266 / 38,295, relative gap 0.076%.
- Reported profiled FLOPs/update: 42,257,505 / 42,546,619, relative gap 0.680%.
- CPU/update: 0.038523s / 0.035837s, paired relative overhead 8.007%, 95% paired bootstrap interval 6.353–9.999%. The original v2 5% engineering CPU gate **failed**, unchanged.
- 960 repository tests reported on ancestor and existing PR #334 GitHub checks reported green.

## New source-code changes, not yet independently CI-confirmed
- Prospective `tac-osm.parity.v3` permits unequal CPU time but preserves original v2 checks.
- Tests cover v3 CPU nonparity allowance, real FLOP and data equality, retained historical v2 CPU failure.
- Real-world source checker validates original publisher metadata and all FIT original episode bytes before future training.
- Separate GitHub Actions job performs this source audit and stores **only hashes and logs**.

## Hard stop
The current `FAIR_COMPARISON.json` remains **noncomparative_diagnostic** with `training_authorized: false`. Future scientific model fitting requires a separate fully registered architecture-comparison manifest built with the independently verified FIT source receipt, exact training objective FLOPs, sealed evaluation and matched exposure/compute conditions.

Neither original TEST results nor the asserted predictive superiority of a model are established by this branch.
