# ME098: coherent trajectories, failed uncertainty qualification

**Full registered gate FAIL: all5models fail coverage.** No further ME098fits.

| Horizon(frames) | Persistence MSE | Candidate mean MSE | MSE reduction | Nominal90%coverage |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.000493550 | 0.000302775 | 38.65% | 81.01% |
| 4 | 0.007526495 | 0.004534700 | 39.75% | 75.86% |
| 16 | 0.094477579 | 0.056010884 | 40.72% | 76.20% |

All5seeds pass forecasting, same-task frozen-parent H1preservation and proper interval-score gates. Coverage falls below85%at every horizon for all5seeds; joint NLL is also worse than the FIT-moment Student-t reference. A mathematically coherent density is not necessarily well calibrated.

New learned capability: simultaneous1/4/16frame predictions from one fixed conditional trajectory law, preserving the priorone-step skill.13already exposed DEV episodes,104windows,19,968window-target positions permodel; repeated/overlappingpositions and5seeds are not additional independent episodes. Source, oldstats, previous-frame actions and all originalfuturelabels are pinned. No TEST evaluation. Report allrawper-episode/horizon scores inCASE_SCORES.json.

103,322parameters,512newupdates eachfrom5fixedME097parents;768totalweight-lineageupdates. Fixed loss/budgets and no sweep. All originalweights/optimizer/RNG/schedules and live receipts frozen36e8371b before scoresde167f43. Original score replay and concrete trained-source math verification are separate gates; empirical coverage failure is never repaired by a software pass.

Lean finite-gain composition, fixed-coefficient mean semigroup, velocity-deviation bound and Gram nonnegativequadratic form compile with standard Lean axioms(propext/Classical.choice/Quot.sound), no sorry. Typed evidence interface preserves observedworldstate across imaginedqueries. Independent full neural Python state/sensitivity propagation matches sourceFP32/64 outputs/state/gradients. Full neural/finite-precision Lean refinement remainsOPEN.

Bounded next iteration: ME099 may add one coherent shared uncertainty regime and a proper marginal interval objective, with a NEWpreregistration, all5originals retained and no changed thresholds. No posthocDEVinterval inflation, hidden retry or further ME098updates. Limit this continuation to one correctedfive-seed architecture study, then preserve its actual result.

Capability value: read-only physical forecasting. Control, action intervention, other5domainheads, newsite/longmemory, customer and commercialvalue remainNOT_MEASURED. No architecturewinner or Mamba-equivalence claim; all research remainsdraft/unmerged.
