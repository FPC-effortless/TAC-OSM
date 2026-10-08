# TACOSM-BASELINE-001

**Status: FROZEN.** This is the immutable baseline. Future modifications must
not silently replace these numbers. If a subsequent run differs, that
difference is a finding to be reported, not a correction to be applied here.

## Provenance

| field | value |
|---|---|
| baseline_id | TACOSM-BASELINE-001 |
| commit | `91597ab` |
| model | tac_osm v0.1 |
| scheduler | strict round-robin over `families[_task_index % len(families)]` |
| task seed | `cfg.seed * 100_003 + task_index * 7919` |
| generator | `build_lookup_task` draws a family-distinct RNG offset, so no two families share a task stream |
| families | relational / state_lookup / replay |
| config | mixed (all three, in that order) |
| seeds | 0, 1, 2, 3, 4 |
| measurement command | `python scripts/measure_baseline.py --n-steps <N> --seeds 0,1,2,3,4` |
| interface symbols | 19 |
| test count @ freeze | 327 (current count in `tests/`: 804) |

## Measured baseline

Accuracy, averaged over 5 seeds, **under the mixed schedule** so the task
stream, the exposure and the schedule are identical across arms. Per-family
exposure is equal by construction: ~N/3 tasks each.

| arm | 60 | 200 | 500 |
|---|---|---|---|
| oracle | 1.0000 | 1.0000 | 1.0000 |
| random | 0.1600 | 0.1390 | 0.1296 |
| static | 0.0300 | 0.0350 | 0.0376 |
| full_context | 0.0300 | 0.0350 | 0.0376 |
| **learned** | **0.2433** | **0.3340** | **0.4396** |

Per-family accuracy of the learned arm:

| family | exposure @500 | 60 | 200 | 500 |
|---|---|---|---|---|
| relational | ~166 | 0.2400 | 0.3612 | 0.5126 |
| replay | ~166 | 0.2300 | 0.2939 | 0.5325 |
| state_lookup | ~166 | 0.2600 | 0.3463 | 0.2743 |

## What this baseline establishes

1. **Oracle preflight holds** at 1.0000 on every family at every step count.
   The environment is solvable and the benchmark is valid.
2. **The learned arm learns, monotonically in the aggregate**: 0.2433 → 0.3340
   → 0.4396.
3. **The learned arm dominates every fixed baseline** on relational and
   replay, where the total-agreement rule is actively anti-correlated with relevance (it scores 0.0000).
4. **`static` and `full_context` are identical by construction** — same total
   agreement rule, two roles in the ablation. Their agreement is expected and
   is not a defect; see below.
5. **No two families share a task stream**, verified by
   `scripts/inspect_scheduler.py --mode cross-family` at 0/60 for all three
   pairs, and guarded by a regression test.

## What this baseline does NOT establish

- **Temporal persistence is not established.** No measurement here varies the horizon between a write and the query that reads it.
- **Persistent write is not established.** The loop's verified-write path is exercised, but its contribution is not isolated.
- **`state_lookup` non-monotonicity is unexplained.** Accuracy rises to 0.3463 at 200 steps then falls to 0.2743 at 500, over equal exposure at each point. This is a real effect in the measurement, not a scheduling artifact — the streams are now distinct and exposure is equal — but the cause is not identified here.
- **History scaling is not tested.** H is fixed at the environment default.

## The measurement defects this baseline replaces

Both were found by running `scripts/inspect_scheduler.py`, and both affected
numbers reported before this baseline existed. The aggregate mixed-run figures
moved slightly because fixing defect 2 changes the `state_lookup` task stream.
