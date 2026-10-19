# DEV064B — validation-only audit and disposition

**Status: VALIDATION_REJECTED — NO PROMOTION, NO SEALED TEST EXPOSURE.**
This outcome is *not* a scientific demonstration of CASM superiority or a
commercially qualified learned maintenance controller. Confirmatory
20-seed training was not launched.

## Evidence and source boundaries

- Protocol: [PROTOCOL.md](PROTOCOL.md).
- First passing preflight and validated headroom:
  [workflow run 39297088473](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39297088473).
- Original ten trained pilot checkpoints, all 1,024 updates:
  [workflow run 39297333375](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39297333375),
  artifact `dev064b-validation-pilots`, original two seeds
  641650/641651 and five model arms.
- **Independent posthoc, validation-only checkpoint replay**:
  [workflow run 39298684632](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39298684632),
  artifact `dev064b-closed-validation-report`. Audit source:
  [scripts/audit_enabling_casm_dev064b.py](../../scripts/audit_enabling_casm_dev064b.py).
- Train namespace: 641641. Validation namespace: 641642. Reserved test
  namespace: **641643, unopened**. Original source, checkpoints, and
  dataset hashes remain unchanged. DEV068/REAL-MAINT-004 and MetroPT
  July B were neither accessed nor modified.

The initial preflight satisfied the **necessary** task headroom gate:
mean myopic one-step regret 4.1373708855 cost units and 84.375% positive
regret decisions (registered minima 0.25 and 20%). This says one-step
greedy decisions can be expensive, **not** that TAC/CASM is needed to solve
the task.

## Recorded initial-action results, validation only

All five networks (TAC/CASM, no-CASM, GRU, attention and dense MLP)
were trained independently with similar parameter budgets. On 1,024
validation starts, the oracle's initial action distribution was:

| Action | Count | Proportion |
|---|---:|---:|
| STOP | 34 | 3.32% |
| PREPARE | 847 | 82.71% |
| REPAIR entity 0 | 41 | 4.00% |
| REPAIR entity 1 | 48 | 4.69% |
| REPAIR entity 2 | 54 | 5.27% |

A constant PREPARE baseline scores only 82.7148%, so the trained
99.2–99.3% validation accuracy is not explained by *that* constant
action alone. Nevertheless, nine of the ten fitted models reproduced
the same 1,017/1,024 correct initial actions, with identical error
indices; the other no-CASM seed correctly selected 1,016/1,024.
The architectures therefore offered almost no discrimination in
this validation task.

The initial-action regret was 0.002973 for nine pilot models and
0.003313 for the remaining no-CASM pilot. These are *known-law
validation-oracle Q regrets*, not actual mean downstream costs.

## Closed-loop physical validation: decisive limitation

The independent audit restored every raw pilot checkpoint, validated
SHA-256 hashes, regenerated the exact training-era validation inputs,
replayed all initial decisions and simulated 1,024 **validation-only**
physical episodes for each agent with repeated, documented physical
random streams. Each episode has only 2–4 actions and stochastic
repairs. The known-law finite-horizon public **cost-minimizing**
oracle was evaluated on the same starts.

| Arm (pilot seeds 641650/641651) | Explicit STOP & healthy | Mean physical cost |
|---|---:|---:|
| TAC/CASM | 37.402% / 37.402% | 5.2241 / 5.2208 |
| No-CASM | 37.402% / 37.402% | 5.2364 / 5.2508 |
| GRU | 37.305% / 37.402% | 5.2374 / 5.2364 |
| Attention | 37.402% / 37.402% | 5.2364 / 5.2364 |
| MLP | 37.402% / 37.402% | 5.2363 / 5.2364 |
| Known-law cost oracle | 37.402% | 5.1711 |

The oracle optimizes expected **cost**, not probability of task completion;
its 37.4% score alone is not proof that no other policy could exceed
95% completion. Nonetheless, neither the learned candidates nor this
strong known-law reference approaches the registered >=95% per-stratum
reliability gate. All model-arm differences are tiny compared with
the task's inherited budget and stochastic physical constraints.

The posthoc trajectories are **validation diagnostics, not a sealed
confirmatory evaluation**. They do not justify changing the existing
acceptance threshold or test set. CASM's observed cost gap versus
controls is neither a valid 20-seed superiority claim nor a
statistically established economic gain.

## Additional generator defect (kept intact for reproducibility)

The initial-fault fallback is written as:

`tuple(int(j == int(rng.integers(0, 3))) for j in range(3))`

It draws a *different* random target index for every `j` instead of
choosing one index and constructing exactly one fault. Consequently,
33 of the 1,024 registered validation examples have zero initial
faults, although the fallback intended to prevent this. Original
pilot inputs and results are **not** rewritten; correct this only
in a new, separately registered generator and data namespace.

## Scientific disposition

The original benchmark DEV064 (PR230) lacked useful myopic-oracle
headroom. DEV064B fixed headroom but produced a saturated learned
initial-action task, almost identical across models, and failed
closed-loop reliability. The complete preregistered confirmatory
20-seed test is therefore **NOT AUTHORIZED** by these validation
findings. Opening test namespace 641643 or changing the existing
budget/physics within this identity would contaminate attribution.

Next research needs a **new study** (new task source and namespaces),
with both:
1. A feasible oracle-completion reference satisfying the required
   reliability threshold, *and*
2. A nontrivial compositional attribution contrast after including
   tuned independent GRU, no-CASM, MLP and attention policies.

Separately analyze economic advantage and longer-horizon update
dynamics; a strong myopic regret score is not enough.

If strong controls continue to match the learned operator, retire
or narrow the CASM-specific superiority claim. This draft remains
unmerged. Never present a validation-only pilot as a successful
held-out finding.
