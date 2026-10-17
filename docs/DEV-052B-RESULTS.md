# DEV-052B — End-to-end learned predictive belief research

**Completed and replay-audited adaptive development; no robust learned cost/reliability advantage established.**

The joint recurrent world model runs from public observations through persistent state, action-conditioned forecasts, cost-aware decisions, native constrained actions, and receipt-gated updates. Across all five initializations it reproduced the estimated Bayesian reference's native actions exactly. No neural variant passed the registered strict cost-and-failure guard on both cohorts for every initialization.

* Accepted scientific source: `b7a82cf9d122c44657770a70d278f3559f0d9010`; execution ID **DEV-052B-FRESH1**.
* [Registered fresh execution protocol](../research/dev052b/fresh-execution-protocol.json), [machine-readable audited results](../research/dev052b/audited-summary.json), [complete compute accounting](../research/dev052b/resource-accounting.json).
* Five model seeds, **35 fitted checkpoints**, 256 updates each; **512 distinct held-out worlds**, 256 in each cohort, reused across initializations.
* Separate process reconstructed every checkpoint tensor, calibration choice, policy action and verifier outcome. Independent primitive accounting replayed **20,992 policy/world executions and 154,024 native calls**. These are repeated executions of 512 worlds, not additional independent samples.
* **1,299 tests passed.** Primary evidence was executed locally in the managed CPU environment, not a GitHub Actions research job.
* [Draft PR #212](https://github.com/FPC-effortless/TAC-OSM/pull/212), [archived evidence release](https://github.com/FPC-effortless/TAC-OSM/releases/tag/dev052b-learned-belief-evidence-2026-10-17). Research remains draft/unmerged.

## Subsequent scope and forecast-dependence review

DEV-052B is a **generic GRU recurrent world model**, not an MTSK/CDL/CASM experiment. Exact actions do not establish exact conditional forecasts or the TAC-OSM thesis. [DEV-053A](DEV-053A-FORECAST-REVIEW.md) verifies active forecast dependence under zero/shuffle interventions, records value margins and proper forecast scores, and adds crossed stream/seed uncertainty. Original source, outcomes, success rule, excluded split and archive remain unchanged. The true-parameter planner's lower cost with more faults is an economic trade-off, not evidence of brittleness. [DEV-053B](DEV-053B-ADAPTATION-RESULTS.md) separately tests the actual MTSK component under changing rates.

## Audited held-out results

Each table contains **256 distinct worlds / 512 entities**. Neural figures are means over five initializations of the same worlds; brackets give the initialization range of residual faults. Baselines are evaluated once on those same worlds. Lower cost is better; all reported controllers had **zero blocked actions and zero trajectory violations**.

### In training regime

| Controller | Cost/entity | Residual faults (seed range) | Completed worlds, mean |
| --- | ---: | ---: | ---: |
| Known-config Bayes | 1.402344 | 26 | 231.0 |
| Estimated Bayes | 1.402344 | 26 | 231.0 |
| Public event table | 1.414844 | 32 | 225.0 |
| Public verify/retry | 1.531641 | 59 | 201.0 |
| Service all three | 3.600000 | 0 | 256.0 |
| Wait | 2.171875 | 278 | 56.0 |
| Last-event head | 1.531094 | 38.6 [31–45] | 218.4 |
| Recurrent belief + estimated rates | 1.403125 | 24.8 [21–27] | 232.2 |
| Recurrent action forecasts | 1.405117 | 25.4 [23–26] | 231.4 |
| Joint recurrent world model | 1.402344 | 26.0 [26–26] | 231.0 |
| Frozen writer + trained heads | 2.251875 | 278.0 [278–278] | 56.0 |
| Three-attempt BC | 1.403906 | 26.0 [21–37] | 231.0 |
| Three-attempt DAgger | 1.400469 | 25.0 [21–26] | 232.0 |

### Four shifted sensor/repair families

| Controller | Cost/entity | Residual faults (seed range) | Completed worlds, mean |
| --- | ---: | ---: | ---: |
| Known-config Bayes | 1.457031 | 64 | 199.0 |
| Estimated Bayes | 1.551562 | 46 | 211.0 |
| Public event table | 1.542773 | 51 | 207.0 |
| Public verify/retry | 1.589063 | 71 | 191.0 |
| Service all three | 3.639063 | 5 | 251.0 |
| Wait | 2.023438 | 259 | 65.0 |
| Last-event head | 1.639258 | 47.2 [39–56] | 210.2 |
| Recurrent belief + estimated rates | 1.547539 | 44.6 [40–47] | 212.2 |
| Recurrent action forecasts | 1.553359 | 45.2 [42–46] | 211.8 |
| Joint recurrent world model | 1.551562 | 46.0 [46–46] | 211.0 |
| Frozen writer + trained heads | 2.103438 | 259.0 [259–259] | 65.0 |
| Three-attempt BC | 1.551719 | 47.0 [40–60] | 209.8 |
| Three-attempt DAgger | 1.547969 | 44.8 [40–46] | 212.0 |

## Interpretation

The **joint model achieved exact action parity**, rather than an economic improvement: 1.40234375/entity with 26 faults and 231 complete worlds in regime; 1.5515625/entity with 46 faults and 211 complete worlds under shift. Its paired cost and fault differences from estimated Bayes were identically zero. The cost guard requires a strict improvement, so ties do not pass.

Some initializations improved observed cost and faults on both cohorts: recurrent-belief 1/5, BC 2/5, DAgger 1/5. These successes are retained alongside their other initializations. **None passed across all five seeds.** Recurrent belief's average shifted cost improvement was only 0.00402344/entity; its paired descriptive 95% interval for cost difference was [-0.01945410, 0.00953223], spanning zero. Its in-regime mean cost was slightly higher than the reference. A favorable initialization or pooled mean is not the registered robust success criterion.

The trained writer substantially outperformed this initialized frozen-writer control, but this does not establish superiority to tuned reservoirs or causal necessity of a particular memory architecture. The last-event control was worse; encoder/capacity differences prevent attributing that whole gap solely to temporal memory.

Known-config Bayes achieved lower shifted cost (1.45703125/entity) while leaving **64 faults versus 46** for estimated Bayes. It optimizes expected monetary cost, not zero residual faults, and its realized lower cost cannot be described as better reliability. All-three service left **5 shifted faults**, because the third repair is no longer guaranteed; its cost was 3.6390625/entity. No policy is validated as safe for industrial use by this synthetic penalty trade-off.

No held-out outcomes were used to tune a further model or retry a favorable seed. The full excluded first matrix, both preflights, negative controls, and all final initializations are preserved. The next unresolved question is learning and identifying variable latent dynamics from training histories with enough public evidence to adapt, rather than adding components merely to reproduce a compact planner. That follow-up has not been run here.

## Compute and evidence accounting

The accepted run performed 8,960 optimizer updates. Including the excluded original matrix, full training reconstruction, both smoke fits and both smoke reconstructions, research performed **26,992 optimizer updates**, **1,143,832 paid native action calls**, and **709,052 constructed scalar privileged target queries**. Targets include common training, validation/calibration collection, and DAgger states; repeated reconstructions are charged again. Harness-only final scoring reads are not learner queries.

Original matrix elapsed 627.54s; fresh accepted matrix 647.94s; full audit 365.75s. These sum to 27.35 process-minutes before small preflights. Runs overlapped and the summed elapsed figure is **not CPU-seconds**. The accepted matrix's summed fit time was 153.42s, including DAgger collection inside fitting. Regression verification time is separately retained in its logs. No total-compute, FLOP, or runtime advantage over Bayes is claimed.

## What runs end to end

A public maintenance record initializes a 32-dimensional entity-local GRU bank.
Paid noisy inspection receipts and accepted isolate/service receipts update that
bank. The public entity address retrieves one bank; heads predict current fault
risk, action-conditioned repair risk, and observation accuracy. The same bounded
Bellman planner used by the explicit references chooses whether to inspect,
service, or finish. A service decision executes two separately charged native
actions, isolation and service, under the exclusive-slot contract. Accepted
receipts commit the next recurrent state; blocked receipts cannot commit.
Independent native verification scores the final physical state and path costs.
The joint arm trains the writer and prediction heads alongside an auxiliary
public-planner action objective; the deployed joint controller uses forecasts
and cost planning, rather than the auxiliary action-head argmax.

This implements the complete **DEV-052B experimental loop**. Entity addresses,
retrieval, admissibility, Bellman recursion, and native execution are explicit.
It does not train generic CDL retrieval, CASM program induction, the original
multimodal architecture, or a general-purpose world model. Two entities and one
shared resource do not establish routing scalability or resource scheduling.

## Prospectively fixed method

The registered design is `research/dev052b/study-protocol.json`. Execution v1
was committed at 5b79d0c, implemented at ee9b243. An engineering review found its
smoke path and tests exposed its reserved evaluation namespace. While that run
was still executing, before inspecting its numerical results, the fresh
execution ID **DEV-052B-FRESH1** was frozen at **b7a82cf**. The full original
matrix is preserved and excluded from the endpoint. The integrity note and
both source snapshots are included in the evidence archive.

Fresh execution uses `research/dev052b/fresh-execution-protocol.json`:
640 training, 160 validation, 160 calibration, 256 in-regime evaluation and
256 shifted evaluation worlds, each with two entities. Whole worlds are split;
the five initialization seeds reuse the same 512 final worlds. Smoke and tests
have separate seed namespaces. Shifted worlds cycle through four fixed sensor/
repair configurations, including 65–95% sensor accuracy and 70–90% third-repair
success. Healthy state is absorbing and later failures cannot resurrect a fault.
Final completion is based on physical target state and an empty isolation slot,
without the inherited hidden fault-count action quota.

Each neural arm receives 256 AdamW updates (learning rate .003, batch 128,
weight decay .0001, gradient clip 1) on a single CPU thread. Model seeds are
7201–7205. A common exploration trajectory contains four inspections and all
three repairs per entity, with inspection positions randomized independently
of hidden state. Privileged synthetic fault and paired repair outcomes are
training/calibration targets only. Estimated Bayes and the empirical event
table receive the same labeled training data. Neural temperature/risk selection
additionally uses the reserved calibration cohort; the explicit estimated
reference keeps its training-estimated parameters and cost-optimal risk penalty.
This is not a fully matched developer-selection-budget comparison. Only the known-config Bayes reference
receives the actual shifted probabilities at inference.

The seven neural arms are last-event, recurrent fault belief, action forecast
only, joint forecast/action training, frozen writer, matched three-attempt
behavior cloning, and matched DAgger. DAgger uses 128 initial updates, one
student trajectory per training world with public-teacher labels, then 128
updates on the combined dataset. Every arm and unsuccessful seed is retained.
Validation reports final-update predictive scores; it does not select updates.
Calibration chooses a fault temperature by proper NLL, then a risk multiplier
from a fixed four-value grid using actual native costs subject to the explicit
reference's calibration reliability. The choice is saved before evaluation.

The forecasting planner freezes its learned rates within each imagined tree
and updates them after actual observations. The recurrent-belief control uses
estimated explicit rates; policy-only arms use their legal masked action head.
No controller receives a hidden fault bit, pre-sampled repair outcome, true
shifted probability, world seed, or future verifier result as a model feature.

Actual cost is `0.1*inspections + 0.2*isolations + 1.0*services + 4*residual faults`.
The strict benefit guard requires no additional residual faults, blocked calls,
or trajectory violations and strictly lower cost than estimated Bayes, for
**each model seed on both final cohorts**. The known-config reference is a
strong additional comparator, rather than the matched unknown-config guard.

## Replay and scientific limits

The separate audit process reconstructs all checkpoint tensors from training,
repeats calibration selection and deployed decisions, then replays recorded
primitive actions through pinned Veritas `b21d829b62d3cab2811e746267ca305e110ed004`.
A separate action-accounting implementation sums paid charges and checks
receipts, faults, blocked calls, path constraints, and final verifier output.
The same author wrote both paths; this is reproducibility and independent
accounting within one implementation effort, not third-party scientific review
or a second simulator implementation.

The state-size figure is 256 bytes of neural bank storage (two FP32 banks of
32 scalars). Public counters, bounded observation histories, Python object
allocation, logs, and model parameters are additional. Decision latency includes
forecasting/planning; total run time includes collection, fitting, calibration,
and native evaluation. No FLOP or total process-memory advantage is claimed.
Policy-only heads have untrained forecasting heads, so their forecast scores
are not evidence about trained prediction quality.

The paired bootstrap averages model seeds within each world and resamples
worlds, stratifying the four fixed shifted families. It never treats repeated
model-seed worlds as independent observations. Intervals are descriptive
adaptive-development diagnostics, not confirmation across new families or
industrial environments. Synthetic privileged training supervision, bounded
horizons, and the cost-defined residual-failure penalty limit applicability.

## Reproduce and run a checkpoint

Use Python 3.12, PyTorch 2.14.1 CPU, NumPy 2.3.5, pytest 9.1.1, pydantic 2.13.4.
Install TAC-OSM and pinned Veritas editable packages; keep Veritas clean.

```bash
python scripts/run_learned_belief_dev052b.py --veritas-root /path/to/veritas --output results/dev052b
python scripts/audit_learned_belief_dev052b.py results/dev052b --veritas-root /path/to/veritas
python scripts/demo_learned_belief_dev052b.py --evidence results/dev052b --veritas-root /path/to/veritas --arm joint --seed 7201 --shifted
```

The demo loads saved weights, prints action-conditioned forecasts, executes
native actions, updates persistent state, and prints the final physical cost
and verifier result. The manual GitHub workflow provides the same matrix;
local execution must not be described as a GitHub Actions research run.
