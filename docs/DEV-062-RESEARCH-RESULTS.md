# DEV061.5–DEV062: learned identity, uncertainty and external maintenance

Completed **380 fresh fits plus 38 validation-only pilot fits**, retaining all twenty initializations per arm. Portable learned control works on the registered bounded worlds; commercial maintenance superiority is not established.

| Study | Result | Promotion limit |
|---|---|---|
| DEV061.5 | 100.00% completion, 0 primary residual faults, 20/20 combined seed passes | No-CASM and public reference match; supplied identity binding and relationship stream |
| DEV062 | 100.00% completion, 0 primary residual faults, 20/20 completion seed passes | Registered belief gates: KL=False, Brier=False; competitive=False |
| DEV062-R1 | 100.00% completion, 0 primary residual faults, 20/20 completion seed passes | Registered belief gates: KL=True, Brier=True; competitive=False |
| REAL-MAINT-001 | TAC sensor MSE 191.8306; persistence 7.4646 | Forecasting gates fail; no measured inspection prices or verified repair-policy outcomes |

R1 passes its registered median belief gates, but only12/20 seeds individually meet KL<=.02. Its H512 secondary cell records1 residual faults across shared-seed evaluations. Primary success is not a general long-history or zero-failure guarantee.


## What changed

Shared entity content and separate identity binding replace memorized ID embeddings. The identity/robustness factorial tests episode-random identities and shared multi-student DAgger/EMA/curriculum, with raw-history and matched structured controls. Both pure imitation and robustness arms succeed; the treatment bundle did not establish a unique gain. DEV062 removes copied public relationship-state shortcuts and trains joint public posterior distributions, action consequences and action/STOP scores from public histories with missing initial states, noisy receipts and paid inspections. The exported inference contains neither analytical belief filter nor repair planner.

Belief supervision uses exact synthetic public posterior targets during training. Successful completion does not establish that explicit belief predictions cause decisions: the actor also has a latent pathway. Training without posterior targets, general symbolic programs, discovered entity schemas and scalable indexed retrieval remain open. Structural priors and runtime contracts are itemized in [the learned-boundary audit](LEARNED-BOUNDARY-DEV062.md).

## Engineering qualification

The initial per-record incremental cache preserved decisions but was1.51x slower and is rejected. Available-packet batching used0.5202x baseline TAC model CPU locally with unchanged weights and exact recorded decisions. Batched caching failed the additional10% improvement gate against an equally batched control. These narrow measurements include terminal observation work but exclude JSON/tensor creation and external verification. They do not measure physical DRAM traffic, C5 scaling or production SLA. R1 runtime qualification is separate; remote evidence remains UNKNOWN until its actual CI audit passes.

All raw/EMA checkpoints, learning curves, pilots, held-out failures and strict audit corrections are retained in SHA256-indexed split archives. First-seed portable exports were declared before fitting, not selected from favorable tests. Veritas replays use recorded synthetic effects and native execution constraints; they are reproducibility checks, not independent validation of physical dynamics.

## Real outcomes and commercial qualification

Downloaded and hash-verified MetroPT-3 (1,516,948 sensor rows) and the Open Repair Alliance electronics source (416,282 recorded repair attempts). The latter reports226,733 Fixed outcomes, but is community repair reporting rather than independent industrial post-repair measurement. Its problem text includes solutions and must not enter a pre-repair predictor. Neither source provides the full public observations, action alternatives, inspection/repair prices and independent final outcome contract needed to establish economical repair decisions. Unknown and Repairable statuses are not silently encoded as successful/failed repairs.

The external sensor experiment failed against persistence; no neural product deployment, ROI or superiority claim is justified. Do not tune against its opened test period. A prospective external dataset or newly registered held-out corpus is required for subsequent forecasting promotion. A complete commercial claim additionally needs an authorized outcome-verified shadow trial with actual costs. The [acceptance goal](../research/commercial/GOAL.md) remains open at those gates.

## Reproduction and evidence

- [DEV0615 protocol, outcomes, artifacts and limitations](../research/dev0615/RESULTS.md).
- [DEV062 protocol, outcomes, artifacts and limitations](../research/dev062/RESULTS.md).
- [DEV062-R1 protocol, outcomes, artifacts and limitations](../research/dev062-r1/RESULTS.md).
- [REAL-MAINT-001 protocol, outcomes, artifacts and limitations](../research/real-maint-001/RESULTS.md).
- [Recorded replay and portable inference commands](../research/REPRODUCE-DEV0615-062.md).
- [Every sourced dataset and its actual availability contract](../research/commercial/DATA-SOURCES.md).
- [Batched intake qualification](../research/dev062-batch/RESULTS.md) and [rejected cache](../research/dev062-cache/RESULTS.md).

Local full regression:1438 passed. Independent-host CI status must be read from the draft PR; local float64 parity is not universal hardware determinism. No original DEV058–061 checkpoint or historical result was rewritten. No merge or live physical intervention was performed.
