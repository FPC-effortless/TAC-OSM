# DEV063 — belief integrity and relative event clocks

**NO PROMOTION. The commercial model-quality goal remains ACTIVE.**

All 140 fresh fits and 14 validation-only pilots completed before opening the test corpus. Fitting used 4307.12 CPU seconds, below the 7200-second cap. Every failed model and outcome is retained. The predeclared primary is `tac_rel_belief`; the export uses seed 93401, not a test-selected winner.

The primary completed 5120/5120 primary rollouts with zero **final residual** fault bits: 20 fresh training seeds x 64 physical starts × H32/H128/H512/H1024. Those histories, seeds and arms reuse physical starts; they are not 5120 independent environments or real-world trials. All 20 seeds passed each history stratum.

| Arm | Seeds passing every history | Mean completion | Final fault bits | Mean synthetic cost | Median fixed p95 KL | Median fixed p99 KL |
|---|---:|---:|---:|---:|---:|---:|
| tac_abs_latent | 0/20 | 83.574% | 291 | 7.124609 | 0.050573 | 0.278378 |
| tac_rel_latent | 20/20 | 100.000% | 0 | 5.470502 | 0.046907 | 0.288669 |
| tac_abs_belief | 1/20 | 81.348% | 357 | 7.588602 | 0.108974 | 0.248172 |
| tac_rel_belief | 20/20 | 100.000% | 0 | 5.463494 | 0.105104 | 0.251430 |
| gru_rel_belief | 20/20 | 99.961% | 2 | 5.469496 | 0.111723 | 0.287715 |
| no_casm_rel_belief | 20/20 | 100.000% | 0 | 5.461215 | 0.100839 | 0.235101 |
| mlp_rel_belief | 14/20 | 95.312% | 203 | 6.179619 | 1.609496 | 3.832928 |

The primary missed the registered median-seed p95 KL<=0.05 and p99 KL<=0.1 gates: 0.105104 and 0.251430. Every primary seed has p99 KL>0.1. Good action completion is therefore not qualification of a calibrated belief state. These 12-class joint health/relation tails are not directly comparable to the old four-state mean KL.

At 20% sensor error, the primary completed 1267/1280 rollouts (98.984%). Nine worlds retained 12 fault bits; another four were physically repaired but failed to terminate. The higher-inspection-price secondary has zero residual fault bits. These secondary outcomes remain visible despite perfect primary completion.

Relative clocks are the clear controlled result: absolute-clock TAC variants passed all history strata in 0/20 and 1/20 seeds. The paired factorial cost effect of relative encoding was −1.889607 units/world (unadjusted paired-seed bootstrap 95% [-2.419368,-1.386044], Holm sign-flip p=0.00000572). The four TAC factorial arms all selected LR 0.003. This tests an explicit time-independent event-clock prior; it is not learned physical-time extrapolation. Histories contain explicitly typed noise, not 1024 relevant retained facts.

## Does the belief drive decisions?

All 140 checkpoints completed 560 primary cells and 173,231 forced history intervention queries, within 402.09 CPU seconds of the 1200-second cap. Identity reinsertion produced zero action flips; all relative arms had zero clock-origin/unit action flips and zero clock logit change.

| Primary posterior replacement | Action flips | Fraction of primary decisions |
|---|---:|---:|
| identity | 0/23614 | 0.000% |
| exact | 8346/23614 | 35.343% |
| shuffled | 11798/23614 | 49.962% |
| blurred | 12921/23614 | 54.718% |
| clock_shift | 0/23614 | 0.000% |
| clock_scale | 0/23614 | 0.000% |

Among 11652 decision-relevant shuffles within the same public graph, 11182 changed actions (95.966%). The latent control changed only 22/23794 actions (0.092%) under exact-posterior replacement.

The bottleneck structurally eliminates the parallel latent policy path and its probabilities affect actions. But 35.3% exact-posterior action flips expose a substantial interface mismatch: the trained actor cannot be assumed to interpret its input as interchangeable, calibrated public probabilities. Joint encoder/actor coadaptation is a hypothesis, not a demonstrated sole cause. These probes force the original recorded histories; alternative actions were not executed, so flips do not establish better or worse physical outcomes.

The validation 64 diagnostic curves give a further **posthoc clue**: median KL at warmup/final was0.02739/0.03328 for restricted TAC, versus 0.02739/0.01716 for latent TAC. A fresh frozen-belief/gradient-isolation comparison is required before attributing that difference or claiming a repair. No current weights or thresholds were changed.

## Controls, economics and physical risk

TAC-minus-GRU mean cost was −0.006002 (paired-seed bootstrap 95% [-0.021851,+0.006610], Holm p=0.553959). The cost advantage is unestablished. TAC beat the ordered raw-history MLP on this bounded family, but no-CASM also achieved 20/20 passing seeds and zero final faults at 5.461215 units/world. TAC-minus-no-CASM was +0.002279,95%[-0.004741,+0.009598]. This is not evidence for CASM superiority or formal equivalence.

The public-information reference completed 100%, with zero final faults,5.477656 cost/world,4.65625 actions and 2.109375 inspections. The privileged reference completed 100% at 4.919375 cost and 2.0625 actions, without inspections. The latest-record shortcut completed 48.4375% at 17.515 cost. The public reference uses known laws, depth 2 planning and a training/reference-only near-certainty STOP guard; it is not a globally optimal unrestricted policy.

Zero final faults do not mean fault-free interventions. Exploratory recorded-transition diagnostics count 165 new 0→1 fault-bit transitions across primary model replicates; the public reference has 8 per 256 rollouts (160 if repeated 20 times). Primary pre-action fault exposure totals 9254 faulty-entity steps. Healthy-target repair counts are ex post physical classifications, not proof an action was irrational given uncertain information. No transient-risk or real-equipment safety guarantee follows.

## Evidence and limitations

See [statistics](statistics.json), [per-seed table](statistics.csv), [learning curves](learning-curves.png), [seed distributions](per-seed.png), [interventions](interventions.json), [posthoc diagnostics](diagnostics.json), [hardware measurements](hardware.json), [protocol](PROTOCOL.md), and [replay instructions](REPRODUCE.md). Full checkpoint/native/portable replay status is recorded separately in [audit](audit.json); remote CI is a separate gate.

All arms share public inputs and data. Parameters match within 5%; the latent actor has approximately 4% more parameters. Shared 97-parameter source-type gates are frozen after explicit supervision. Public cosine identity binding, FIFO64 and the supplied four-health × three-graph vocabulary remain structural priors. This is posterior-supervised training; neural CASM is a learned operator mixture, not symbolic discovery or C5 indexing. The model does not independently learn all hidden physics.

The separate REAL-MAINT-002 study remains NO_PROMOTION, and July B remains unopened. FHWA LTPP acquisition identifies 334 positive-cost treatment records with before/after condition measurements across 238 sections; that source audit is not a trained road-maintenance benchmark or factory outcome trial. Commercial viability, scalable generalization and architectural superiority remain unestablished.

Local replay completed: all154deployment checkpoints,58,688recorded trajectories,
11,287native Veritas actions (first fresh seed across all seven arms) and28portable
decisions.35,678,379compared telemetry values matched exactly on the local CPU.
This includes failed control trajectories and does not imply task success.
Native replay uses effect tables from the same synthetic physics; it is
engineering reproducibility, not independent physical validation.

The first-seed single-host H128/batch1float64 median decision CPU times were
3.080ms for restricted relative TAC,2.545ms for GRU and2.578ms for no-CASM.
Concurrent CPU load and small samples limit the timing comparison. Logical state
bytes and unique tensor-storage allocations are reported separately; DRAM traffic,
FLOPs, energy and production SLA remain unmeasured.
