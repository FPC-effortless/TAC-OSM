# DEV-061 — Entity-local learned memory: useful, not sufficient

**Decision: NO_PROMOTION.** Entity-factorized memory substantially improves the generic-slot learned controller on this bounded public-state benchmark. It does not outperform the strongest controls or meet the registered reliability and cost gates. No claim of general hidden-state reasoning, CDL/CASM superiority, or production readiness follows.

## Registered design and learned boundary

The original protocol was committed before fitting (first registration `86425fd`, finalized fitting source `1e27956`). Seven arms each received ten fresh initialization seeds (9101–9110), plus two learning-rate pilots on a separate seed (9100): 84 original fits. Every fit received 256 belief/read warmup and 1,200 controller updates, following 600 admission updates. All 70 fresh fits closed before original test generation. Learning rates were chosen using validation only: 0.001 except entity-aging at 0.002. Validation ties retain recorded pilot completion order. Curves, both pilot rates, negatives and all weights are archived. Fixed updates are not proof of convergence.

The TAC inference pathway retains learned admission/encoding, recurrent updates, CDL-style reads, neural CASM operators, action-conditioned latent prediction and action/correction/stop scores. Action selection is masked argmax of learned scores. There is no scripted repair planner, retry procedure, or stop override inside these TAC artifacts.

**Programmed structural priors remain:** a 22-number typed public-event interface, categorical public IDs and entity-local routing, fixed slots/FIFO layout, action vocabulary and candidate validity interface. Public IDs are disclosed experimental priors, not learned entity discovery. The matched structured-GRU control removes CASM but retains the other shared pathway; it is not a plain standalone GRU policy. The strong public-feature MLP uses a programmed latest-state/parent parser and actor-focused objectives; it is a control with a representation/optimization advantage, not a fully learned observation-to-state system.

The four entity arms share dimension 47; generic uses 48 and the structured GRU 61. All active parameter counts are within 5% of generic. Updates/examples are matched; FLOPs, training time and state bits are measured rather than claimed equal. Admission is pretrained on explicit public record-type labels and then frozen. This does not establish unsupervised relevance discovery.

The shared 4,096 training and 512 validation snapshots cost 6,894.56 synthetic collection units, charged once for observations and executed collection actions, not once per epoch/model. Analytic counterfactual targets and private instrumentation are free in this simulator; their real acquisition/labeling cost is UNKNOWN. No commercial break-even is claimed. See [corpus accounting](../research/dev061/training-data-cost.json).

## Primary clean results

Each seed is evaluated on the same 384 worlds: N=2/4/6, H=32/128/512/1024, 32 worlds per cell. The 3,840 outcomes per arm share worlds across seeds and are not 3,840 independent environments. The registered primary endpoint is the fraction of seeds strictly above 95% pooled completion; the stricter all-cell endpoint is also retained. Cost includes initial/receipt observations, interventions and terminal fault penalties; logical writes are reported separately.

| Arm | Seeds >95% | Seeds >95% in every cell | Mean completion | Mean physical repair | Final faults, all ten seeds | Cost/world |
|---|---:|---:|---:|---:|---:|---:|
| generic | 0/10 | 0/10 | 69.43% | 70.55% | 1991 | 16.3454 |
| entity | 10/10 | 5/10 | 98.54% | 98.62% | 106 | 6.1702 |
| entity_aging | 9/10 | 2/10 | 96.90% | 97.89% | 138 | 6.3832 |
| entity_retention | 9/10 | 4/10 | 98.67% | 98.67% | 135 | 6.3020 |
| entity_aging_retention | 6/10 | 1/10 | 95.29% | 95.44% | 289 | 7.1646 |
| structured_gru | 9/10 | 8/10 | 98.96% | 99.06% | 84 | 6.0445 |
| public_mlp | 10/10 | 10/10 | 100.00% | 100.00% | 0 | 5.5156 |
| Public rule / privileged teacher | n/a | n/a | 100.00% | 100.00% | 0 | 5.5156 |

The public and privileged references coincide on these clean, fully observable worlds. Public-rule features plus an MLP reproduce that ceiling at the same measured physical economic cost. Thus neural modules have not earned an economic advantage over the public rule. Observation costs and worlds differ from DEV-059: these cost totals cannot be compared directly to its 2.20-unit reference.

Entity memory improves completion over generic slots by 29.11 percentage points (paired-seed bootstrap 95% interval 27.24–30.99 points); the seed-pass comparison has Holm-adjusted p=0.0078125 across four entity comparisons. Economic cost falls by 10.1753 units/world (paired-seed interval −10.9197 to −9.4512). This supports the implemented entity-memory recipe relative to generic slots. It does not isolate whether routing, recurrence, supervised latest-read learning, or their combination produces the gain.

Entity memory costs 0.6546 more than the public rule (paired-seed interval +0.2493 to +1.1008), and its mean cost is 0.1257 above the structured GRU (paired-world interval −0.1152 to +0.3798 after averaging paired seeds). The structured GRU is a competitive explanation; CASM-specific superiority is not established. Aging does not improve the aggregate result. Its independently selected learning rate also means the factorial compares trained recipes, not just identical-optimizer kernels.

Only five entity-memory seeds pass every N/H cell, below the required nine. Registered stale-error and Brier improvements were each required to reach 0.01; the generic model starts below those absolute thresholds on the fixed probe set, so these gates cannot be met here. This is a registration-design limitation: feasibility of those absolute improvements was not established before fitting. The thresholds remain unchanged. Validation generic-to-public cost headroom is 1.5192, but this is an optimization gap, not evidence of intrinsic task uncertainty or a universal oracle opportunity.

See [all paired seed outcomes](../research/dev061/statistics.csv), [full statistics and unchanged gates](../research/dev061/statistics.json), [all learning curves](../research/dev061/learning-curves.png), [both LR pilots](../research/dev061/learning-rate-pilots.png), [seed completion](../research/dev061/per-seed-completion.png) and [seed costs](../research/dev061/per-seed-cost.png). Bootstrap summaries use paired seeds; world bootstraps first average paired seeds. No pooled seed/world independence assumption is made.

## Retrieval, state estimation and stopping

| Arm | Auxiliary stale rate | Fixed current-state Brier | Early-stop worlds | Missing-stop worlds | Actions/world |
|---|---:|---:|---:|---:|---:|
| generic | 0.386% | 0.00356805 | 816 | 358 | 3.759 |
| entity | 0.000% | 0.00000011 | 7 | 49 | 3.246 |
| entity_aging | 0.000% | 0.00000001 | 32 | 87 | 3.317 |
| entity_retention | 0.000% | 0.00000006 | 10 | 41 | 3.217 |
| entity_aging_retention | 0.000% | 0.00000002 | 101 | 80 | 3.306 |
| structured_gru | 0.000% | 0.00000002 | 0 | 40 | 3.230 |
| public_mlp | n/a | 0.00007284 | 0 | 0 | 3.091 |

The registered stale metric is the top-ranked **auxiliary current-state query head 0**, not proof that a policy acted on stale evidence. Additional action-head-0 status reads exclude stop and assume a status role; they are exploratory. Full-public-history visibility is reconstructed independently of retained slots. Zero stale reads on these probes is not a guarantee of correct belief or stopping. Stop precision/recall, completion and stop-probability calibration, physical repair, unnecessary repairs and healthy-to-stop lag are retained per seed in statistics.json. Lag censored at budget exhaustion is explicitly recorded; repair on the final action has a censored lower bound of zero.

Frozen DEV-058 checkpoint diagnostics force retained latest-status/latest-relation reads, or oldest-status reads with the same relation rule. All weights stay fixed. Stop reads are null under both forced conditions. The intervention changes addressing/concentration as well as recency; it is a conditional-format diagnostic, not an oracle belief updater or universal causal explanation.

| Frozen seed | Learned reads completion | Forced latest completion | Forced oldest completion |
|---|---:|---:|---:|
| 7581 | 78.12% | 95.31% | 19.27% |
| 7582 | 79.43% | 100.00% | 19.27% |

Forcing latest reads helps these frozen checkpoints, but does not uniformly close the gap. The original DEV-058 evaluation remains intact; these are new, longer-history DEV-061 worlds.

The original restoration audit records zero admission false positives/negatives against explicit public signal/type labels. This is fidelity to the supervised type rule, not discovery of task relevance, and the skip fraction follows the signal base rate. Counters include validation, test, fixed-equivalence and portability paths, with repeated observations; they are not independent samples.

The episodic retention gate mostly collapses to discard-all: nine of ten original entity-retention seeds and all ten aging-plus-retention seeds make **zero episodic writes** across those audited paths. The remaining seed makes 279. Entity-local recurrent state is still updated. This is compatible with an objective whose next-decision labels rarely require episodic evidence; it does not establish useful long-horizon retention. Original training-label integrity remains unqualified as described below.

## Retention integrity finding and DEV-061-R1 correction

A float32 training defect was identified independently before numerical held-out results were inspected: left-associated `hard + p - p.detach()` can produce 0.99999994 instead of 1. Raw timestamps then lose exact identity, potentially misaligning the counterfactual retention target. The number of affected original labels was not logged and is UNKNOWN. Both original retention arms are unqualified as evidence for the intended retention mechanism regardless of score. Their source, weights and results are preserved, not overwritten.

DEV-061-R1 was separately registered before correction fitting/new tests. A training-only subclass uses `hard + (p - p.detach())`, with the same gradients and exact forward binary gates; deployment inference, loss weights, data and learning rates are unchanged. Twenty refits use the same ten paired initialization seeds per arm, not twenty independent seeds. Original plus corrected fitting consumed 176.23 CPU minutes, within the shared 180-minute cap. Every registered fit finished its full updates.

Fresh clean worlds (test seed 5611203, fixed probe seed 5611204) compare original versus corrected retention checkpoints. There is no full matched neural-control grid or corrected OOD evaluation on these new worlds.

| Arm / cohort | Seeds >95% | Mean completion | Final faults | Cost/world |
|---|---:|---:|---:|---:|
| entity_retention / original | 10/10 | 98.85% | 127 | 6.1685 |
| entity_retention / corrected | 9/10 | 98.70% | 125 | 6.1765 |
| entity_aging_retention / original | 6/10 | 95.42% | 317 | 7.2199 |
| entity_aging_retention / corrected | 6/10 | 94.71% | 362 | 7.4826 |

Corrected entity-retention reduces the observed total fault count from 127 to 125, but the paired fault-difference interval spans zero (−0.00443 to +0.00312 per world): it does not establish the registered no-regression guard. Corrected aging-plus-retention increases faults from 317 to 362, with interval −0.00756 to +0.03438 per world. **Neither correction is promoted as a served model.** Exact arithmetic repairs training integrity; it does not establish a useful retention mechanism or improved reliability.

The retention target is a local next-decision deletion proxy, selecting only retained episodic items while entity state is held fixed. Its asymmetric risk surrogate includes an extra fault penalty on top of teacher stop cost; it is not literal charged economic regret. Long-horizon decision-preserving retention remains DEV-063 work.

See [separate preregistration](../research/dev061/retention-repair/protocol.json), [paired correction outcomes](../research/dev061/retention-repair/statistics.csv), [correction statistics](../research/dev061/retention-repair/statistics.json), and [paired completion](../research/dev061/retention-repair/paired-completion.png).

## Generalization stress tests

| Original arm | Unseen categorical IDs | 10% noisy initial bits | 10% noisy receipt bits |
|---|---:|---:|---:|
| generic | 35.2% | 81.2% | 93.9% |
| entity | 49.8% | 83.9% | 97.1% |
| entity_aging | 49.4% | 84.3% | 98.2% |
| entity_retention | 48.8% | 84.0% | 97.8% |
| entity_aging_retention | 50.4% | 84.0% | 97.5% |
| structured_gru | 50.0% | 84.2% | 98.0% |
| public_mlp | 100.0% | 84.4% | 99.4% |

These are separate stress strata, not pooled primary outcomes. Learned TAC variants reach only about 49–50% completion on unfamiliar categorical IDs despite programmed ID routing; this is a major transfer failure. The MLP public parser generalizes across these IDs because its feature rule is explicit. The old noisy-receipt schema still leaks changed-entity information and lacks adequate action provenance. These noise tests are **not a valid hidden-belief/POMDP demonstration**. Missing initial states, continuous unfamiliar identities and cancellation schemas are unsupported/unvalidated here.

## Audits, portability and compute

All **104 registered checkpoints** restore successfully (84 original including pilots, 20 corrected). The local full suite passes **1,395 tests**. The two separate audits replay **72,384 trajectories and 248,747 native actions**, including 6,483 unsuccessful rollouts across all arms/splits. Cached versus incremental execution agrees on **46,080 action choices**. Ninety fresh-model standalone checks cover 899 decisions; the five downloaded research candidates also pass the isolated published-bundle check. Original and correction replay telemetry have maximum local float difference zero. The correction additionally checks 301 retained timestamp identities in training-mode cached states; separate regression fixtures cover exact forward gates and matching gradients.

See [original audit](../research/dev061/audit.json), [correction audit](../research/dev061/retention-repair/audit.json), [archive hashes](../research/dev061/evidence-index.json), [experimental inference bundle](../research/dev061/learned-agent.tar.gz) and [complete evidence](../research/dev061/evidence.tar.gz). The 28 preliminary smoke-only fits are separately archived and excluded from every scientific count/comparison.

One-thread CPU timing uses predeclared seed 9101, H128, batch1, seven measured repeats after two warmups. Observation processing and the decision with its auxiliary probe are timed separately; end-to-end has its own repeats, so medians need not add. Other batch sizes, full samples, parameter counts and limited profiler-supported FLOPs remain in [original timing](../research/dev061/benchmark.json) and [correction timing](../research/dev061/retention-repair/benchmark.json). This is a narrow local profile, not production throughput or an equal-FLOP comparison.

| Model | Observe ms | Decision ms | End-to-end ms | State tensor bytes/world |
|---|---:|---:|---:|---:|
| Original entity | 7.224 | 2.222 | 9.474 | 8964 |
| Original entity_aging | 8.403 | 2.470 | 14.058 | 8964 |
| Original entity_aging_retention | 7.797 | 2.454 | 11.016 | 8964 |
| Original entity_retention | 7.367 | 2.312 | 9.630 | 8964 |
| Original generic | 9.310 | 2.053 | 12.275 | 9088 |
| Original public_mlp | 2.297 | 0.368 | 2.908 | 1472 |
| Original structured_gru | 7.370 | 1.417 | 8.903 | 10756 |
| Corrected entity_aging_retention | 7.839 | 2.497 | 11.153 | 8964 |
| Corrected entity_retention | 8.150 | 2.360 | 10.409 | 8964 |
| Incremental public rule | 0.051 | 0.008 | 0.059 | not tensor-state matched |

There is no demonstrated TAC runtime advantage over the MLP or optimized incremental public rule here. The public control has different representation priors and Python/NumPy implementation; its timing is a strong task-specific reference, not a general hardware lower bound. Input conversion is excluded for both Torch and public control. State bytes exclude weights, intermediates, allocator overhead and peak process memory. [Hardware constraints](../research/dev061/hardware.json): AMD EPYC host, two-CPU quota, 8GiB memory limit, one Torch thread per worker. **Physical DRAM traffic and peak process memory remain UNKNOWN.**

**Sparse-equivalence limit discovered by the broader sweep:** the frozen DEV-058 dense/sparse pair loses bitwise state equality in three of twelve batch/acceptance settings (maximum state difference 9.54e−7). All twelve probe action choices and occupancies agree exactly. At batch32/10% signals, predicted outcomes differ by up to 0.00016975 and fail the unchanged atol1e−4/rtol1e−5 bound. This case remains `bounded_equivalence_pass=false`; the aggregate equivalence gate is false. Raw latency ratios for failed cases are not qualified as output-preserving speedups. The initial bitwise and bounded-output assertion failures are retained in the evidence directory. No runtime weights or old archived results were changed. This limits generalization of earlier exact-equivalence claims beyond the recorded DEV-060 cases; it does not invalidate the separate exact DEV-061 rollout audits.

Admission cost remains linear in arriving observations. Logical write suppression is not proportional physical compute/DRAM savings. The explicit public-type/latest filter timings are controls; the latest filter changes retained history and has no claimed learned-policy equivalence.

Standalone inference bundles contain PyTorch weights and serving code without importing TAC-OSM, Veritas, NumPy, the simulator or teacher. The primary download uses the predeclared seed 9101 for five TAC research candidates; it excludes scripted MLP and GRU controls and the original flawed retention weights, which remain in full evidence. Corrected retention candidates are explicitly experimental and fail promotion. Standard-library plus PyTorch portability is not raw-language or multimodal task generality.

Native replay checks recorded actions against the pinned Veritas runtime and recorded synthetic effects. It verifies accounting and state consistency under those effects, not independent physical laws, industrial sensors, operational authorization, or real-world safety. Aggregate replay failures include all controls, pilots and splits; they are not failures of a single selected complete agent.

## Next research gate

Do not add depth or combine all proposed components to conceal these limits. DEV-062 is a draft, not registered/executed: first repair the observation/action provenance schema and establish an information-matched posterior baseline; then train belief reconstruction and charged information acquisition under genuine partial observability. Stage posterior imitation separately from learning without posterior targets. Explicitly separate information, representation and decision limits.

DEV-063 retention, DEV-064 causal prediction and the named same-feature tuned-MLP CASM deadline, DEV-065 training stability, and DEV-066 integration remain independent gates. MTSK superiority pursuit stays retired; C5 indexed acquisition, equal-bits/hash controls, continuous entity association and general symbolic CASM discovery remain open.
