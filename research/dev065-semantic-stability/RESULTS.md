# DEV065 — semantic stability and runtime batching

**NO PROMOTION.** All120fresh fits and12pilots completed; the primary recipe fails completion, physical reliability, both calibration tails and competitive-cost qualification. Independent replay and closed-loop substitution completed successfully; their final records are linked below. The commercial goal remains ACTIVE.

## Registered results

Primary: `tac_isolated_mixed`. Twenty seeds96501–96520, new heldout namespaces, four raw-history strata32/128/512/1024 and two secondary conditions. All arms use the semantic-only actor; there is no parallel history-latent input to action selection. The four TAC arms form the registered gradient-isolation × exact-training-semantic-exposure factorial. GRU/no-CASM are independently retrained isolated/mixed controls.

| Arm | Seeds ≥95% in every primary H | Completion | Primary final fault bits | Secondary fault bits | Mean cost | Median fixed p95 KL | Median fixed p99 KL |
|---|---:|---:|---:|---:|---:|---:|---:|
| tac_coupled_predicted | 20/20 | 100.000% | 0 | 0 | 5.473496 | 0.119409 | 0.365214 |
| tac_coupled_mixed | 20/20 | 99.688% | 0 | 0 | 5.442270 | 0.089811 | 0.299637 |
| tac_isolated_predicted | 3/20 | 76.953% | 0 | 20 | 5.549273 | 0.084236 | 0.235303 |
| tac_isolated_mixed | 4/20 | 78.047% | 0 | 20 | 5.538385 | 0.082467 | 0.254629 |
| gru_isolated_mixed | 5/20 | 74.941% | 0 | 21 | 5.573729 | 0.112653 | 0.349486 |
| no_casm_isolated_mixed | 2/20 | 68.555% | 0 | 20 | 5.623990 | 0.073967 | 0.262677 |

The primary completed3996/5120rollouts (78.0469%). Every primary rollout ended physically repaired, but1124ran out of budget without stopping. Both metrics matter: this is closure/efficiency failure, not1124broken machines. Its20secondary fault bits occurred under sensor error.2; the higher inspection-price condition had zero. These are repeated-seed outcomes on shared worlds, not20independent machines or field trials.

No arm passes the fixed-query tail limits: p95≤.05, p99≤.1. No individual seed in any arm passes both. The primary has4/20successful primary-history seeds rather than the required18, and zero seeds pass the combined per-world physical guard. Its secondary completion averages are98.3594% at(error.2,price.1) and99.375% at(.1,.4), but completion cannot waive physical faults.

Gradient isolation reduces paired-seed p99 KL by.070395 on average (95% bootstrap[-.104868,-.036865], Holm p=.005674), but increases cost by.085946([.027593,.151208], Holm p=.045925). This is a rejected trade-off, not a successful repair. The primary’s median p99 improves about30.3% versus coupled/predicted, satisfying the relative mechanism-gain gate while failing absolute calibration and reliability.

Exact-semantic training exposure reduces mean cost by.021058([-.027475,-.014758], Holm p=.0000572); its p99 contrast is inconclusive. The six cost/tail factorial contrasts receive Holm correction. Mixing also removes encoder policy gradients on exact-input rows, and global gradient clipping can couple update magnitude between blocks despite detachment. Interpret these as controlled recipe contrasts, not a complete causal account of optimization.

Primary minus GRU cost is-.035344, paired-seed95%[-.109672,+.042034]; its upper bound misses the registered+.02noninferiority tolerance. Primary minus no-CASM is-.085605[-.160440,-.008535], Holm p=.090363. These figures do not establish CASM superiority. The full table and all paired seeds are in [statistics.json](statistics.json) and [statistics.csv](statistics.csv).

## Information, representation and decision limits

The public known-law reference completes256/256primary rollouts with zero faults at cost5.485625. The privileged physical-state reference costs4.860312. The public last-record shortcut completes67.1875% with104fault bits at cost12.995. The public reference has a depth-two planning rule and near-certainty stopping guard; it is not a globally optimal long-horizon oracle. The gap cannot be labeled pure neural-representation regret.

All four TAC arms have identical validation KL at warmup completion for each paired seed. Median validation64 KL moves from.021566 at warmup to.032806(coupled/predicted),.021397(coupled/mixed),.013212(isolated/predicted) and.013991(isolated/mixed). Better average belief fitting did not yield reliable stopping. Full curves, all query contexts and stopping diagnostics are in [diagnostics.json](diagnostics.json), [learning-curves.png](learning-curves.png) and [per-seed.png](per-seed.png). Diagnostics are posthoc, not new selection rules.

Closed-loop predicted/exact-public/blurred posterior substitutions use fresh namespace6651006, each following its own visited history. Their completion, faults and costs belong to separate modes; aggregate damaged-agent faults must not be attributed to the complete model. See [interventions.json](interventions.json). The primary completes87.8125% in predicted mode and87.3047% under exact substitution; exact-minus-predicted cost is+.011164, paired-seed95%[+.005569,+.017108]. Both have zero final faults on these fresh intervention worlds. Exact beliefs do not repair this actor. Blurred mode has zero completions and439fault bits, which belong to the intentionally damaged variant. Action changes alone do not measure regret, and exact public beliefs are never deployed.

## What is learned and what remains supplied

Learned weights determine history encoding, CDL-style two-bank relevance, neural CASM mixtures, the12-class health×graph belief, action consequences/costs, action choice and STOP. The actor consumes only that belief and current public choice/price/identity-binding features. The exported model has no Bayesian update, repair law, analytical controller, scripted retry or STOP override.

Supplied structure remains explicit: exactly two entities, five actions, public16-vector identity keys with cosine binding, typed46-field records, a supervised six-type admission gate, FIFO64, the12-state vocabulary, two per-role GRUs reading bounded admitted history, and an event-gap clock. Neural CASM densely evaluates three MLP operators; it is not general symbolic program discovery or sparse indexed acquisition. The teacher uses exact public posteriors and known transitions for training labels. This is posterior-supervised learning, not autonomous discovery of physics.

The event clock is valid for this time-independent toy world; it cannot establish physical-duration invariance for industrial equipment. H1024mostly adds explicitly typed noise, not1024relevant facts. Training now contains both error laws.1/.2 for every arm, unlike DEV063, so historical improvement cannot be attributed solely to isolation or mixing. MTSK novelty remains retired. The separate DEV064 chained-task CASM deadline is unexecuted, with no superiority claim from this lane.

## Physical CPU optimization, independently qualified

The optional `runtime-batched.py` applies the unchanged learned admission gate to a chunk and performs FIFO insertion once. It never inserts a handwritten signal/noise rule or changes weights. Valid inputs require one-hot source6 records; invalid types are rejected. Original `runtime.py` remains available.

The predeclared first seed, all six arms, batches1/16 and histories32/128/512/1024 give48cases with3warmups and20measured repetitions each. Median CPU ingestion speedup is6.940×, range2.255–51.920×; every case is faster, passing the1.2×median/no>10%regression timing gate. Profiling used27.435CPU seconds. [runtime-profile.json](runtime-profile.json) retains every repetition, acceptance counts, input/retained storage and source hashes. All132restored checkpoints pass exact state parity; the original and batched standalone adapters each reproduce the same26logical decisions.

Every raw input is still scanned, and batching needs temporary input/gate tensors. This is an ingestion implementation improvement, not learned C5 retrieval, lower decision-network cost, a quality gain, measured DRAM traffic or a production SLA. At batch1/H128, the narrow concurrent CPU profile measured primary decision3.475ms, GRU3.075ms and no-CASM3.124ms; it does not support a TAC inference-speed advantage. [hardware.json](hardware.json) reports all cases and its limitations.

## Reproducibility and goal status

Fitting source/protocol commit: `1510f7ae2e299cc9087e55d4442cd1c048d73c3a`. Runtime profile source: `9ab10d7`. All models received2048updates, first1024belief-only; both pilot LRs were preserved, and validation selected.003for every arm. Total fit/preparation/selection CPU4159.051seconds, within7200; all seed groups completed within420CPU seconds. Final checkpoints are EMA.995; raw weights, all pilots, DAgger pools/trajectories, curves, failed outcomes and source commits are retained. Evaluation CPU was not separately instrumented by the frozen fitter; replay/counterfactual usage is recorded separately, rather than presenting fit CPU as total research cost.

Final regression:1472passed. Boundary checks cover policy gradients, fixed-belief exclusion of history latents, clock invariance, numerical compatibility, FIFO overflow, masks and adapter inputs. Independent replay restored132checkpoints,50304trajectories,11353native Veritas actions and26portable decisions per adapter;30,118,754telemetry values have zero local drift. It also recomputed validation metrics/temperature selection and264clock-invariance cases. Audit process usage was721.245CPU seconds; closed-loop substitutions used590.893CPU seconds including process startup. Remote replay remains pending until final-head checks. Records are in [audit.json](audit.json), [local-verification.json](local-verification.json), the evidence index and the draft PR checks. Raw checkpoints are hashed/preserved; strict policy replay concerns registered EMA checkpoints. Native effects derive from the same synthetic physics and are engineering reproducibility, not independent real-world validation.

Neither model qualification nor the commercial goal passes. REAL-MAINT-002 remains failed and its July B holdout remains unopened. The source-only LTPP follow-up found just4events with both condition measurements within90days and no other recorded maintenance/rehabilitation in between; road roughness and component costs cannot qualify a factory-repair controller. Real inspection prices, actions, independent physical outcomes and out-of-time/site validation remain unmet. Prior models/evidence and draft PR222 stay intact.

Use [REPRODUCE.md](REPRODUCE.md) for recorded replay. A failed registration closes here; do not tune these opened worlds, select another seed/arm from test results, or weaken the gates. Any next candidate requires a new protocol and untouched evaluation.

Posthoc stopping context:388of1124primary noncompletions had a publicly certain healthy decision opportunity;736did not. Across2757publicly certain healthy non-STOP decisions, median predicted healthy mass was.999848. This distinguishes decision failure from insufficient information acquisition; neither observation alone proves a cause. See NEXT-DECISION.md for the next falsifiable question.
