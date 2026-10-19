# SYS1–SYS2–001: Frozen-checkpoint capability diagnostics

**October 19, 2026 — Exploratory research only; NO PROMOTION.** The experiment uses synthetic engineering diagnostics, not real operational outcomes. In keeping with the real-world-evidence policy, no commercial or real-world System 1/2 capability is claimed.

## Operational definitions and scope

- System 1 proxy: low-latency reactive action selection, state-sensitive immediate behavior, candidate permutation invariance.
- System 2 would require correct action-consequence models, comparison of consequential *multi-step* alternatives, reliable adaptive planning/verification, and demonstrated improvement over strong reactive controls where verified oracle-versus-greedy headroom exists.
- Neural prediction-head dependence by itself is not multi-step search or System 2.

## Verified original trained checkpoints

- DEV067 final-EMA exported first seed **96701**: **tac_frozen_log**, **gru_frozen_log**, **no_casm_frozen_log**; original saved GitHub workflow artifact **12194964127** on run https://github.com/FPC-effortless/TAC-OSM/actions/runs/39271150621. The exported checkpoint metadata SHA256 matched the original model files. All loaded in CPU PyTorch; no weights updated.
- SEQ008 six original 400-update models: **tac_gated**, **routed_gru**, **global_gru**, two optimizer seeds **780820** and **780821**. Original artifact **12215367412** from run https://github.com/FPC-effortless/TAC-OSM/actions/runs/39309620973; original recorded-only audit passed in run https://github.com/FPC-effortless/TAC-OSM/actions/runs/39310750160. Checkpoint hashes verified. These are TAC-inspired component models trained from scratch, NOT complete maintenance-decision TAC-OSM controllers.
- No original reserved SEQ008 TEST namespace opened, checkpoint refitted, model promoted or merged.

## System 1: observed single-step policy behavior

| Frozen DEV067 arm | Exact saved demo replay | Healthy-state STOP | Faulty-state non-STOP | Candidate permutation agreement | Local single-thread decision median |
|---|---:|---:|---:|---:|---:|
| TAC | 4/4 | 3/3 | 9/9 | 120/120 | 2.97 ms |
| GRU | 3/3 | 3/3 | 9/9 | 120/120 | 2.47 ms |
| No-CASM | 5/5 | 3/3 | 9/9 | 120/120 | 2.84 ms |

All saved demo action indices/logits reproduced exactly on this host (zero recorded max difference). Reactive sanity checks used twelve manually supplied one-hot belief-state probes (four health states times three relations), not collected maintenance data or independent predicted beliefs. Each case underwent ten candidate-order permutations. CPU forward timing is local and illustrative, excluding JSON/process/external effects. Inspection-price changes 0.1→10 altered choices in 7/12 TAC, 6/12 GRU, 7/12 no-CASM inputs, including some unwanted healthy-state repairs under extrapolation. This is not evidence of calibrated economic rationality.

## System 2: learned consequence-head intervention

| DEV067 arm | Future-head removal changed selected action, 12 extreme beliefs | Changes among 269 broader constructed beliefs |
|---|---:|---:|
| TAC | 0/12 | **49/269 (18.22%)** |
| GRU | 0/12 | **20/269 (7.43%)** |
| No-CASM | 0/12 | **21/269 (7.81%)** |

The broader 269 intervention inputs included manually generated distributions and recorded demonstration beliefs; they are not a natural population and include out-of-training-manifold beliefs. Scores changing or decisions switching shows **causal reliance on an outcome-head feature**, NOT correct action predictions, lower actual regret, planning or chain-of-thought. The DEV067 exported inference itself is a single forward policy computation; there is no learned multi-step search loop, Bellman planner, external independent verifier, or symbolic CASM program executor.

On a separate tiny deterministic two-entity parent/child repair fixture supplied with **perfect oracle world state after every step**, all three frozen actor heads completed **12/12** cases (up to three decisions each). This may be explained by repeatedly invoking a reactive policy. The external oracle-state injection, identical GRU/no-CASM success, and tiny known-rule world prevent any System-2-specific inference.

Previous independent work in [DEV064B PR #231](https://github.com/FPC-effortless/TAC-OSM/pull/231) registered ~99.2% initial action accuracy but ~37.3% native closed-loop STOP-and-healthy completion, with its cost oracle also ~37.4%. That earlier study was validation-rejected and was **not rerun** or counted as a new result here.

## Latest SEQ008 frozen model: delayed entity state updates

A separately generated, balanced diagnostic set has **120 distinct examples per length**, exactly 40 each for late overwrite, repeated toggle and noncancelling two-entity XOR (20 per class). Unique probe source seeds **951859 / 951987 / 952243**, not the original reserved SEQ008 TEST seeds. The last target-entity informative event precedes the readout by at least **39 / 77 / 154** steps for L128 / L256 / L512. Both original training initializations were evaluated on the same new examples per length.

| Frozen checkpoint family, mean over two seeds | L128 | L256 | L512 |
|---|---:|---:|---:|
| TAC-inspired gated recurrence | 89.58% | 88.33% | 89.17% |
| Independently trained entity-routed GRU | **96.25%** | **93.33%** | **95.83%** |
| Global GRU | 47.50% | 46.67% | 47.50% |

TAC-inspired L512 results were **79.17% vs 99.17%** for its two seeds, while independently trained entity-routed GRU scored **95.83% in both**. This measures a narrow deterministic, publicly identity-routed sequence updating skill, not the ability to select real actions or reason over alternative future plans. These synthetic tasks cannot establish real-world model reliability.

## Verdict, limits and reproducibility

- **System 1: demonstrated in a bounded proxy sense.** The trained full TAC controller responds quickly and produces consistent state-sensitive actions. No unique TAC improvement over the GRU/no-CASM controls was found.
- **System 2: not yet demonstrated.** Future-feature dependence exists, but no tested full learned search/deliberation loop, correct multi-step consequence model, verification-driven correction, or significant planning-specific control advantage has been established.
- **Real-world capabilities: unqualified.** The new SCANIA source-audit [PR #238](https://github.com/FPC-effortless/TAC-OSM/pull/238) concerns acquisition/lineage only, not real-action model evaluation. Research promotion requires authentic temporal observations, observed actions, verified outcomes, real costs, proper holdouts and strong reactive/planning baselines.

Original unchanged checkpoint artifacts and standalone executable local diagnostic scripts are provided in the independent conversation ZIP **TAC_OSM_System1_System2_Checkpoint_Audit_2026-10-19.zip**, containing README, original export files, SHA256 evidence, scripts, and result JSON. This pull request commits the read-only report and summary, **not** full experimental code; it does not claim new GitHub CI or independent real-world replication. Keep draft/unmerged.