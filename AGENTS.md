# Agent guidance

For substantive engineering, product or research tasks, read `.agents/skills/fpc-universal/SKILL.md` and then the relevant repository-local instructions, specs, README, workflows and tests. Repository-specific safety, release, data, science and architecture requirements always override general skill guidance. Never assume a skill grants permission to access secrets, incur compute costs, deploy, merge or make external changes. Use a task branch and verify before submitting a PR.


## Mandatory TAC-OSM real-research gateway (new model work)

For any proposed TAC-OSM model-training or architecture-comparison experiment,
read `research/REAL-WORLD-EVIDENCE-RULES.md` and
`research/CAPABILITY-FIRST-RESEARCH-RESET-2026-10-22.md` first. Do not begin
new ME/DEV training when the machine-checkable preregistered
`research/mathematical-engineering/<study>/FAIR_COMPARISON.json` is absent
or fails `scripts/validate_research_parity.py --phase preflight`. Require
original matched trainable parameters, real data/order, training and tuning
budgets, and identical evaluation conditions before any architecture winner
is discussed; report CPU time as an efficiency metric, not a parity gate;
run `--phase final` on honest resource receipts after training. A text caveat
is not parity. Research novelty starts with peer-reviewed prior-art reuse and
one unresolved capability objective, not a small recurrent-memory ablation.
Legacy studies without this gate are immutable historical evidence, NOT
prospectively qualified superiority claims. Keep work in draft branches.


**Scientific-rule clarification (2026-10-22):** CPU wall-clock/process time
is an efficiency measurement and is never required to match between
architectures. The compulsory fair-comparison requirements are model
parameters, identical real training data/exposure, matched training/tuning
budgets and equal evaluation conditions; see the latest amendment to
`research/REAL-WORLD-EVIDENCE-RULES.md`. Historical CPU observations are
preserved but do not make a comparison scientifically invalid by themselves.


## Binding 100k trained-model scale (effective 2026-10-23)

For EVERY future trained TAC-OSM neural model or comparator (including event/robot models, one-off fits and each seed), require **100,000 trainable parameters ±5% inclusive (95,000–105,000)** AND the same band for all deployed learned model parameters. For matched architecture studies, independently require **≤1% relative parameter gap between every trained arm** on exactly matching real data/budget. Count ALL active encoder, fast/slow/associative memory, concept, shared-weight decoder, routing, action, forecast, uncertainty and policy heads; no dummy padding or bypass with a frozen pretrained encoder. Measure the actual instantiated model **BEFORE the first optimizer step** and remeasure the actual final model at the postfit gate. Use `scripts/validate_research_parity.py --manifest ... --phase preflight` and `--phase final`, or `--model-parameters COUNT --deployed-parameters COUNT` for a standalone prefit. For solo training use `study_type: single_model_training`; prospective diagnostics are explicitly no-fit. If outside the band or missing receipts, DO NOT TRAIN. Do not relax the matched data, training compute, checkpoints/replay or real-world evaluation obligations, and do not require equal CPU time. Historical unchanged ME001–ME072/DEV001–DEV061 experiments and their existing model bytes remain as originally measured; any NEW trained fit belongs to this prospective rule regardless of recycled old code or paths. See rule 12 in `research/REAL-WORLD-EVIDENCE-RULES.md`.

## Compulsory cross-repository mathematical-engineering rule — 2026-10-24

For **every** TAC-OSM variant or new computational-graph research candidate, first read [research/PNDS-MATHEMATICAL-ENGINEERING-RULE.md](research/PNDS-MATHEMATICAL-ENGINEERING-RULE.md), the canonical [PNDS mathematics/formulation policy](https://github.com/FPC-effortless/pnds-lean/blob/main/foundation/v1/MATHEMATICAL_ENGINEERING_RULE.md) and the source-pinned reciprocal [research/pnds-variant-math-links.json](research/pnds-variant-math-links.json). The PNDS model/math record must contain **both Lean 4 and executable Python mathematical formulations** matched to this repository's exact variant, implementation commit and model input-to-output computation graph.

Do not present Lean compilation, shared operator bounds or Python reference tests as source-equivalent full TAC-OSM behavior without full actual-source forward/backward, state, shape, time, mask, precision and gradient verification. Declare incomplete properties OPEN. Keep immutable original data/checkpoints, authentic research scores, failed gates, matching parameter/data/training-compute comparative controls and capability-first priorities. This is **not** a mandate to stop real-capability research for numerous small ablations.


## Current shared-core trajectory evidence — 2026-10-25

Read the master-plan branch and ME098/ME099 original results before selecting the
next architecture study. Six active families share one candidate factory; only
continuous physical forecasting was trained in these two studies. ME098 learns
coherent 1/4/16-frame means but fails uncertainty coverage in every seed. ME099
adds a shared uncertainty regime and proper interval training: four of five seeds
pass, but seed 99001 has 84.192562% four-frame coverage versus the unchanged 85%
minimum. FULL GATE FAIL. Do not select favorable seeds, round into a pass, alter
thresholds, overwrite originals or silently repeat either protocol. This bounded
correction cycle authorizes no additional fits after scoring.

The source-pinned PNDS PR10/11 references and kernel proofs pass; full neural and
IEEE refinement, independent cohorts, useful long memory and action/customer
utility remain OPEN. Both studies preserve original Git models/optimizer/RNG and
independent score replay. Capability scorecards report MIXED_TRADEOFF. Continue
substantial capability development under a new prospective protocol on qualified
real evidence, following the current master plan; attribution and CPU tuning are
secondary. The current primitive mean law uses recorded frame counts, not proven
irregular-time physics. No Mamba performance equivalence, whole-network proof,
new active family, authority, deployment or automatic merge is established.


## Binding research anti-cheat and diagnosis policy (effective 2026-10-26)

Every NEW TAC-OSM variant/training/evaluation must register an explicit
prediction-time information boundary and a source-to-output leakage threat
model. Read research/ANTICHEAT-DIAGNOSTICS-PROTOCOL-2026-10-26.md and run the
appropriate data partition, transform-fit, future-target poison, deterministic
prediction, checkpoint, gradient, and activation checks at each actually
implemented pipeline boundary. Preserve per-check PASS / FAIL / INCONCLUSIVE
records and output identities/receipts. Missing, inapplicable, unstable or
uninstrumented checks are UNKNOWN/INCONCLUSIVE, never automatically PASS.
A passed constructed fixture or source file hash is engineering evidence, not
proof of real-world model intelligence or source authenticity. Do not promote
an architecture or unseal TEST on these grounds. Real-world, matched-budget,
100k and PNDS rules remain binding. Audit existing physical-window adapters
using src/tac_osm/robot_anticheat.py; other variants MUST integrate their
own actual graph interfaces before claiming end-to-end information isolation.
