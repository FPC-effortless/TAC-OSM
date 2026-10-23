# Agent guidance

For substantive engineering, product or research tasks, read `.agents/skills/fpc-universal/SKILL.md` and then the relevant repository-local instructions, specs, README, workflows and tests. Repository-specific safety, release, data, science and architecture requirements always override general skill guidance. Never assume a skill grants permission to access secrets, incur compute costs, deploy, merge or make external changes. Use a task branch and verify before submitting a PR.


## TAC-OSM capability priority — owner direction 2026-10-23

**Develop real learned abilities first; attribution is optional diagnosis, not the goal.** For the next TAC model iteration, prefer one **integrated native world model** that is actually trained, independently replayed and evaluated on authentic observed outcomes for *new useful capabilities*: multi-horizon prediction, uncertain timing, carried and corrected entity state, missingness resilience, and later action/outcome consequences if factual intervention records exist. Use proven mathematical techniques and existing research rather than many tiny diagnostic experiments. Prioritize genuine end-to-end ability over proving which memory cell caused it, CPU micro-optimization, or a new GRU comparison every iteration.

A single-architecture `capability_development` preregistration **may train and report its own authentic developmental scores** without a matched ablation/comparator matrix, but **may not claim architecture superiority or memory-module attribution**. Only require equal parameter/data/exposure/compute and matched competitors when asserting a comparative winner. The prior 64-event kill rule concerns a narrow TAC-memory-*superiority* claim, not permission to abandon full-model capability development. Never call an implementation unit test a learned skill, use synthetics to claim real-world success, open a sealed TEST for tuning, or skip in-repository original checkpoints and independent clean-clone replay. See `research/MASTER_RESEARCH_PLAN.md` owner-priority section.

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

## Mandatory in-repository checkpoints and replay (2026-10-23)

Before claiming **any** locally trained or externally trained (GitHub Actions/cloud/notebook) TAC-OSM study complete, require a permanent **repository-backed trained checkpoint for every arm/seed** and all lawful loader, environment, configuration, prediction, scoring, data-source and integrity materials needed to reproduce the original experiment. Use the canonical [repository replay and checkpoint policy](research/REPOSITORY-REPLAY-AND-CHECKPOINT-RULE.md). Temporary CI artifacts, local results folders, hash-only receipts, external links and "can be retrained" explanations are insufficient. Preserve optimizer/RNG and sampler states when training continuation or full training replay is claimed. Validate actual tracked/LFS bytes and independent **clean-clone restore + scoring replay** before results/promotion; explicit `REPO_REPLAY=FAIL/BLOCKED` if absent. No secrets or unlicensed raw data in Git. Existing gitignored local checkpoint scripts are historical only, not a permissible delivery mechanism for new work. This preservation rule is independent of and cumulative with real-data, data/parameter/compute parity and experimental integrity gates.

