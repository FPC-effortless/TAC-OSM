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

## Compulsory live enforcement (2026-10-23)

All fitting paths, including capability development and source-deferred studies,
use `src/tac_osm/parameter_budget.py`. Construct optimizers with
`budgeted_optimizer(whole_model, optimizer_type, ..., budget_manifest=...)`,
update with `checked_optimizer_step`, and persist originals with
`save_training_checkpoint`. Prefer `scripts/guarded_research.py` as the launcher.
The guard measures the actual whole model before any optimizer update, checks
both counts against the preregistered arm/seed, writes a durable receipt before
the first update, and binds saved model tensors to it. Final validation requires
that receipt and the actual checkpoint, not declared counts alone. Run
`scripts/audit_training_budget.py` when adding entrypoints; CI rejects raw
PyTorch constructors, updates and saves in supported repository training code.
A reused historical study ID never exempts a new fit. Replay-only exceptions
require an explicit original registration, full Git commit and checkpoint hashes
in `research/governance/FROZEN_REPLAY_REGISTRY.json`. They authorize integrity
replay only. Do not rewrite old registrations to manufacture pre-update evidence.
See `research/governance/LIVE_PARAMETER_ENFORCEMENT.md` for the contract and limits.
