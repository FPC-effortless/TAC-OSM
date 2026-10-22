# Agent guidance

For substantive engineering, product or research tasks, read `.agents/skills/fpc-universal/SKILL.md` and then the relevant repository-local instructions, specs, README, workflows and tests. Repository-specific safety, release, data, science and architecture requirements always override general skill guidance. Never assume a skill grants permission to access secrets, incur compute costs, deploy, merge or make external changes. Use a task branch and verify before submitting a PR.


## Mandatory TAC-OSM real-research gateway (new model work)

For any proposed TAC-OSM model-training or architecture-comparison experiment,
read `research/REAL-WORLD-EVIDENCE-RULES.md` and
`research/CAPABILITY-FIRST-RESEARCH-RESET-2026-10-22.md` first. Do not begin
new ME/DEV training when the machine-checkable preregistered
`research/mathematical-engineering/<study>/FAIR_COMPARISON.json` is absent
or fails `scripts/validate_research_parity.py --phase preflight`. Require
original full-parameter, original real-data/order, actual FLOPs and CPU,
tuning and evaluation equality before any architecture winner is discussed;
run `--phase final` on honest resource receipts after training. A text caveat
is not parity. Research novelty starts with peer-reviewed prior-art reuse and
one unresolved capability objective, not a small recurrent-memory ablation.
Legacy studies without this gate are immutable historical evidence, NOT
prospectively qualified superiority claims. Keep work in draft branches.
