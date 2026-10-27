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


## Mandatory architecture-agnostic diagnostics contract (2026-10-26)

The anti-cheat, architecture introspection, memory, gradient, information-flow,
entity-isolation, timing and evidence comparison tools are **cross-version
research infrastructure**, NOT native architecture features. For every new
model family/version, register a source-reviewed `ResearchProbeAdapter` from
`src/tac_osm/research_probe.py` declaring actual source-time inputs,
state ownership, prediction heads, optional legal candidate semantics,
per-entity splitting and local AD graph. Never infer or reuse native
fast/slow/matrix shapes for another model. Missing adapters or unimplemented
probe dimensions are **INCONCLUSIVE**, never PASS. The native and robot
specific entrypoints remain clearly labeled compatibility adapters. Run
`tests/test_research_probe_agnostic.py` in automatic research CI.
Do not equate cross-model fixture coverage with trained-checkpoint or real-data
qualification; these require direct source-to-output end-to-end audits.


## Optional simplified experiment lab (2026-10-27)

New and historical TAC-OSM workflows MAY index existing authorized FIT/DEV/diagnostic report files using `scripts/lab.py record` and view a static, offline dashboard using `scripts/lab.py report`. Use `research/LIGHTWEIGHT-LAB-2026-10-27.md` for the default GitHub/Python approach. The immutable lab receipt is *metadata only*, never a standalone source authority, checkpoint archive, fair-comparison gate, model capability result or reward-hacking certificate. Optional vendor/cloud adapters are disabled by default and must not upload originals, open TEST or incur compute charges without authorization. Preserve all existing experimental commands, source-attestation prerequisites, checkpoint-replay and PNDS Lean/Python rules. The native CI runs only no-fit fixture indexing.
