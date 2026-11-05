

## Phone-first autonomous research lab compute and status contracts (2026-10-27)

Read CHARTER.md, CODEX.md and research/PHONE-FIRST-AUTONOMOUS-LAB-2026-10-27.md.
For ALL proposed training/evaluation experiments, agents MUST preregister a
reviewed research/phone_queue protocol and run scripts/phone_lab.py plan to
choose GitHub Actions for bounded CPU/no-fit tasks or Kaggle NvidiaTeslaT4
(EXACTLY TWO T4s) for justified CUDA/GPU speedup tasks. Never claim that
the planner authorizes training or reward evaluation. The Kaggle staging path
must be private, no internet, with approved source-provenance and explicit
two-way seed-shard/DDP; runtime must reject fewer than two physically observed
T4s. The user approves actual quota through the protected GitHub environment,
which is not configured by source control. Do not silently substitute other
accelerators or consume Kaggle minutes. Training and promotion still require
trusted independently signed source/outcome/reward/ACL attestations and full
checkpoint/replay, registered model scale, data/compute parity and sealed TEST policy.
Research agents may draft PRs but MUST NOT autonomously merge novel-model
research or represent no-fit diagnostics as intelligence. GitHub Issues and
Actions provide compact phone-first progress, blockers, and result evidence.

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


## Binding model-scale policy v2 (effective 2026-10-31)

For every genuinely NEW model study, preregister `model_scale_policy: declared-target-v2` and explicit integer trainable/deployed targets in `FAIR_COMPARISON.json`, with a fixed ±5% absolute band and scientific rationale. There is no global size ceiling. Count every learned encoder, memory, router, head and deployed pretrained/frozen weight; no dummy padding. Independently require **≤1% trainable/deployed parameter gap between matched arms**, identical authentic data/access, training/tuning budgets and measured compute. Measure actual instantiated counts BEFORE optimization and actual checkpoint counts after fitting. Run the full signed `scripts/validate_research_parity.py --manifest ... --phase preflight` and `--phase final` gates. A standalone `--model-parameters` check is count-only and NEVER authorizes training. Missing receipts or failed gates block fitting. CPU time is an efficiency result, not a parity requirement.

Historical registrations remain immutable: LANGMATH101 and existing v1 studies retain 95,000–105,000 trainable/deployed parameters; unchanged ME001–ME072/DEV001–DEV061 retain their original grandfathering. Never edit an old registration to adopt v2; use a new study ID. Real-source, signed independent evidence, PNDS, checkpoint/replay, sealed TEST and compute-execution requirements remain binding. See rule 12 in `research/REAL-WORLD-EVIDENCE-RULES.md`.

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
registered model-scale and PNDS rules remain binding. Audit existing physical-window adapters
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


## Binding model-independent reward-hacking prevention (effective 2026-10-26)

For every new TAC-OSM model, policy, training loop, benchmark or evaluation
that optimizes a reward/proxy metric, read
`research/REWARD-HACKING-GATE-2026-10-26.md`. Define an independently
sourced external goal and actor-vs-evaluator trust boundary BEFORE fit.
Do NOT treat increased proxy score as improved intelligence/real outcomes.
Recompute reward and goal independently, poison postdecision reward inputs,
verify protected evaluator files against externally frozen hashes and
out-of-band actual writer ACLs, and test aligned real outcomes including
per-group/horizon safety and decision costs. All five checks in
`research_reward_integrity.reward_audit_ledger` MUST be present and PASS
before model-improvement/safety claims or reward-based checkpoint promotion;
any FAIL, INCONCLUSIVE or missing check BLOCKS promotion. Fix the scorer,
leakage or harm before rerunning. Never claim no reward hacking from synthetic
CI fixtures; actual source-verified real FIT/DEV audit integration remains
a distinct prerequisite. Keep sealed TEST unopened for iterative debugging.

### Independent verifier required for ALL prospective reward-based study runners

The prospective `scripts/validate_research_parity.py --phase preflight` and
`--phase final` gates now require `--reward-evidence` plus an operator-injected
`TAC_OSM_REWARD_VERIFIER_PUBLIC_KEY_HEX`. This applies to single-model and
architecture-comparison registrations **for any TAC-OSM family or seed**.
Each registered model/seed must have signed, externally verified outcome
evidence tied to its exact trained checkpoint. Before fit, independently
verify original source, separated goal evaluator, protected scorer artifacts
and actual trainer-vs-evaluator ACL boundaries. After fit, independently
recompute proxy reward AND factual goals and check leakage, harm and costs,
then have an access-separated evaluator sign the complete findings. Missing
trusted signatures, real original outcomes, denied assessor access, changed
checkpoint, or incomplete family/seed coverage must BLOCK training/promotion.
Never sign inside the trainer or put the signing private key in repo secrets
visible to model agents. See
`research/REWARD-INDEPENDENT-RUNNER-GATE-2026-10-26.md`. No synthetic test
signature implies any of six variants has achieved real-world qualification.

## Optional simplified experiment lab (2026-10-27)

New and historical TAC-OSM workflows MAY index existing authorized FIT/DEV/diagnostic report files using `scripts/lab.py record` and view a static, offline dashboard using `scripts/lab.py report`. Use `research/LIGHTWEIGHT-LAB-2026-10-27.md` for the default GitHub/Python approach. The immutable lab receipt is *metadata only*, never a standalone source authority, checkpoint archive, fair-comparison gate, model capability result or reward-hacking certificate. Optional vendor/cloud adapters are disabled by default and must not upload originals, open TEST or incur compute charges without authorization. Preserve all existing experimental commands, source-attestation prerequisites, checkpoint-replay and PNDS Lean/Python rules. The native CI runs only no-fit fixture indexing.
