# Independently verified reward/outcome/access evidence for every TAC-OSM runner

Effective 2026-10-26; prospective model/variant-specific training and evaluation.
Implemented on draft PR #430. **No historical checkpoint is newly qualified.**

## Enforcement before fitting and before claiming a result

The shared research parity validator at
\`scripts/validate_research_parity.py\` now requires an externally signed
Ed25519 attestation for EVERY prospective new model fit, including a single
model and a fair architecture comparison. It is impossible to obtain
\`check_manifest(..., "preflight")\` or \`check_manifest(..., "final")\`
success for a new study without the independent verifier evidence.

The signed payload is bound to the frozen real source SHA256, study ID,
model family or architecture name and **every seed**. Final attestations
must also bind actual measured checkpoint hashes and a verified factual
outcome evidence hash/count for every model in the registration. Missing
one variant/seed, a tampered checkpoint, false signature, unknown signer,
mismatched source or absent required check => BLOCK.

### Required information by phase

| Time | Independently verified facts |
|---|---|
| **Before first fit** | Real-world source provenance, external goal evaluator separation, the actual actor-versus-auditor permission boundary, and read-back SHA256 verification of protected scoring/goal/source artifacts |
| **After trained evaluation, before promotion** | Same protected-artifact verification, reward recomputation, independent real-goal recomputation, decision-time reward-noninterference negative control, and independent real-goal/safety/decision-cost comparison against a preregistered baseline, including required groups |
| **Every variant + seed** | Registered model/architecture name, seed, same original real source, and final trained checkpoint SHA256 and independently verified outcome evidence digest/count |

A passing **signature** means the operator's pinned audit key signed a
receipt. It does NOT alone prove the signer is genuinely independent,
that outcomes are causal, or that every reward exploit was found. Private
signing keys must be stored and used only by a separate trusted evaluator
process or service, inaccessible to the training process/agent. The trusted
public verification key is injected by operator-controlled CI environment
or a separate protected configuration, never read from the PR, checkpoint,
experiment manifest or model-produced JSON.

### Preflight CLI — mandatory before running a new trained model

    export TAC_OSM_REWARD_VERIFIER_PUBLIC_KEY_HEX='<independently-pinned-operator-public-key>'
    python scripts/validate_research_parity.py \
      --manifest research/mathematical-engineering/<study>/FAIR_COMPARISON.json \
      --phase preflight \
      --reward-evidence /trusted-supervisor/prefit-attestation.json

Exit status 2 blocks the training job. Do not run a fit if preflight fails.

### Final CLI — mandatory before experiment qualification

    python scripts/validate_research_parity.py \
      --manifest research/mathematical-engineering/<study>/FAIR_COMPARISON.json \
      --phase final \
      --reward-evidence /trusted-supervisor/postfit-attestation.json

Exit status 2 blocks architectural superiority, real-world model improvement,
reward score promotion, or product readiness claims. Final evidence must come
from actual authorized FIT/DEV records, not constructed tasks or cached
model-generated metrics. TEST data remain sealed from iterative debugging.

### Trusted supervisor, not self-attested strings

\`src/tac_osm/research_reward_supervisor.py\` provides
\`issue_privileged_runner_attestation(...)\` to a PRIVILEGED evaluation
service. The caller must supply independently controlled source and access
\`Finding\` receipts, genuine audited scorer + independently controlled goal
judge, protected files and ACLs, real outcome rows, exact per-model source
outcome hashes and a private signing operation. It invokes the real
\`research_reward_integrity\` audit primitives, **refuses to sign any missing
or failed check**, and signs a canonical stage+study+variant-bound message.
Never run this supervisor inside a trainable model or expose its signing
callback / private signing key to model-writing agents.

The model or experiment optimizer can read the **public verification key**;
it cannot produce a trusted signature without the private signer. This is a
cryptographic evidence handoff; isolation requires actual separate OS/cloud
credentials and permission boundaries. A supervisor using the same uid,
filesystem write domain or evaluator implementation as the actor is **NOT**
independent, even if the digest/signature checks pass.

### Wiring *every prospective training workflow*

New ME/DEV registered workflows are rejected by
\`validate_research_parity.py --check-changed\` unless their source contains
both resource-parity hooks AND BOTH phase-specific reward evidence hooks
(\`--reward-evidence\` in prefit and final) and a trusted key variable.
Each architecture variant's actual source-authorized runner invokes those
commands before optimizer step 1 and before releasing any score/weight.

Important: the current GitHub repository contains no actual independent
reward/goal/ACL receipts for **all six trained variants**. Merely adding a
native/GRU/stateless test, passing a fixture, or listing six names does not
establish six integrated production runners. Those must each be connected
to their real original dataset, their actual scorer/decision graph, separate
external goal judge, and access-enforced evaluator service. Do not promote
any of those runs until this is complete.

## Six variant research registration (no silent green badges)

| Variant | Generic signed gate coverage | Authenticated real postfit reward audit |
|---|---|---|
| System 1 | Required by shared training validator | OPEN |
| Enterprise | Required by shared training validator | OPEN |
| Temporal | Required by shared training validator | OPEN |
| Physical | Required by shared training validator | OPEN |
| Edge-Lite | Required by shared training validator | OPEN |
| Counterfactual | Required by shared training validator | OPEN |

All six are exercised as distinct named contracts by a **constructed**
signature/variant/seed CI test. This proves missing variant slots are
rejected. It does not imply trained variants have already run those checks.

## CI and deliberate negative controls

- \`tests/test_reward_runner_integration.py\` verifies positive verifier
  round trips, spoofed signatures, mismatched source, wrong checkpoint, missing
  models, missing goal/reward/ACL checks, CLI missing trusted key, and actual
  privileged audit refusing to sign mismatched rewards or writable evaluators.
- These are no-fit unit tests. They use locally generated test keys and
  fabricated fixture observations to falsify broken proof enforcement.
  **Never** use such fixture keys, fixture source hashes, or test signals to
  report scientific qualification.
- \`.github/workflows/research-scientific-gate.yml\` and
  \`.github/workflows/research-anticheat.yml\` run the signed evidence tests;
  full \`.github/workflows/ci.yml\` also runs the entire suite.
- All historical research baselines and original trained checkpoints remain
  unchanged, preserving recorded failures and resource parity rules.

## Open integration requirements

Before any actual new family/seed can be declared reward-integrity qualified,
the operator must deploy an access-separated evaluator with external signing
keys, independent publisher source authentication, original outcomes with
identity/cutoff proof, a non-agent-controlled real goal evaluator, actual
scorer execution, protected-file read-back, verified ACL enforcement,
closed-loop reward poisoning, signed run/seed/checkpoint receipt production,
and PR/release status protections against bypass. No self-reported registry
can substitute for these independent checks.


### One-command guarded launcher for actual training runners

For each variant / seed training program, use the shared executable:

    python scripts/run_qualified_research.py \
      --prefit-manifest /frozen/FAIR_COMPARISON.json \
      --final-manifest /trusted-postfit/MEASURED_COMPARISON.json \
      --prefit-evidence /trusted-evaluator/prefit-signed.json \
      --final-evidence /trusted-evaluator/postfit-signed.json \
      -- python scripts/YOUR_AUTHORIZED_TRAIN_RUNNER.py --source real-FIT-data

The launcher checks signed source, goal-evaluator and access-control evidence
**before executing** its provided training command, without invoking a shell.
It checks that the frozen preregistration did not change during the run and
that the final document has the identical study/source/variant/seed matrix.
It checks the final scientific resource manifest plus independently signed
reward, goal, safety and checkpoint outcome evidence **after the command**.
Any missing/failed final report yields nonzero exit and no qualification.

This does not make the optimization itself safe or keep the actor from
deliberately bypassing the launcher. Authorized CI must use it as the
required entrypoint and separately enforce branch protection, least-privilege
evaluator storage and verifier-key custody. The final signed evaluator receipt
must be created by an independently permissioned source-authenticated service
after the actual evaluation, not by the model process or test fixture.
