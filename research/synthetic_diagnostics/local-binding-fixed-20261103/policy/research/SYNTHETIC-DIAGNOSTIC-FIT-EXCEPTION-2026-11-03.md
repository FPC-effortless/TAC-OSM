# Narrow prospective synthetic diagnostic-fit exception

**Scope approved by the user/operator on 2026-11-03.** This document records
that category-level decision, not approval of unseen implementation bytes or
permission to execute a particular fit. Its coordinated implementation,
source review and each bounded execution release remain separate steps.

This amendment is additive to the canonical research policy at commit
`9828e5a61e8dff128248c35b284c32a2a146c63e`. Historical policy copies, C12/C18
registrations, failed gates, model sources, checkpoints and observations stay
immutable. No historical identifier or previous campaign grant authorizes a
new fit. Publish a new study/version and pin its approved policy snapshot.

## Purpose and evidence boundary

The new `study_type: synthetic_diagnostic_fit` permits a bounded, single-seed
optimization or mechanism screen on explicitly generated data to answer a
concrete implementation or learnability question before real-data research.
Every result is **synthetic, single-seed, exploratory, noncomparative and not
independently qualified**. Fitting fixed cases is optimization evidence only.
A finite held-out generated battery supports only the conditional behavior
actually measured, not population generalization or real-world capability.

Multiple trained controls are allowed only to diagnose a preregistered
mechanism. They cannot rank or select architectures, name an architecture
winner, promote a checkpoint, justify deployment, or establish superiority,
safety, reliability, commercial readiness or real-world qualification. Do not
use a synthetic leaderboard, score ordering or best-checkpoint selection as a
back door to those claims. Any architecture-selection or superiority question
must use the existing real-source, full-comparison and signed qualification
paths instead. Synthetic evidence has zero primary promotion weight.

`noncomparative_diagnostic` remains strictly no-fit. Neither
`single_model_training` nor `architecture_comparison` gains an
`allow_synthetic` switch, evidence waiver or changed acceptance condition.

## Explicitly limited exceptions

Only this new category, after all admission steps below, is excepted from:

- The synthetic-only fitting prohibitions in requirements 2 and 9 of
  `REAL-WORLD-EVIDENCE-RULES.md`. Historical studies still cannot be rewritten
  or silently extended; use a new registration.
- The no-fit restriction for prospective diagnostics. It remains binding on
  the existing `noncomparative_diagnostic` category.
- Treating a non-ranking mechanism screen with several controls as an
  eligible architecture comparison. All controls still require 100k scale,
  within-1% parameters and matched exposure, updates and selection privileges.
  This exception does not assert training-FLOP parity or comparable speed.
- The general phone-queue/planner and GitHub Actions routing requirement for
  CPU fits. A reviewed, bounded **local CPU** execution is an explicit routing
  exception only for this category, under its exact one-shot release. Its
  frozen manifest records the route and finite CPU, wall-time, memory and
  numerical-call limits. Source-only registration checks run in CI; CI passing
  is not a compute or execution grant.
- The real-source signed reward-runner path as a prerequisite for collecting
  this unqualified synthetic diagnostic alone. The new admission path is not
  an independent real-outcome attestation. All original signed-source,
  reward, goal, evaluator and ACL gates remain unchanged for original study
  types and any qualification, promotion or real-world claim.

No protected TEST, real corpus, private checkpoint, GPU, paid compute,
provider-account, publication, merge or deployment permission is created.
The existing anti-cheat, actual-source introspection, checkpoint/replay,
mathematical-formulation and honest-evidence obligations remain applicable.
An unimplemented or unresolved check is INCONCLUSIVE, never automatically
PASS. This policy-only amendment does not change a model graph or close an
open PNDS Lean/Python obligation.

## Admission and authority, before numerical work

Use three separate records with acyclic hash bindings:

1. **External scope approval.** The trusted root execution controller verifies
   the actual operator approval and its narrow limits from authenticated
   governance/conversation evidence. It supplies the approved receipt bytes
   and trusted receipt SHA-256 from outside the candidate and manifest. Public
   source documents record the scope and date, not private conversation IDs.
2. **Independent exact implementation review.** Review the canonical policy
   change, source-only regressions, exact candidate and protocol, data bytes,
   runner, admission helper, runtime and wrapper. Record immutable hashes,
   limits, unresolved checks and the review outcome. Category approval is not
   approval of these previously unseen bytes. No implementation may review
   itself into authority.
3. **One-shot bounded release.** Only after review, the trusted root controller
   issues a separate grant binding the approval, policy, review, candidate,
   protocol, stage/regime, arm/family set, single seed, full update/exposure
   budget, call limits, resource ceilings and exact output destination. The
   grant's trusted digest is injected externally too. The external controller
   enforces one-shot consumption; passing a stateless validator cannot prevent
   replay. No autonomous retry or extension is allowed.

A manifest boolean, self-derived hash, model-written signature or unrelated
historical grant is not approval. Hash and schema checks establish consistency
with externally supplied trust anchors, **not cryptographic proof of user
identity or authorship**. Keep scope approval, implementation/protocol,
independent review and release bindings acyclic; do not require a file to
contain its own hash or a downstream grant hash.

The launcher must validate admission before importing Torch, constructing a
model/optimizer or executing numerical work. It must then verify the live
parameter counts before the first update and use the frozen runtime and
source. A metadata pass never substitutes for these actual execution checks.
Any changed source, cases, seed, objective, optimizer setting, selection rule,
checkpoint policy or limit needs renewed review and a new exact release.

## Required frozen protocol and complete collection

- Register a falsifiable diagnostic question and exact source/runtime/helper
  hashes, generated-data recipe and bytes, one seed, all arms, causal input
  and target boundaries, owner/state-reset semantics, objective, auxiliary
  losses, optimizer/clipping settings and deterministic sample order.
- Every trained arm has **95,000–105,000** actual trainable parameters and
  deployed learned parameters, inclusive. The trainable counts and deployed
  counts each match within **1%** across controls. Count all active heads and
  learned components; no padding, hidden learned models or frozen-backbone
  capacity escape. Verify instantiated and final saved models, not estimates.
- Match original synthetic bytes/targets, ordered examples, sequence lengths,
  supervised exposures, update budgets, seed, checkpoint/evaluation schedule,
  evaluation inputs and tuning/selection privileges. Disclose unchanged
  family-specific auxiliary objectives. Paired optimizer controls begin at
  byte-identical within-family initialization and vary only the preregistered
  optimizer setting.
- Run the **full fixed budget for every arm**. No early-success stop, selected
  winner continuation, rescue tuning, added seed or outcome-dependent retry.
  Freeze all fits before opening held-out generated labels. No held-out-driven
  checkpoint, optimizer or architecture selection. Distinguish independent
  base cases from correlated transformations and report all planned cohorts.
- Freeze positive finite aggregate CPU/wall-time/memory and numerical-call
  ceilings before release. Record actual per-arm exposures, attempted and
  completed updates, resource use and every failure. **CPU time is an
  efficiency measurement, never an equal-time requirement.** This category
  does not establish FLOP-matched architectural superiority.
- Require same-weight implementation and causal input/target checks; final
  live/deployed counts; finite model/optimizer state; checkpoint identity and
  replay; unambiguous attempted/completed update accounting; output identities;
  process cleanup; and bounded actual-resource receipts. A partial matrix,
  unresolved update, missing artifact or cap breach cannot be complete.
  Preserve failures and interrupted runs, never turn them into a winner.

A complete diagnostic collection is not scientific success or promotion.
Numerical failures, behavior failures and INCONCLUSIVE findings remain visible
alongside any collection-completion status.

## Canonical registration and source-only CI

Place each new registration at
`research/synthetic_diagnostics/<study>/FAIR_COMPARISON.json`, with
`study_type: synthetic_diagnostic_fit`. The familiar filename is only a
registration convention; it does not label this a fair architecture comparison.
Declare `runner_path` and `admission_path` as existing repository-relative
Python sources. The admission source declares both preflight and final calls
to the canonical gate, and the runner references that admission module.

`scripts/validate_research_parity.py --check-changed BASE...HEAD` separately
discovers changed files beneath these study directories, requires the root
manifest, calls `synthetic_diagnostic_policy.validate_scope` and checks declared
source hooks. Existing ME/DEV discovery is unchanged. This is static
registration consistency only: it cannot prove runtime call order, source
review, actual counters, one-shot consumption, trust-anchor authenticity or
permission to run. Independent review and the external controller must verify
those boundaries. A source-only CI success authorizes no numerical work.

The canonical `check_manifest` dispatches only the exact new study type to
`scripts/synthetic_diagnostic_policy.py`. The helper is loaded from adjacent
source bytes without relying on a Python bytecode cache. `validate_scope` is
metadata-only. Full `validate_synthetic_diagnostic` admission requires external
approval, review and grant evidence plus independently supplied approval and
grant digests; final collection also requires actual receipts. No trusted
identity is derived from the manifest or from the evidence being checked.
The child final gate returns integrity-complete/resource-pending, not complete
resource acceptance. After the owned process group has exited and its children
are reaped, the outer controller must separately call `validate_resource_receipt`
with the current externally trusted grant digest and actual wait4/component
receipts. Disclose that the outer receipt's own final-write/exit tail is outside
the captured CPU/wall measurement; do not call that observation an exact
whole-process cost or silently include unmeasured tail time in the evidence.

CLI inputs are `--synthetic-approval`, `--synthetic-review`,
`--synthetic-grant`, `--trusted-synthetic-approval-sha256`,
`--trusted-synthetic-grant-sha256` and, for `--phase final`,
`--synthetic-actual`. File inputs hold the exact frozen evidence bytes; actual
receipts are JSON. External trust-anchor values have no permissive default.
These options do not weaken or replace the existing `--reward-evidence` and
operator-provisioned reward-verifier key for original study types.

Regression coverage must preserve the exact old parity-only gate and all
old signed evidence behavior, and falsify missing/stale/substituted approvals,
reviews and grants; wrong source/scope/resource identities; extra tuning or
selection; forbidden claims; parameter/exposure violations; partial matrices;
cap breaches; missing final integrity checks; and attempts to import numerical
libraries during source-only admission. No model fit is part of CI.
