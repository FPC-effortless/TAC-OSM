# First-class Woyengi / Veritas / PNDS / Forge / TrainLab integration contract
**2026-10-26 · Proposal v0.1.** Source-grounded research contract + constructed-data boundary validator; **not** a live cross-repository integration, trusted verification, real training, formal full-model proof, or production deployment. Master-plan evidence, security, authority and original source rules take precedence.

## 1. Architecture-level decision

These are not merely optional apps for a new AI model: their real capabilities define **what an intelligence model should represent and which claims are independently trustworthy**.

Woyengi owns canonical state, identity, valid/transaction time, conflicts, consent/policy, proposals, governed effects and reconciliation. TAC-OSM **consumes authorized as-of information** and predicts typed distributions/decisions, not authoritative state. PNDS specifies state and source operator mathematics, Lean and finite-precision refinement obligations. Forge publishes lawful point-in-time eligible datasets, identities, availability, split/provenance receipts. TrainLab preserves original budget, seed, checkpoint and replay evidence; a separately deployed TrainLab platform is not verified. Veritas checks prediction-to-outcome correctness, actual changed state, safety, reward gaming and agent qualification independently. The mobile SDK hosts offline advisory inference; it cannot grant itself authority.

Preserve exactly six TAC-OSM research families: System 1, Enterprise, Temporal, Physical, Edge-Lite and Counterfactual, all subject to source-derived model semantics and matched-science rules. This integration adds *no seventh model* and does not duplicate Woyengi's platform kernel or Veritas's benchmark verifier.

## 2. Sources actually inspected

| Repo and source | Actual contract | Interop consequence |
| --- | --- | --- |
| [Woyengi state](https://github.com/FPC-effortless/Woyengi/blob/main/packages/state/src/index.ts) | ClaimLedger.projectAt(subject, predicate, validAt, recordedAt) retains selected, candidates, conflicts, trace | Valid time and knowledge time distinct; selected projection is not necessarily a verified fact |
| [Woyengi ingestion](https://github.com/FPC-effortless/Woyengi/blob/main/packages/ingestion/src/index.ts) | IngestionEnvelope has provisional lifecycle, source principal, content hash, transactionTime, locality policy | An ingestion hash or receipt never equals authoritative acceptance |
| [Woyengi authority](https://github.com/FPC-effortless/Woyengi/blob/main/packages/authority/src/index.ts) | AuthorityEngine contextual policy and source candidate authority vs confidence | Model probability cannot determine write/execute permission |
| [Woyengi governed effects](https://github.com/FPC-effortless/Woyengi/blob/main/packages/effects/src/index.ts) | ActionIntent, ExecutionManifest, Receipt, ObservedEffect, Reconciliation, IndependentVerificationResult, AcceptanceOutcome, SemanticCommit | Prediction or proposed action cannot be promoted to verified execution |
| [Woyengi architecture](https://github.com/FPC-effortless/Woyengi/blob/main/docs/architecture.md) | Append-oriented canonical records and bitemporal reconstructable state | No writable neural shortcut around canonical history |
| [Veritas qualification](https://github.com/FPC-effortless/veritas/tree/main/src/investigation_world/qualification) | Distinguishes code/science/frontier/release validation | Outcome evidence must come from independent qualified source/evaluator |
| [Forge draft #420](https://github.com/FPC-effortless/veritas/pull/420) | JSONL observations and availability clocks, actions/outcomes, grouped splits and hashes | Source rights, authentic import, independent security review remain open |
| [PNDS architecture](https://github.com/FPC-effortless/pnds-lean/blob/main/foundation/v1/ARCHITECTURE.md) | Lane-specific typed state, limited affine scan lemmas, observational action vs intervention law | Refine exact interfaces; no fictitious full Lean/FP32 proof |
| [TAC-OSM charter](COMPANY_ARCHITECTURE_CAPABILITIES_CHARTER_2026-10-26.md) | Six research families; Remember/Understand/Predict/Decide/Verify/Deploy | Interop becomes a binding input to new model research questions |

## 3. Proposed versioned boundary record types

This is a deliberately narrow **adapter** and NOT Woyengi's native canonical schema, Veritas's native verifier format, a source-license authority or a secure cryptographic attestation service.

- **SourceObservation:** identifier, source/entity ID, event-valid time, recorded/known time, real available-at time, model cutoff and content SHA256, split and provenance reference. Input eligibility requires both recorded/available times <= prediction cutoff. Late-recorded past events must not be treated as historically known. Cross-entity and source/time split isolation remain separate scientific gates. Conflicts must not be silently suppressed by adapters.
- **PredictionReceipt:** original model/checkpoint hash and input observation IDs, prediction time, horizon, target definition, typed output, stated uncertainty and advisory-only status. No future gold labels, privileged scores, authoritative accepted state or permissions.
- **OutcomeEvidence:** independent later original observation source, record ID/time/availability and source hash; explicit prediction link. A receipt is not automatically true simply because it exists or hashes correctly.
- **VerificationReceipt:** separate verifier and original outcome/reward/goal/safety references, trust/access proof and result. A local schema checker cannot authenticate signer or evaluator independence; that remains **UNVERIFIED** until a protected independent runner validates it.
- **ActionProposal:** non-authoritative candidate action requiring a Woyengi policy/permission and effect-reconciliation route to cause a real action. Never construct ExecutionManifest(AUTHORIZED) or VerifiedSemanticCommit from a model receipt.

Common *structural research preflight* files:
- [Contract sample](../contracts/TACOSM-ECOSYSTEM-BOUNDARY-v1.json) specifies strict candidate record fields and negative-claim status.
- [Validator](../scripts/validate_ecosystem_boundary.py) checks identity, time, references, equality/role restrictions and forbidden future-source fields, but is not a real Woyengi or Veritas API adapter.
- [No-fit tests](../tests/test_ecosystem_boundary.py) exercise constructed valid and malicious cases only; zero real-world qualification.
- [CI](../.github/workflows/ecosystem-boundary-gate.yml) enforces the validator tests independently of actual training, hidden evaluation, source access, external effects and cloud credentials.

### Native Woyengi state projection shape adapter — 2026-10-26 incremental implementation

A new [TAC-OSM native projection adapter](../scripts/woyengi_projection_adapter.py) now reads the actual `ClaimLedger.projectAt` StateProjection shape (ClaimRecord with validTime/transactionTime, candidate, selected, conflicting records and source trace). It uses the source's real bitemporal clocks, retains all eligible competing candidates, rejects future-known records, and requires a **host-provided, principal-and-purpose-scoped read-authorizer** to approve the complete claimed record set and a separate allowlisted feature mapper. It creates no canonical write and returns `STRUCTURAL_ONLY`, `NOT_VERIFIED_BY_ADAPTER`, `NOT_EVALUATED`, and `NOT_GRANTED` trust markers. The adapter's derived claim-content SHA256 does **not** authenticate the original publisher, lawful use, Woyengi authorization provenance, or an upstream sensor's exact availability time.

The [constructed native-shape regression suite](../tests/test_woyengi_projection_adapter.py) and [no-fit CI workflow](../.github/workflows/ecosystem-boundary-gate.yml) validate the shape transformation, knowledge-time eligibility, retained conflicts, withheld features and callback denial; none establish a live Woyengi host integration. The Woyengi-owned read-only permission/feature export port is **proposed**, not implemented, in [Woyengi draft PR #55](https://github.com/FPC-effortless/Woyengi/pull/55), with its constitutional active tickets and canonical kernel unchanged. Its independent trusted read/record authorization, local-only privacy, source-rights, and read-only semantics must pass prior to processing lawful real data.

### Woyengi source-side implementation and TAC consumer — 2026-10-26

The Woyengi-side implementation candidate now exists as **[draft Woyengi PR #56](https://github.com/FPC-effortless/Woyengi/pull/56)**, adding `packages/predictive-export/src/index.ts` and the companion source-shape/capability tests. It uses actual Woyengi `ClaimLedger.projectAt` and `CapabilityEngine`: present-time `RECONSTRUCT` and `EXPORT` on the projection, then `READ` and `EXPORT` on **every eligible claim**, including competing candidates. Its real authorization time is supplied separately from historical inference cutoff, so an expired or revoked capability cannot be resurrected by an as-of query. The actual model destination is host-controlled. Denied conflict, unavailable source, missing AI-use permission, forbidden locality, unsafe features or future source cutoff aborts the entire export. The exporter never grants action authority or creates a Woyengi semantic commit.

TAC-OSM now includes a corresponding **[serialized Woyengi v1 export consumer](../scripts/validate_woyengi_predictive_export.py)** and [16 constructed boundary regression tests](../tests/test_woyengi_predictive_export.py) in the same no-fit CI as its original 23 malicious-envelope and 19 native-projection tests. This tests a **format contract**, not an authenticated cross-repository network exchange. A declared feature SHA256 has its shape checked, but the TAC Python consumer cannot independently authenticate Woyengi's trusted host, source publisher, right-to-train claim, native projection conflict equality or an independent Veritas evaluator. Woyengi's source host must be reviewed and connected to real policy/rights data before receiving real model inputs; its initial source-side CI passed at [run 40411110263](https://github.com/FPC-effortless/Woyengi/actions/runs/40411110263) before subsequent hardening. No real dataset has been exported, trained or independently scored.

The two repository modules are experimental drafts under independent code review and **must not** be described as a deployed cross-product feature, general Woyengi authorization API, secure attestation, qualified TAC-OSM model or validated physical-device runtime.

## 4. Required source-to-outcome sequence

1. **Woyengi projection or actual publisher:** consented/legally authorized, principal-and-purpose-scoped snapshot with full actual validAt + recordedAt, source identity, candidates/conflicts and locality policy. Reject a projection that hides contradictory claims without reason.
2. **Forge adapter:** preserve actual observation and knowledge/availability clocks; checked rights and per-event original content hashes; immutable source/splits; **never** retrofit a later available outcome as an earlier feature. Do not transfer protected TEST labels into TrainLab FIT.
3. **TrainLab:** freeze source/seed/splits/compute/budget, exact original model bytes, checkpoint and runner allowed-access manifest. Do matched active parameters within 1% and equal exposure/data/tuning/compute for a claim of architecture superiority; simpler one-arm capacity research may proceed only under its own existing source gates.
4. **TAC-OSM:** update compact source-authorized working state, output typed Forecast/Predicate/Choice/Score/Defer with source cutoff and a model digest. This is *advisory*; neural state is not Woyengi canonical state.
5. **Veritas:** independently recompute real observed outcomes, proper scores, critical subgroup/decision cost, hidden-path/reward-gaming evidence. A merely signed self-report is not an independent qualified verifier receipt; absent independent source/ACL material must fail closed.
6. **Woyengi action path (optional, separately governed):** ActionIntent → explicit policy/authority/budget/risk decisions → execution manifest → observed effects → reconciliation → independent verification → acceptance. Playback of a prediction or model experiment must NEVER replay consequential external side effects.
7. **Model belief correction (prospective):** only future legal source observations within independently authorized training scope; keep original source immutable and sealed TEST inaccessible to online learning.
8. **Mobile SDK:** offline predictive state/output with signed/syncable *proposals* only; backfilled source observations need actual availability/authority and cannot retroactively rewrite an earlier prediction.

## 5. Engineering and PNDS mathematical obligations

- Identity types: source entity ID and known-at-time resolution, conflicting evidence, causal event order, source availability and action provenance as first-class inputs; no identity or future-outcome oracle shortcut.
- Memory usefulness: authentic long histories with entity grouping and interference vs structured canonical state and matched model baselines, practical decision cost and overservicing metric; optional external Woyengi memory size differs from bounded neural working-state size.
- Typed data boundaries: model prediction, empirical observed outcome, verifier assessment and authorized action are distinct sorts; no implicit promotion.
- PNDS: formalize source-time eligibility, state update and scan legality, source law of distributions/censoring, numerical kernel/AD and finite-precision equivalence. Closed exact-real theorems do not prove the full Torch/Android result.
- Causal honesty: conditioning on recorded actions is not an identified intervention distribution; counterfactual family requires additional causal evidence.
- Security: protect storage/locality, non-consented source fields, hidden TEST labels, verifier secrets and policy data. Source hash authenticates bytes, not license, correctness, independence or authorization.
- Mobile deployment: original source model → genuine exporter → per-step numerical/latent state parity → physical Android CPU latency/RAM/energy benchmarks → carefully governed offline queue/sync and SDK release gates.

## 6. Concrete cross-repository acceptance gates

| Milestone | Acceptance requirements | Status |
| --- | --- | --- |
| TAC-OSM structural boundary | Constructed malicious late-knowledge, forged role/authority, source-hash and mismatched outcome tests pass; no training | **Candidate only** |
| Woyengi source adapter | Source-authored projection with real bitemporal and conflict/denial tests, no kernel mutations | **NOT BUILT** |
| Forge/TrainLab authentic source | independent rights, source-time correctness, real grouped train/DEV/TEST, exact replay and budget receipts | **NOT QUALIFIED** |
| TAC skill | authentic source unseen cohort, trained full model, per-seed gates, proper baselines and independent replay | **NOT QUALIFIED**: S1 0/3 and ME110 0/5 still failed |
| Veritas evaluator | independently trusted actual outcome/reward/goal recomputation and hermetic access evidence | **NOT ATTESTED** for these TAC checkpoint runs |
| Woyengi action | external governed effect manifest, authorization, independent verification and acceptance | **NOT INTEGRATED**; prediction advisory |
| PNDS proof | source-pinned whole operator/FP32/AD/probability-law refinement | **PARTIAL** |
| Mobile | real trained checkpoint/export/Android device tests and actual operator-authorized sync | **NO QUALIFIED DEVICE RUN** |
| Customer value | actual consented developer/customer adoption, utility and economics | **NOT MEASURED** |

**Phased action:** P0 strict boundary and source projector; P1 one real Enterprise/Temporal end-to-end Forge → TrainLab → TAC → Veritas study with new authorized data; P1 one Android source-equivalent edge pilot; P2 Woyengi governed advisory proposals under independent real effect verification. Each repository integration requires its own review and explicit acceptance; this document neither changes the other repositories nor grants training, spending, merge or effect authority.

**Next falsifier:** an integrated source-verified model must outperform a meaningful stateful/structured reference in real calibrated outcome and decision cost on a genuinely untouched cohort without forbidden source access or unaffordable mobile overhead. Passing envelope-fixture CI alone does not establish such a result.
