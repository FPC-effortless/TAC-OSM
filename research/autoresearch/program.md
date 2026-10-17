# TAC-OSM bounded autoresearch operating rules

Apply the experimental cadence of https://github.com/karpathy/autoresearch
(reference commit 228791fb499afffb54b46200aca536f79142f117) to the existing
TAC-OSM research program. The objective remains usable causal memory with
reliable decisions and accounted intervention cost.

1. Choose the next hypothesis from the preceding audited evidence. Register
   controls, fixed held-out metrics, decision thresholds and exposure budget.
2. Commit the experiment before opening outcomes. Verify integrity and smoke
   execution, then execute its entire registered matrix without outcome-based
   stopping or dropping bad seeds/arms.
3. Save all checkpoints and negative outcomes. Replay separately and audit raw
   evidence independently before accepting a scientific result.
4. Enter source/run/evidence and accepted, unsupported or inconclusive claims
   in the research ledger. Retain hypotheses only to the extent evidence permits.
5. Use the finding to prepare the next hypothesis. Stop at the current budget;
   keep research PRs draft/unmerged and report incomplete audits as incomplete.

The first batch is about 30 minutes of experimental compute. A matched causal
experiment uses fixed update/exposure budgets rather than letting wall-clock
variation change training per arm. No generic hyperparameter search, metric
replacement, cherry-picking, or irreversible resets. The ledger records studies,
not a leaderboard that erases failed controls. Changes to future protocols get
new experiment IDs, never retroactive changes to the current acceptance rule.

## Priority correction: DEV-029

The user's 2026-10-16 research review supersedes the earlier next-step proposal.
Defer DEV-028 module initialization until optimization controls are adequate.
Use the registered DEV-029 grid for hypothesis-driven tuning; the earlier
"no generic hyperparameter search" rule does not prohibit this explicitly
requested, bounded, validation-only optimization study. Keep a connected
complete-system integration track: memory-only scores are not full-system
scores. Preserve development successes and fresh replication failures together.
See `docs/DEV-029-OPTIMIZATION-AND-INTEGRATION-PROGRAM.md`.

## DEV-055 outcome and next decision

The learned dynamics/CDL-style/neural-CASM path now trains and restores, but the registered superiority guard fails. Preserve the complete 15-fit matrix and native replay evidence; do not select a checkpoint or tune the same held-out worlds into a positive claim. DEV-056 is the registered memory-novelty deadline, with matched-GRU superiority, reliability and adequate headroom required. Retire the memory-novelty claim if that deadline fails. DEV-057 must test the separate indexed-acquisition, equal-bits scalar and hash controls; fixed-size expert routing is not C5.

## DEV-056 deadline enforced; DEV-057 reference controls completed

DEV-056 did not pass. MTSK-specific superiority pursuit is retired for this lane under the registered deadline; insufficient public-oracle headroom does not reopen the tuning loop. The implemented kernel remains a reference component, without a superiority claim. DEV-057 reproduces the indexed-acquisition/equal-bits scalar/hash reference costs, not a learned C5 result. Future research should target nontrivial learned acquisition and useful conditional computation, using matched total storage and charged build/write/address/verification work. Do not treat public-key lookup success, a correct simulator replay or zero mask skips as evidence for learned architecture novelty. See research/memory-novelty-policy.json.
