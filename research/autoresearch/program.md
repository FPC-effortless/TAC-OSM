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
