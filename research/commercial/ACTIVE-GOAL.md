# Active goal: a reliable learned maintenance model that earns its complexity

Status: ACTIVE — replaces milestone-only completion as the operating objective.
Owner: Codex research agent. User authorization: continue model development and
source public maintenance data. No completion claim from trained checkpoints,
synthetic success, passed CI, or changing an already-open test metric.

Deliver a runnable learned state/prediction/control model with honest uncertainty,
reliable stopping, useful real-data forecasts and competitive verified-task cost.
Prefer the simplest architecture meeting these goals. CASM/CDL claims must be
retired or narrowed if matched retrained controls do as well; superiority is a
falsifiable hypothesis, not a promised outcome or reason to weaken controls.

## Completion checks

1. Real sensing: persistence-start and reversible instance-normalization studies
   on a new recording; per-channel train-scaled MASE and raw-unit MAE, with a
   simultaneous no-regression gate against persistence on EVERY analog channel.
   Preserve original NMSE and expose normalization/drift contributions. Reject a
   trained-mean collapse even if a new aggregate metric makes it look favorable.
2. Belief integrity: restricted policy with no parallel history-latent input;
   exact/shuffled/blurred public-posterior interventions on fresh worlds, p95/p99
   per-decision KL and state-dependent action sensitivity. Near-zero exact-posterior
   flips alone are not evidence of bypass if learned beliefs are already accurate.
3. Temporal robustness: clock-origin invariance and time-unit invariance with
   corresponding unit metadata; H512/H1024 and unseen identities. Rescaling actual
   elapsed physical time is not generally a symmetry. Benchmark event-clock
   priors must be disclosed. Reject reliability trade-offs across required strata.
4. Architectural attribution: independently retrained GRU/MLP/attention/no-CASM
   controls, matched information/parameters/training and measured inference cost;
   fresh chained dependencies with registered oracle headroom, and separate R1
   warmup/supervision ablations. No superiority claim from ripped-out modules.
5. Autonomous prediction: public action/observation learning without exact
   posterior or hidden-state training targets; compare against supervised models
   on the same fresh evaluation, preserving information/representation/policy limits.
6. Real utility: permissioned operational action/inspection logs, actual costs,
   independent post-repair outcomes and out-of-time/site evaluation. Public source
   acquisition continues autonomously. Missing economics or verification remains
   UNKNOWN; do not manufacture data by joining unrelated sources or using closure
   as physical success. No foundation-model, valuation or deployment claim follows
   from a bounded compressor forecast or simulation study.

Operational acceptance requires >=18/20 fresh seeds reaching >=95% completion in
each required long-history/identity stratum, with no physical-fault regression.
Track per-decision p95/p99 KL; numerical qualification thresholds must be committed
with each task before fitting, never inferred from a favorable median.
CASM-specific claim deadline: DEV064-CHAINED-CASM. Register a tuned-MLP/GRU/attention
comparison and minimum oracle headroom before fitting; if its complete registered
budget fails the cost/reliability/compute gate, narrow or retire the superiority
claim instead of searching for a favorable benchmark after opening outcomes.

Every candidate uses >=20 fresh training seeds, fixed validation-only selection,
registered held-out gates, all failed fits, proper probability losses and physical
risk reporting. Each tranche has a committed budget and decision rule. A failure
closes that registration; a new design uses a new ID and genuinely fresh holdout.
No retrospective resealing, cherry-picked successful seeds, widened replay
thresholds, cost-free inspections, or scripted stopping fix hidden in an export.

## Current execution order

- REAL-MAINT-002 is closed NO PROMOTION; July B remains unopened.
- DEV063 is closed NO PROMOTION; original weights/evidence stay intact.
- DEV065 is closed NO PROMOTION after120fresh models/12pilots, full local replay
  and closed-loop semantic substitutions. Optional batched ingestion independently
  passes its local engineering gates without changing model quality.
- Next: a fresh stable-versus-evolving semantic-interface experiment, with generic
  probability/log-probability actor coordinates and matched encoder/actor updates.
  Exact beliefs failing to repair the actor makes decision fidelity a first-class
  question. NEXT-DECISION.md is a proposal, not an already registered experiment.
- Dependency complexity/R1 attribution and posterior-free learning follow their
  registered gates. Scale/SSM/attention arms wait for a valid real-data baseline.

Prior artifacts and draft PRs221/222 stay intact. Work remains active until these
checks pass or a concrete missing dependency prevents a specified gate. Report
which gate is blocked; do not rename an incomplete goal as success.
