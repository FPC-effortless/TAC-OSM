# DEV064B — prerequisite-enabled chained maintenance study

**Registered design, not a completed experiment.** New hypothesis and data
namespaces, distinct from the closed DEV064 NO_HEADROOM preflight in PR230.

## Why another benchmark is required

DEV064's fully observed parent-linked repairs produced only 0.0279317
cost units mean oracle advantage over myopic decisions on its 1024 validation
situations. That benchmark is closed as NO_HEADROOM. Do not lower the gate,
reuse the same validation/test, or rebrand the failed study as qualified.

DEV064B introduces a new *preparation* action that can enable later repairs
but does not immediately remove a fault. The environment varies prerequisite
graph order, action prices, preparation cost, per-machine repair success and
collateral damage. No control receives oracle future outcomes at inference.

## World and public information

Three entities in a randomly permuted causal chain. State contains three
public fault bits and one public preparation state. Actions: STOP; PREPARE;
REPAIR entity 0, 1, or 2. Preparation incurs immediate cost but can improve
later repair success; an upstream fault may block a downstream repair;
healthy-target repairs can harm downstream components. All arms receive the
same statuses, graph, success rates, preparation status, action costs,
collateral rates and remaining budget. Terminal fault costs are explicit in
the teacher/scorer. Horizon up to four actions.

Known-law finite-horizon DP computes public action values for training labels
and exact comparison ONLY; deployed neural agents never call it. The
reference can achieve full knowledge only because this is a deliberately
fully observed synthetic benchmark; no posterior-free real-world claim.

## Frozen headroom gate before pilot fitting

Validation namespace 641642 has 1024 independent randomly generated
situations. Exact-optimal expected cost must improve on a *one-step myopic*
choice by >=0.25 cost units per situation on average, and at least 20%
of validation situations must have positive one-step greedy regret
(>1e-6). If either gate fails, close NO_HEADROOM and keep TEST unopened.

These bounds are set BEFORE any DEV064B validation run; do not revise
the same benchmark based on its result. 20 model seeds are not
independent physical assets.

## Train/test and controls

Training namespace 641641 has 4096 situations; fresh TEST namespace
641643 has 2048 situations and remains unopened until eligible pilots,
all full fits, and a permanent test-exposure record. Two validation-only
pilot seeds 641650/641651; complete candidate seeds 641701..641720
(20 per arm).

Five independently trained arms: multi-round learned compositional operator
TAC, retrained no-CASM, GRU, attention, tuned MLP. All five receive the
same public input, identical train/validation examples and 1024 AdamW
steps (LR 0.001, decay 1e-4, batch 128). Enforce within10% active
trainable parameter count and report operations/CPU separately.
Predict expected action-Q returns with SmoothL1 and policy CE
supervision; this does NOT independently establish autonomous
transition learning. A separate learned-outcome head and closed-loop
evaluation are required before architectural claims.

Pilots are screening only. If headroom, physical validation,
numerical checks, and candidate selection fail, do not open test.

## Accept/reject

Compare complete closed-loop cost, transient and final physical harm,
completion and STOP, regret against public exact DP, and CPU.
A CASM-specific advantage requires >=18/20 qualifying model seeds
per registered stratum without added final faults and paired
cost superiority over all four independently trained controls
(upper paired 95% bound <0 and practical improvement >=0.02
cost units/world) with no >10% end-to-end CPU regression
against best effective control.

No programmed policy fallback; no privilege in the learned runtime.
If controls match or exceed TAC, reject the CASM-specific superiority
claim rather than adding depth after test exposure. Retain all
candidate failures, hashes, checkpoint evidence, and runnable source.
Do not modify DEV068, REAL004, DEV067, or MetroPT July B.

## Scope and publication

This branch is an *independent benchmark successor* stacked on
DEV067, not on PR230. The initial preflight CI may run software
smoke and the exact oracle-headroom validation calculation; full
training and TEST are separately gated. Draft until all qualifying
checks and external replay pass. A result on this synthetic domain
does not prove real repair economics or general symbolic CASM.
