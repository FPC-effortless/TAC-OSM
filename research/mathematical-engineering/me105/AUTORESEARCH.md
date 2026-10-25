# Bounded capability autoresearch

The unit of iteration is a complete candidate plus its exact mathematical graph,
not an attribution study. The current batch is three registered integrated
training recipes × five parent lineages, with two CPU workers. All candidates
are published, including failures; no favorable-seed promotion.

The executable cycle is:

1. Trace the actual source and loss-to-parameter paths on FIT only:
   `PYTHONPATH=src:. python scripts/me105_flow_trace.py --output PATH`.
2. Implement candidate and independent chronological reference. Check input,
   state, masks, chunk carryover, forward/backward, distribution and numerical
   boundaries; build matching typed Lean equations and applicable proofs.
3. Register all hypotheses, parent lineage, plans, objectives and stopping rules.
   Run the no-optimizer `me105_protected_panel.py prepare` once. It measures actual
   parameters/active gradients and graph/FLOP estimates. Source and registration
   must then be committed and mathematically source-linked before fitting.
4. Execute `me105_protected_panel.py fit` once. It checks the frozen closure,
   instantiates every registered model, enforces optimizer budgets and emits
   original checkpoint/optimizer/RNG/gradient/data/resource receipts. Existing
   attempts cannot be overwritten. No training is triggered from CI.
5. Commit all original models **before** running `score`; then execute fitted
   source/reference audits and `me105_panel_summary.py`. Reject qualification
   unless every fixed gate passes for every seed. Better averages alone do not
   override a failed gate. Protected data is not an optimization oracle.
6. Commit original scores, independently restore/replay, append results to the
   master plan and ideas ledger. A changed hypothesis requires a new versioned
   protocol; ME105 admits no additional fits after scoring.

This is deliberately a bounded, auditable adaptation of the propose–implement–
measure–keep/reject loop. It does not claim to be a reproduction or execution of
Karpathy's external autoresearch software. The inherited common runner already
parallelizes independent fits and shares the fixed plans/profile convention;
we avoid adding a second opaque optimizer or an unconstrained DEV search loop.

The current candidate tests three integrated ideas together: source-gap scale
protection, a fast median innovation, and discriminative fine-tuning. Separate
attribution is deferred. Real capability gates and all failures remain visible.
