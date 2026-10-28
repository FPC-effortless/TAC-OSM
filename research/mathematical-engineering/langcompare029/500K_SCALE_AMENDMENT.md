# LANGCOMPARE029 explicit language-model scale amendment — 2026-10-28

This research-registration-only amendment implements the owner's current
instruction to compare the **existing 500,122-parameter language TAC-OSM**
against actual matched Mamba-3, BDH and Transformer models. It is NOT a
global override of the 100K prospective rule and NOT GPU authorization.

- Authorized new study ID: `LANGCOMPARE029` only, in addition to the
  already frozen original `LANGTRAIN025` program.
- Machine contract marker: `"parameter_budget_contract":
  "owner-2026-10-28-500k"`.
- Each trained and deployed learned arm must be in **[475000,525000]**;
  for the four-way matched study, the (max−min)/max trainable and deployed
  count gap must independently be <= 1%. Count all actual weights, reject
  dummy/frozen padding, and reject any model constructed differently
  from the preregistered source graph.
- **All other** newly fitted studies continue to require [95000,105000]
  unless separately amended. Existing original checkpoint outcomes are
  preserved; no retroactive training or checkpoint replacement.
- Equal original FIT byte sources, sample order, seeds, budgets, FLOPs,
  tuning privileges, causal windows and exposed-DEV scoring are still
  required. No unverified FLOP estimates can pass a matched study.
- Model and source receipts must come from original objects; an analytical
  count is a preliminary hypothesis, NOT an instantiated model receipt.
- The already-exposed DEV results are internal, never a protected test
  nor independently qualified superiority. No commercial or safety claim.
- Original Mamba-3 BF16 SISO kernel is potentially **incompatible with SM75
  T4**, so the hardware preflight hard-fails without a supported accelerator.
  Do not change model family to bypass this. No charged GPU work is initiated.

Companion: draft PR #557, protocol `PREREGISTRATION.md`. Explicit owner-scale
amendment in `scripts/validate_research_parity.py` checks only this study ID
with the marker. The actual four-arm parity manifest and GPU runner cannot
claim readiness until sourced compute profiles and hardware receipts exist.
