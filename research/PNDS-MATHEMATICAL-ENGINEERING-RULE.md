# Mandatory TAC-OSM ↔ PNDS mathematical-engineering research rule

The normative rule is [PNDS MATHEMATICAL_ENGINEERING_RULE.md](https://github.com/FPC-effortless/pnds-lean/blob/main/foundation/v1/MATHEMATICAL_ENGINEERING_RULE.md) in [PNDS merged PR #5](https://github.com/FPC-effortless/pnds-lean/pull/5). This TAC-OSM document is its **source-repository counterpart**. PNDS #5 has merged into `main`; the TAC-OSM implementation is promoted here to `master` through an isolated PR.

**Compulsory for every new or revised architecture variant:**

1. Register the variant here with a stable ID, exact TAC source commit SHA, evidence/checkpoint paths, inherited version lineage and preserved failures. Every version must have a reciprocal mathematical record in PNDS and a source record here, indexed by `research/pnds-variant-math-links.json`.
2. In PNDS, define full typed computational graph and equations, **Lean 4** definitions/proof assumptions/remaining obligations, plus executable **Python** mathematical reference and tests. Both formulations must refer to the **same** operator, variant ID and source; no toy reference is represented as complete original model.
3. Establish pointwise numerical, gradient, mask, time/action legality, finite-precision and serial/parallel correspondence **against the actual original TAC source**. Unmet guarantees must be `OPEN` and stop claims of mathematical completion. No `sorry` theorem may count as proved.
4. Require PNDS Lean compilation/axiom inspection, Python test gates and source-to-reference parity before declaring the mathematical release gate passed. Reinspect reciprocal manifests and original source SHA whenever a branch or variant changes. A formal proof cannot upgrade synthetic diagnostic evidence into authentic predictive capability.
5. Preserve authentic original real data/checkpoints/negative findings; uphold parameter and training budget/data fairness, but **do not require equal CPU wall-clock runtime**. No new results are inferred from documentation or proof alone.

**Source-pinned per-variant reciprocal manifest:** `research/pnds-variant-math-links.json`; exact PNDS counterpart: [foundation/v1/variant-math-links.json](https://github.com/FPC-effortless/pnds-lean/blob/main/foundation/v1/variant-math-links.json). This is a 17-version *partial historical index*; further DEV/ME/ESWAM variants must be appended rather than replacing the prior registry. Every current row is marked *partial shared-kernel only*, not whole-model Lean/Python/source verification.

For each implementation PR, include the PNDS formulation commit/PR and its status in the description. For each PNDS mathematics PR, include the TAC implementation commit/PR and original empirical scope. Both PRs must be reviewed separately.
