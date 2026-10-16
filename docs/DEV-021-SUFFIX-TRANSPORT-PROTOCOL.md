# DEV-021 — Frozen suffix transport execution protocol (development)

**Status: fixed before GitHub Actions full execution. No observed DEV-021 outcomes yet.**

The original [DEV-021 proposal](../docs/DEV-021-SUFFIX-TRANSPORT-PROPOSAL.md) governs scientific intent. This concrete implementation only investigates trained recurrence *difference dynamics*. DEV-018/019/020 checkpoint identities, primary endpoints, negatives and audit artifacts remain unchanged.

## Research question

For two identical-ending but counterfactually distinct public histories, how does the actual frozen TAC `DeltaMemory` transport their state difference through the same public suffix? How much comes from its learned multiplicative decay and how much from the input-conditioned erase term?

Actual recurrence for each expert (`S` with value×key dimensions):
```text
S' = d*S + a*(-(S @ (e*k)) outer k + wv outer k)
```
When public suffix inputs are identical for both histories, additive write `wv` cancels:
```text
D' = d*D - a*((D @ (e*k)) outer k)
```
All gates, keys and assignments are computed from the unchanged frozen trained weights and the *actual* equal public suffix. The identity does not establish that the difference encodes the correct future label.

## Frozen sources and grid

- DEV-018 immutable run [38802569971](https://github.com/FPC-effortless/TAC-OSM/actions/runs/38802569971), with 60 checkpoints and the independent audit; original dataset and success gates remain authoritative.
- Two separate native checkpoint arms: `tac_joint`, `tac_encoder`. Three purely observational frozen controls: `schema41`, `gru768`, `memoryless`.
- Seeds 43, 59, 71; training regimes short and long. **6 checkpoint/regime jobs; each contains 2 native × 192 pairs = 384 paired rows.** No omission or cherry-picked seed allowed.
- Fresh synthetic evaluation domain `dev021-transport-v1`, **H512**, 3 due delays (32/128/512) × 2 mature deadlines (at/after) × 2 cancellation roles (pending/reissued) × 16 repetitions = **192 opposite-outcome public pairs** per regime/seed.
- Determine the *actual* last unequal public event per pair. Validate that every later raw public event **and all encoded model input features, entity IDs and values** are identical for both partners.
- Run original frozen memory updates for both histories through H512. Capture native state differences at last divergence, H32, H128 and H512. Query final target entity using the inherited head; these outputs are **observational and not compared to altered-state predictions**.

## Complete 2 × 2 factorial transport

Starting with the *same* post-divergence state difference `D`, execute on the identical suffix:

| Transport | Decay | Erase |
|---|---|---|
| Actual trained recurrence | Trained d | Trained e |
| No decay | 1 | Trained e |
| No erase | Trained d | 0 |
| Neither | 1 | 0 |

Hold the actual public suffix, trained write/read keys, assignments and every other parameter fixed. Record absolute norms, ratios and the per-expert trained decay product. **The four variants are difference-dynamics counterfactuals, not full model states or new trained models.** There is no intervention on the actual native reader, predictions or service policy, and neither intervention licenses a capability claim.

## Predefined numerical/experimental validity gates

- Run the difference equation under float64 (explicitly different from DEV-018's original float32 predictions) so arithmetic identity can be checked without long-suffix subtraction noise; read weights from the unchanged archived checkpoint, convert to float64 only for this post-hoc algebraic analysis. No original fp32 scientific endpoint is retested or overwritten.
- Every time step must satisfy max componentwise deviation between independently transported difference and direct two-state recurrence **<= `2e-9 + 2e-5 * max(abs(direct difference))`**. The maximum error is recorded per pair. Missing or failing any gate invalidates that job.
- Original archived checkpoint file SHA256 and parameter hash must match the DEV-018 report for all five arms. Every group must have balanced causal outcomes, distinct fingerprints and a genuine common suffix.
- Structured/GRU/memoryless controls use the exact paired prefixes; controls report raw L2 only, without cross-architecture normalized information claims.
- Report all per-seed/regime/arm histories. Mark zero post-divergence norms explicitly; a ratio with near-zero denominator is not valid causal information.
- The **separate stdlib audit** regenerates public counterfactuals and the true suffix pivot from its own deterministic world generator; rejects missing pairs, altered prefix/seed grids, invalid decomposition agreement or corrupted metadata. It aggregates raw rows rather than trusting the research runner's printed means. It cannot independently replay floating-point training or automatically interpret a state distance as semantic information.
- Invalid/failed/skipped jobs and diagnostic contradictions remain visible, not positive model claims.

## Interpretation

A large increase in counterfactual difference norm when decay is disabled would be evidence that trained multiplicative decay contributes to contraction *of that measured difference*; likewise for erase. These terms interact, so report all four arms rather than attributing the net loss additively.

The result cannot alone show that true causal information is recoverable or that reducing decay improves prediction. Replacing native learned memory requires a **subsequent separately preregistered train-and-evaluate experiment** with original native action/cost gates, controls for entity addressing, reader alignment, and out-of-domain generalization.

DEV-020 state/read diagnostic and DEV-018/017 full TAC success gates remain **FAILED** regardless of DEV-021's observed magnitude. Do not merge a claim of world-state capability based on these measurements.

## Reproduction commands

```bash
python scripts/run_dev021_suffix_transport.py --seed 43 --regime short --smoke --output artifacts/DEV-021-smoke
python scripts/run_dev021_suffix_transport.py --seed 43 --regime short --frozen artifacts/DEV-018-short-43 --output artifacts/DEV-021-short-43
python scripts/audit_dev021_suffix_transport.py artifacts/DEV-021-*/report.json --output artifacts/DEV-021-summary.json
```

The workflow automatically executes all seeds/regimes and uploads an auditable artifact after the independent audit.
