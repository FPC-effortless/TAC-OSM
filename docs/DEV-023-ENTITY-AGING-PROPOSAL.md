# DEV-023 proposal: separate address isolation from local aging

**Proposal only; no implementation, execution or findings.** This is not yet a
complete preregistration. Fix sampling, training seeds, thresholds and code
before opening any new held-out results.

DEV-022's primary retention-removal comparison failed: native TAC stayed at
50% with or without decay and erase. Its entity-partitioned/local-aging bundle
reached 100% H512 under short training and passed the combined development gate.
That bundle cannot tell which component mattered.

## Minimum next factorial

Keep the same 768-scalar state and trainable writer, with learned decay enabled.
Cross these two interventions in a complete 2x2 grid:

| Write/read keys | Aging scope | Interpretation |
|---|---|---|
| Learned full-coordinate keys | Global | Unchanged native reference |
| Public entity-coordinate masks | Global | Isolate coordinate addressing while keeping unrelated-write decay |
| Learned full-coordinate keys | Entity slice only | Change decay application without isolating additive writes or reads |
| Public entity-coordinate masks | Entity slice only | Reproduce the successful DEV-022 bundle |

For the unmasked/local-aging arm, only the decay term is restricted to the
public entity slice; erase and additive writes retain the full learned keys.
The mask must not accidentally isolate those terms too. For masked/global
aging, keys and reads are restricted but decay still affects the whole state.
Use identical initial tensors, same event/receipt budget, common head and
optimizer, and report actual active parameters and state budget. These factors
still change optimization behavior; do not infer semantic mechanisms from
parameter equality alone.

## Necessary extensions before claims

- Use fresh training/validation/test namespaces. Freeze H512/H1024 accuracy,
  per-seed swap/reset controls, failures, unnecessary services and physical cost
  rules before execution. Keep DEV-022's failed primary contrast immutable.
- Report model-only and inspection-selected costs separately. The DEV-022
  selected-cost constraint failed for several otherwise accurate models by
  small positive increments; assess calibration/acquisition in a separate
  planned comparison rather than relaxing its old threshold retrospectively.
- Test whether the factorial transfers to larger clutter/history and variable
  entity counts. The current hardcoded eight-entity public boundary must first
  be generalized and independently validated; no entity-count transfer claim
  can follow from another fixed-eight run.
- Increase independent training seeds if feasible and distinguish component
  main effects from interactions. A larger state distance is not an endpoint.

The result should determine whether write/read isolation, selective aging or
their interaction deserves the next architectural intervention. It would remain
development evidence, separate from full TAC/CASM capability validation.
