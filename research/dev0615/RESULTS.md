# DEV0615 — registered learned-agent research results

**Status: research evidence; no promotion or commercial-readiness claim.**

Completed 140 fresh fits and 14 validation-only pilot fits. Fitting/collection/validation CPU: 91.56 minutes. All twenty training seeds retained per arm.

| Arm | Completion | Mean economic cost | Residual faults | Passing seeds |
|---|---:|---:|---:|---:|
| fixed_bc | 100.00% | 1.9946 | 0 | 20/20 |
| fixed_robust | 100.00% | 1.9946 | 0 | 20/20 |
| split_bc | 100.00% | 1.9946 | 0 | 20/20 |
| split_robust | 100.00% | 1.9946 | 0 | 20/20 |
| no_casm_robust | 100.00% | 1.9946 | 0 | 20/20 |
| raw_gru_robust | 44.13% | 23.6152 | 11945 | 0/20 |
| raw_mlp_robust | 95.84% | 3.0718 | 584 | 7/20 |

Passing-seed criteria differ by study; consult the registration. Completion and physical repair are reported separately in the per-seed evidence. Averages do not replace the reliability gate.

## Acceptance gates

- 18_of20_combined: **True**
- zero_primary_faults: **True**
- public_cost: **True**
- cross_hardware_exact: **UNKNOWN until remote replay**

The explicit public-history reference achieves 100.00% completion at cost 1.9946. This benchmark remains public-rule solvable. The identity treatment bundles separated content and identity augmentation; the robustness treatment bundles shared multi-policy DAgger, EMA and curriculum. Neither factorial isolates each ingredient.

The public manifest, hard cosine routing and relationship-key stream are structural priors supplied to every control. Recurrent content, CDL strengths, graph/operator computation, action predictions and control are trained. This is not entity discovery, symbolic CASM or learned indexing.

## Artifacts and limits

- [Protocol](PROTOCOL.md) and [per-seed table](statistics.csv).
- [Complete statistics](statistics.json), [local replay audit](audit.json), [measured hardware costs](hardware.json), and [SHA256 evidence index](evidence-index.json).
- Exported seed is preregistered, not chosen for favorable held-out performance. `learned-agent.tar.gz` includes model, weights, public JSON-lines runtime and runnable examples.
- Restore evidence by concatenating indexed archive parts in order, then extracting the gzip tar archive into `results/`. Raw, EMA deployment and failed outcomes remain available.
- Native Veritas evidence uses the same recorded synthetic effects. It is execution reproducibility, not independently validated physics.
- Remote action matching remains UNKNOWN until the corresponding CI audit actually passes. Float64 alone does not prove hardware invariance.
- Synthetic intervention units are not money; CPU measurements are not production SLAs. No measured DRAM-traffic/FLOPs reduction is claimed.

Commercial validation still requires a real task, permissioned outcome-labelled traces, actual prices and an authorized shadow trial. See [commercial acceptance contract](../commercial/READINESS.md).
