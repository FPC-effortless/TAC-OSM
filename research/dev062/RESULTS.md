# DEV062 — registered learned-agent research results

**Status: research evidence; no promotion or commercial-readiness claim.**

Completed 80 fresh fits and 8 validation-only pilot fits. Fitting/collection/validation CPU: 20.60 minutes. All twenty training seeds retained per arm.

| Arm | Completion | Mean economic cost | Residual faults | Passing seeds |
|---|---:|---:|---:|---:|
| tac | 100.00% | 2.0028 | 0 | 20/20 |
| no_casm | 100.00% | 2.0040 | 0 | 20/20 |
| gru | 100.00% | 1.9912 | 0 | 20/20 |
| mlp | 97.52% | 2.4204 | 94 | 20/20 |

Passing-seed criteria differ by study; consult the registration. Completion and physical repair are reported separately in the per-seed evidence. Averages do not replace the reliability gate.

## Acceptance gates

- 18_of20_95_completion: **True**
- median_KL_002: **False**
- median_Brier_excess_002: **False**
- zero_primary_faults: **True**
- competitive: **False**
- cross_host: **UNKNOWN until remote execution**

The exact public reference is an information-matched, reliability-constrained depth2 planner; it is not globally optimal. The privileged control sees hidden state and is a different information ceiling. Latest-observation shortcuts are reported separately.

This is posterior-supervised StageA. Public action-start records identify transitions; repair receipts have a state-independent two-entity schema. No filter, teacher, physical transition law, analytical planner or stop override enters exported inference. The latent state vocabulary and identity binding remain supplied priors. StageB learning without posterior targets has not been demonstrated.

## Distribution and transfer qualification

The registered KL gate is a median across20 seeds, not a per-seed guarantee. 0/20 TAC seeds individually meet KL<=.02; fixed-history KL range[0.039008,0.079326]. Completion reliability and posterior accuracy are different gates.

Secondary cells remain separate from primary acceptance; in protocol order:
- Cell0: worst-seed completion 98.44%, total residual faults across shared-seed evaluations 1.
- Cell1: worst-seed completion 100.00%, total residual faults across shared-seed evaluations 0.
- Cell2: worst-seed completion 64.06%, total residual faults across shared-seed evaluations 25.

Twenty out of twenty successful training seeds has a Wilson95% seed-pass interval approximately[83.9%,100%], conditional on this training corpus and shared worlds. It does not prove>=90% population training reliability or a real-world fault bound.

## Artifacts and limits

- [Protocol](PROTOCOL.md) and [per-seed table](statistics.csv).
- [Complete statistics](statistics.json), [local replay audit](audit.json), [measured hardware costs](hardware.json), and [SHA256 evidence index](evidence-index.json).
- Exported seed is preregistered, not chosen for favorable held-out performance. `learned-agent.tar.gz` includes model, weights, public JSON-lines runtime and runnable examples.
- Restore evidence by concatenating indexed archive parts in order, then extracting the gzip tar archive into `results/`. Raw, EMA deployment and failed outcomes remain available.
- Native Veritas evidence uses the same recorded synthetic effects. It is execution reproducibility, not independently validated physics.
- Remote action matching remains UNKNOWN until the corresponding CI audit actually passes. Float64 alone does not prove hardware invariance.
- Synthetic intervention units are not money; CPU measurements are not production SLAs. No measured DRAM-traffic/FLOPs reduction is claimed.

Commercial validation still requires a real task, permissioned outcome-labelled traces, actual prices and an authorized shadow trial. See [commercial acceptance contract](../commercial/READINESS.md).
