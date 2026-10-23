# ME-001A — Native entity-set mathematical engineering (2026-10-19)

**Status:** exploratory engineering checks passed; **no trained real-world claim and no architecture promotion.** TAC-OSM remains a native predictive-executable world model, **not an SSM conversion**.

## Hypothesis and representation
ME-000 identified identical-unit symmetry collapse. ME-001A adds variable-size observation tokens, runtime exchangeable anchors, shared recurrent entity dynamics, null/unmatched-token probability, and externally observed known-ID routing as an **oracle control**. Learned routing does **not** access oracle ID arrays, including during the probe.

For latent state `h_i`, stored runtime anchor `c_i`, token `v_j`, observed-feature mask `m_j`, and token-valid indicator `a_j`:

```text
s_ij = dot(q([h_i;c_i]), k([m_j*v_j; m_j])) / sqrt(width)
s_null,j = n([m_j*v_j; m_j])
[A_1j,...,A_Nj,A_null,j] = a_j * softmax([s_1j,...,s_Nj,s_null,j] / tau)
u_i = sum_j A_ij * (m_j*v_j) / max(eps, sum_j A_ij)
h_i,next = NativeTACCell(u_i, dt, masked_channels_i, h_i)
```

This is soft token-to-slot binding **with null**, not optimal one-to-one matching, guaranteed entity identification, or verified persistence. The shared native recurrence is the ME-000 bounded TAC-NER kernel.

## Recorded local CPU checks (PyTorch 2.10.0)
Untrained, random numerical input; no supervised scores or commercial claim. Main seed 20261019, B=3, N=4, J=5, D=4, width=12.

| Software contract | Learned | Observed-ID control |
|---|---:|---:|
| Token-order invariance max abs error | 1.86e-8 | 0 |
| Slot permutation equivariance max abs error | 2.98e-8 | 0 |
| Masked-value leakage | 0 | 0 |
| Entirely invalid token induces sensory write | 0 | 0 |
| State intervention changes output | 0.05876 | 0.08998 |
| Router gradient maximum | 2.12e-5 | 0, inactive by design |
| Active parameters | 691 | 298 |

Ten fixed independent random-initialization property seeds (20261019–20261028) passed 10/10. Maximum token/slot permutation errors both 4.47e-8, minimum nonidentical-anchor divergence 0.004283, minimum recurrent-state effect 0.039827. Known-ID array corruption left **learned** outputs unchanged (exact zero observed difference). Identical initial anchors caused exact slot-identity collapse; distinct anchors permitted divergent hidden states.

No model fitting, SCANIA scoring, learned identity accuracy, compositional transfer, and genuine forecasting results were produced.

## Mathematical limitations and gates
- Anchors break symmetry but do **not** identify semantic entities. Observationally identical histories may make true identity impossible to recover.
- Soft binding can contaminate entities via diffuse assignment. Compare content-only matching, nearest-neighbor/Hungarian, no-null, and known-ID oracle controls.
- Fixed slot capacity currently omits robust birth/death/eviction/alias rules.
- Source-level permutation and masking checks do not prove all floating-point/dtype contracts.
- Cost and parameter matching are required for scientific comparisons.

## Registered ME-001B (NOT EXECUTED)
On authentic, SHA-256-verified **SCANIA Component X v3 TRAIN-only** readouts, compare native TAC-NER + learned binder, GRU, entity-routed GRU, persistence, tuned LightGBM, and simple identity matching. Freeze vehicle-disjoint train/calibration/internal-validation cohorts before training; do not open publisher TEST. Evaluate meaningful sensor change, uncertainty, identity association, state continuity and runtime under identical information and budgets. A shuffled multi-truck set constructed from authentic rows is an identity diagnostic, **not** a naturally observed deployment environment or action-outcome proof. Preserve previous reserved SEQ006C tests.

**Decision:** RETAIN FOR ENGINEERING / SCIENTIFIC PROMOTION NOT ESTABLISHED. All failed and competing architectural ideas remain registered.
