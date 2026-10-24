# Original native-state stability verification (PNDS v1.3, 2026-10-24)

**Scope:** exact-real conditional mathematical envelope and numerical source diagnostics for pinned DIAGKEY004. The model's original learned architecture, parameters, training, and failures are untouched.

Source: https://github.com/FPC-effortless/TAC-OSM/blob/081e534a074001c6a3b2251b3ff31ad20ec4fca8/src/tac_osm/aligned_associative_tac.py
Formal PNDS companion: https://github.com/FPC-effortless/pnds-lean/pull/7

## Proven structural result under explicit premises

The native fast and slow registers have 88 coordinates each. Their candidates are bounded via tanh, with a sigmoid fast interpolation and a quarter-sigmoid slow interpolation. The learned 8x8 associative state is Frobenius-clipped to radius 4 after writable updates, and exactly preserved on read-only events. From initialized zeros, under finite tensor operators this structurally implies at every step:

  max_abs(f_t) <= 1; max_abs(s_t) <= 1; Frobenius(M_t) <= 4.

PNDS NativeInvariant.lean proves a conditional complete-register envelope and its indefinite stepwise composition, an exact-real scalar radial clipping bound, and a read-only memory noncontraction obstruction. The proof explicitly assumes bounded candidate/gates and a well-behaved projection. It is not a theorem of original floating-point source equivalence, global nonlinear incremental stability or learned predictive accuracy.

## Verified original-source mechanism tests

[GitHub Actions run 40137008967](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40137008967) PASS. All three kinds of tests:
- 26 pinned original-gated-write versus independent PNDS 8x8 Python reference checks, including FP32/FP64, projection, masks and gradients.
- 40 original full-recurrence logits, fast/slow/memory intermediate trace, all-weight derivative and local nonlinear state-JVP comparison checks (original DIAGKEY004 architecture has 101,527 parameters).
- 1,100 additional native envelope, writable/read-only, source-JVP, 8,192-step orthogonal decay and live radius-clip interference checks. Tested source-shaped generated inputs, not real records or checkpoint-trained predictions.

Observed maximum memory Frobenius norm across generated forward histories was 0.651046; separate radius-four boundary diagnostics activate global projection. No fit, no original checkpoint score and no real-world capability claim.

## Stronger negative mathematical result

Read-only source events preserve memory M exactly while fast and slow registers can change. The complete state therefore cannot be uniformly strictly contractive in a product norm retaining the preserved memory component, regardless of optimizer success. Stable adaptive fast behavior and invariant useful memory must be studied separately.

For an unprojected write, Uq = alpha*M*q + beta*(v-alpha*M*k)*(k^T*q). Orthogonal key k and query q eliminate the direct correction but not retention decay. Crucially the **actual global Frobenius projection** can also attenuate unrelated old reads when the new write increases total matrix norm beyond 4.

Valid unit-address/unit-value counterexample: initial M[0,0]=4; new k=e1, old q=e0, v=e0, alpha=0.9999 and beta=1. Before projection, old read=3.9996 and memory norm=sqrt(3.9996²+1)=4.1227176. After radius-four projection, old read=3.8805472, an additional 0.1190528 loss from a perfectly orthogonal new write. The original source's operator test reproduces this. Thus a preprojection orthogonal-query theorem does not justify projected memory retention.

## Research decision

Prioritize a protected, source-authorized entity-memory budget that prevents global projection from weakening unrelated verified facts. Before training, establish clear source-consistent memory semantics and a falsifiable retention/error certificate. To claim a trained improvement, require authentic longitudinal observations, matched parameters/training data/training compute, actual original checkpoints, preserved negative results, and realistic held-out entity/site/time splits. Equal CPU wall time is NOT a fairness requirement.

Still OPEN: global whole-native Jacobian/ISS certificate, operator-to-full-tensor Lean proof, independently useful 256-distractor retention, and real-world predictive advantage. The 17-variant mirrored historical registry is unchanged.
