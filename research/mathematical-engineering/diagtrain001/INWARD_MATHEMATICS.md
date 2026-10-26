# DIAGTRAIN001-INWARD001: conditional numerical optimizer correction

This is a pure candidate generator, not a new trained model. Actual ME109
observation→encoder→affine native fast/slow/8×8 scan→retrieval→residual role
heads→distribution graph is unchanged. All104,663active parameters,152FP32
recurrent scalars and existing imperfect task outcomes are retained. The new
function is not wired into the historical optimizer or any training entry point.
The original combined loss, weights, clipping, moments and failed gates stay.

For each block of FP32 weights p, point gradients g, original AdamW candidate c:

- n is its number of scalar coordinates; eps64/eps32 are dtype epsilon.
- gamma=(n+8)eps64/(1−(n+8)eps64).
- s=sum g(c−p), N=sum g², R=sum|g|(|p|+|c|), computed in FP64.
- eta=(4eps32+8gamma)R; no outcome-based margin or LR sweep.
- alpha=max(0,(s+eta)/N) when N>0, otherwise0.
- z=FP32(c−alpha g), performing correction in FP64 and casting once.
- b=gamma sum|g|(|z|+|p|)/(1−gamma).
- Certify computed sum g(z−p)+b≤0. On failure, reject nonzero-gradient
  coordinates; zero-gradient coordinates keep their candidate. Finite checks
  happen before returning any candidates. No values/moments are mutated.

The ideal exact-real displacement d'=d−alpha g satisfies g·d'≤−eta for N>0.
PNDS proves that vector identity/inequality and a conditional scalar rounding
certificate. The runtime precision-derived margin is a proposal heuristic paired
with a conditional error envelope; it is not a Lean-proved derivation of the
whole PyTorch/BLAS/IEEE implementation. If the inward margin proves insufficient,
the realized-dot check rejects the block rather than claiming success.
The floating error envelope assumes ordinary IEEE FP64 operations and a reduction
with at most n additions; library primitives, rounding modes, compiler fusion and
all-source error analysis remain OPEN. Small exact-rational FP32 tests provide
independent numerical checks under the tested environment, not a universal proof.

The mapping is continuous in real-valued nonzero gradients before rounding and
certification, unlike immediate all-or-nothing rejection at the zero projection
boundary. Quantization and the remaining fail-closed branch still preclude a
universal cross-platform identity guarantee. It protects only supplied average
point gradients. No per-regime or finite-loss descent theorem is implied.

## Original real-FIT no-fit probe

Fifteen original proposals, five checkpoints, three authenticated FIT batches.
Original-rate AdamW moments unchanged.195/195blockcertificates nonpositive,
0fallbacks on this host. Actual same-batch average point and total objective
improve15/15, but gapPMerror worsens11/15. OnnextFITbatch gapPMworsens12/15.
No weights are applied or checkpoint created. This is a numerical correction
candidate, not an uncertainty/capability fix. Cross-host verification remains a
separate gate; the original zero-boundary guard's replay failure is preserved.
