# LANGLAYER039 — recurrence-only real-source training and paired comparison

**Registered before any LANGLAYER039 optimizer step.** Explicit owner request
on 29 October 2026 to train and benchmark the already constructed
LANGLAYER038 recurrence-only upper candidate (TAC draft #566, PNDS #64).

## Fixed hypothesis

Prior frozen LANGLAYER036 found upper recurrent reads functionally useful
and upper concept graph almost inert, while LANGLAYER037 found the old
recurrent layer outscored a similarly sized feedforward-only control.
The 499,901-param untrained recurrence-only candidate removes only
ADDITIONAL upper concept graph/seed/equilibrium, expands its upper
recurrent state from 48 to 59 using parameter arithmetic, keeps final
SwiGLU width 442, and preserves original CONTEXT026's complete native,
aligned, wave and BASE concept paths.

Primary exploratory comparison: candidate DEV byte NLL against existing
old trained recurrence-depth2, original TAC and FFN depth2 in matched
five-seed original-source FIT schedules. P3 real-text n-gram is an
unmatched non-neural reference. Test real prediction, not merely
construction. No tuning on DEV and no architecture superiority certification.

## Authentic data and budget

- Original hash-registered PSF CPython 3.13.0 FIT owners (13 ASCII docs,
  one verified real UTF-8 supplement), no source crossing, identical
  129-byte owner-valid windows and sample order under seeds
  1701,1702,1703,1704,1705 from LANGLAYER035 sampler.
- Five new CPU fits, each exactly 320 AdamW steps, B=8, 128
  next-byte targets per window = 327,680 real FIT next-byte exposures.
  Learning rate .001, grad clip 1, same torch/numpy runtime and
  known original 4-document 4096-target *previously exposed* DEV set.
- Same optimizer/train budget, active/deployed learned params 499901
  versus 500122 original, 499725 old recurrent depth2, 499661 FFN
  depth2 (all within 1%). **Training FLOPs are NOT matched**; record
  measured CPU time and no full compute-matched superiority claim.
- Real original source bytes/hashes verified independently before fit,
  candidate SHA256 pinned to PNDS #64 and original parent source SHA.
- All weights, optimizer state, Python/NumPy/Torch RNG, sample SHA256
  and source commit preserved in five new immutable checkpoint artifacts.
  Fresh cold load must numerically replay original DEV metrics before
  any result may be aggregated. Any missing/error seed fails closed.
- Old CONTEXT026/recurrent depth2 and depth3 checkpoints from
  LANGLAYER035 run 40785939365 and independent summary 40786118546,
  FFN-only depth2 and depth3 from run 40788869823 are **never refit**.
- Historic P3 NLL 1.877306112 is included only as an honest unmatched
  lexical n-gram baseline; no fabricated equal-param/FLOP interpretation.

## Reproducibility and boundaries

The source contains a scoped six-theorem PNDS #64 upper recurrence
reference. The whole composed parent+upper Lean, IEEE, fully formal AD
and robust longer memory are still open. Internal user-authorized
exploratory real-text training is allowed without external signer or
independent TEST qualification; do not silently claim completion of
these formal obligations. Protected TEST, independent evaluation,
customer/deployment release, performance certification, hyperparameter
selection and model promotion remain disallowed.

Full five-seed paired NLL and top1/top5/Brier/PPL and cpu seconds must be
reported. Paired intervals with only five seeds are descriptive, not
a definitive significance test; overlapping intervals, repeated DEV
exposure and unmatched FLOPs limit causal inference. Selection by
DEV is not allowed. If there is no net original-model improvement,
preserve the negative result. 128-byte reset-window results cannot
establish persistent long-range memory.
