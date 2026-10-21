# ME026 validation

- 906 repository tests pass, including six new runtime/mathematical falsifiers.
- Canonical mutable test-count metadata passes; frozen historical count stays 327.
- Plain `import tac_osm` remains Torch-free in a fresh interpreter.
- Original ME025 model/training/test source is unchanged. Only its workflow is
  guarded against retraining on descendant research branches.
- Runtime, timing source and machine contract remain unchanged since the
  pre-execution push at `ecd7d97`.
- All nine original model/probability hashes and the downloaded ZIP match;
  the official authenticated cached dataset and original fingerprints match.
- All nine final checkpoints have step 300, correct real exposure, reconstructed
  sample RNG, FIT-only normalization/popularity and finite Adam moments.
- Original nine-model replay: 53,568 forecasts, maximum probability error
  1.192093e-7, within fixed tolerance 1e-6; all metric errors below 1e-7.
- Six native standalone exports: 35,712 probabilities match original stored
  predictions exactly. Streaming probes use 32 genuine windows/model with
  maximum logit error 4.768372e-7, within fixed tolerance 1e-5.
- Six fresh Torch-only subprocesses pass; neither `tac_osm` nor `scripts` is
  imported. A deliberately corrupted export is rejected before data access.
- Every retained timing median and engineering gate reconstructs from all nine
  raw repeats. No fastest-seed selection or new timing run occurs during audit.

The nonzero-head causal tests check prefix noninterference under changed
unobserved response, future item and later-duration inputs. Their intentionally
unmasked attention mutant is detected. Original masked attention passes.
Kernel impulse derivatives, segment associativity/identity, active input
derivatives and eager/compiled/streaming/serialization agreement are checked
on constructed inputs solely as software verification.

MATHEMATICS.md separates the summary affine algebra from the native nonlinear
state. Full-window input materialization, separate prior storage, reset/window
semantics and pinned exporter limits remain explicit. No causal/action or
independent generalization conclusion follows from replay or latency.

The artifact contains every original final neural checkpoint and every derived
export. The small historical MLP reference remains unverified because its
weights/probabilities were not supplied. Earlier failed/cancelled ME025
workflows are preserved; neither has executed real training steps. Final
restoration and CI status are available in `restoration.json` and the draft PR.
