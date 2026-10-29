# Complete gradient coverage before parity

The historical053 comparator permits identical missing-gradient lists in both
arms. Agreement between partial records does not imply complete original-model
coverage. This change adds a mandatory live-model contract to a NEW audit runner;
original053/054 source, receipts, failed attempts and checkpoint bytes are intact.

Before either arm runs, capture unique registered parameter names/objects, shapes,
dtypes/devices/trainability and model/buffer hashes. Validate each result against
that contract independently of the other arm. Reject missing/extra/nonfinite or
mis-shaped/dtyped gradients and changed parameters. Finite zero gradients count
as present; they do not demonstrate useful learning. Shared weights count once.
This is reusable diagnostic infrastructure, not a TAC model feature or proof
that all deployed weights contribute useful predictive information.

The preregistered GitHub CPU audit executes only six backward passes on unchanged
original trained seed1701 weights: two original licensed CPython FIT shapes times
native/upper-only/full recomputation. No optimizer step. It records actual55
named tensors/499623 elements only if observed and verifies copied-evidence
common-omission controls before invoking the historical numerical comparator.
This does not repeat the completed054 timing benchmark. The existing500K bytes
are a historical replay reference, not permission for a future out-of-band fit.
All new fits remain95k–105k, <=1% matched-arm gap and otherwise fully gated.

Prior art: PyTorch autograd gradient materialization and nonreentrant checkpoint
semantics, https://docs.pytorch.org/docs/stable/autograd.html and
https://docs.pytorch.org/docs/stable/checkpoint.html . No new AD method is claimed.

PNDS66 (cf14fea287c949aa5aea52f2123d65cce87e0aa4) remains the inherited scoped
Lean/Python full token formulation; no neural graph changed here. Complete
source-to-Lean, AD and IEEE proof remain OPEN. Numerical gradient agreement is
not such a proof. No new PNDS theorem or mathematical release is claimed.

Internal source-authenticated FIT diagnostics only. No DEV/TEST, qualification,
reward/safety certification, promotion, GPU, training, mobile measurements or
quality/capability gain. Absent checks stay NOT RUN/INCONCLUSIVE. All six original-checkpoint audit passes succeeded on GitHub CPU run40832487934.
Each observed55 gradient tensors/499623 elements; maximum logit and gradient
difference zero. Common omission control rejected6/6. Full local suite1433PASS/
4SKIP;42 dedicated software checks PASS. Complete original artifact, raw JSON,
job log and SHA manifest are preserved under research/evidence/gradcoverage055;
see research/results/GRADCOVERAGE055_RESULTS.md. Report closure skips automatic
repeat; explicit workflow_dispatch remains available. Fixtures are engineering
evidence only, and the actual FIT run is internally measured, not qualified.
