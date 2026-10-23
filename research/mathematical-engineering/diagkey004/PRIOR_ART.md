# Established mechanisms and interpretation corrections

The experiment reuses known gated-delta mathematics, embedding tying and explicit
read-only memory operations. It does not claim invention of a new delta rule,
selective SSM, associative scan or attention mechanism. Native fast/slow TAC is
retained and trained with a new address/content interface.

The existing ME075 prototype already provides separate retention/write gates and
no-write queries (`src/tac_osm/variants/gated_delta_memory.py`, draft PR360). That
source was inspected before fitting. DIAGKEY004 uses equivalent delta algebra,
with shared observed-key addresses, tied value decoding, the native nonlinear
fast/slow branch and an explicit radius-four state projection. These are one
integrated candidate: no causal attribution to individual changes is supported.

External numerical positive reference: Flash Linear Attention
54f725e5dad4abab3211be202b13ead380ef26da,
`fla/ops/gated_delta_rule/naive.py`, first recurrent reference function. The
MIT copyright notice, function excerpt and full MIT license are preserved under
`tests/references`. Its unused `einops` import is removed from the excerpt;
function arithmetic is unchanged. Tests compare the unprojected cell in float32,
scale1, allowing1e-6 absolute/relative error and transposing key/value orientation.
Projection and read-only semantics have separate invariant tests.

Source: https://github.com/fla-org/flash-linear-attention/blob/54f725e5dad4abab3211be202b13ead380ef26da/fla/ops/gated_delta_rule/naive.py
Paper: https://proceedings.iclr.cc/paper_files/paper/2025/hash/4904fad153f6434a7bcf04465d4be2cc-Abstract-Conference.html

The old native implementation already L2-normalizes keys and queries. L2
normalization does not force their cosine to be positive or ensure semantic
retrieval. The frozen DIAGKEY003 check further shows why cosine is not a
standalone correctness criterion: three higher-accuracy seeds use strongly
NEGATIVE target/query cosine. A learned decoder can change sign/coding convention.
Oracle-key substitution can therefore make its output worse even if a latent
association survives. The new shared-address convention is a design constraint,
not a proven diagnosis of the old model's unique failure cause.

The original DIAGKEY002 oracle has eight value classes (chance12.5%), while the
trained task has32 (chance3.125%). Both have four stored pairs; the neural task
has16 possible initial keys and64 total key IDs. Oracle100% applies to protected
orthogonal keys, not arbitrary learned overcomplete addressing. Nothing in this
experiment proves universal 8x8 memory capacity,256-step retention, real-world
prediction improvement, safety, commercial value or independent intelligence.
