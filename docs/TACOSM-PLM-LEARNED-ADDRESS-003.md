# TACOSM-PLM-LEARNED-ADDRESS-003

**Status: pre-registered, pending independent measurement.**
Origin: [research issue #150](https://github.com/FPC-effortless/TAC-OSM/issues/150).

## Prior outcome and rationale

The fully measured learned-address-002 mechanism **failed** its joint
preregistered rule: although structured high-noise M32 learned-minus-raw
accuracy was +0.071635, its isotropic mean gap was -0.0324419 (permitted
loss only -0.02). The failed result and its historical artifact remain
unchanged. This is a new, independent test of a proposed cause: the
002 model was trained exclusively on the structured corruption channel.

## Frozen hypothesis

Can the **same, channel-blind 16D learned query/key metric** adapt to
corruption without sacrificing more than 0.02 mean absolute isotropic
accuracy versus raw-dot, and improve on structured corruption at M=32,
sigma 0.2 and 0.4 by at least 0.03 absolute accuracy?

### Prespecified experimental arms

The architecture is the existing unmodified `LearnedAddressMetric002`.
All runs have 10 fixed seeds, identical initialization and 750-step AdamW
training budgets per learned arm, batch 512, M=32 and sigma in {0.2,0.4}.
There is **no inference-time corruption label** or raw-dot shortcut.

- **Structured-only:** retrain the original model using only structured
  queries, alternating sigma 0.2 and 0.4.
- **Mixed-unlabeled:** at every even step use structured corruption, at
  every odd step isotropic; sigma sequence is 0.2,0.2,0.4,0.4.
  This is 50:50 by design.
- **Raw-dot:** no trained parameters; reference on the very same keys/queries.
- **Oracle target-key, wrong-key query and slot permutation:** actual
  evaluation-only controls on the same held-out batch for both trained arms.

All model inputs are exactly `query` and `keys` with no channel input.
The pre-existing generator's diagonal channel and normalization are unchanged.
This stacked research branch depends on PR #141; that dependency must be
preserved until its original model and generator are integrated.

## Fixed evaluation and decision

All seven memory sizes M={3,8,16,32,64,128,256} and all sigma grids are
evaluated (6 isotropic, 5 structured). Each of 10 seeds has 77 conditions,
with 10,000 held-out trials per condition. The evaluation RNG namespace is
8,300,000, distinct from earlier Address-002; condition-specific fingerprints
are unique. **Every trained arm shares the exact same held-out batch**.

A joint success is allowed only if:
1. Mean mixed-minus-raw isotropic accuracy on the entire registered grid
   is **>= -0.02**.
2. Mean mixed-minus-raw accuracy at structured M=32, sigma={0.2,0.4}
   is **>= +0.03**.

Both predicates must hold; passing only one is a scientific failure of the
joint hypothesis. Full condition/seed curves, structured-only reference,
controls, training provenance and seed-bootstrap uncertainty must be reported
even on a negative result.

## Independent validation

The measurement script is `scripts/run_plm_learned_address_003.py`.
The separate `scripts/validate_plm_learned_address_003.py` checks all
10 × 77 held-out fingerprints, 2×750 training schedules and training
hashes, six registered arms' observable controls, complete grids, frozen
source/contract hashes and the **recomputed** joint success rule.

A workflow pass only means the experiment executed and passed integrity;
it is **not** evidence that the scientific hypothesis succeeded. No
post-test selection or retroactive amendments based on observed accuracy.

**Claim boundary:** opaque synthetic associative addresses only. This
does not establish semantic retrieval, open-world memory, or history-scalable
computation C5.
