# PLM functional repair: signed Boolean CASM

## Status

Development-only engineering lane. No held-out evaluation is performed here.

## Why this repair exists

Admissible E2E-005 result run 50 established that the original fixed probability-space
CASM did not learn the task family: mean held-out q2 accuracy was 0.512, while
the oracle was 1.0 and all integrity gates passed.

A separate training-distribution diagnostic showed the model also remained near
chance on fresh episodes drawn from the ordinary training-composition distribution.
An exact-payload CASM isolation reproduced the failure without any multimodal
encoder, localising the functional learning failure to the differentiable Boolean
CASM objective rather than to the multimodal input path alone.

The original CASM decoded bit probabilities and applied polynomial Boolean
functions directly to those probabilities. Around uncertain bits, XOR/XNOR have
zero first derivative at the symmetric point; AND/OR can also sit at marginal
stationary behavior under balanced targets. This creates a poor learning path
even though gradients technically reach the CASM.

## Repair

The repair decodes signed latent logits and defines a differentiable operator
margin:

- XOR: -a*b
- AND: min(a,b)
- OR: max(a,b)
- XNOR: a*b

The environment action is sigmoid(margin). Therefore the action threshold
at 0.5 is exactly the Boolean truth table induced by the signs of the two
latent logits. This changes the learning geometry without adding payload labels,
answer inputs, or a separate representation target.

## Development gate

The development runner uses three fixed seeds, the registered 300-step schedule,
and fresh episodes sampled only from TRAIN_COMBOS. It explicitly checks that:

1. no E2E-005 held-out composition is evaluated;
2. training and development semantic episode keys do not overlap;
3. the model API remains label-free;
4. the full multimodal gradient surface is present; and
5. the exact signed-Boolean truth table is preserved.

The functionality threshold is a development acceptance criterion only:
mean q2 >= 0.90 and minimum seed q2 >= 0.80.

It does not license a scientific generalization claim.

## Selection quarantine

The E2E-005 evaluation set is permanently retired from model selection because
its outcome is known. A repaired model must receive a new pre-registered,
freshly generated confirmatory holdout before any scientific capability claim.

E2E-006 is blocked until this training-only functionality lane is successful.