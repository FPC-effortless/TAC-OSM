# Universal live parameter enforcement

This change closes the size-policy bypasses before further research fitting.
It is an engineering change, not a trained model or predictive capability result.

## Contract

- Every supported fitting registration goes through one absolute check: both
  trainable and all deployed learned `nn.Parameter` values are 95,000–105,000,
  inclusive. Shared objects count once. Frozen learned tables count as deployed
  parameters; deterministic priors, buffers and runtime state are separate.
- Comparisons retain the independent maximum 1% relative parameter difference,
  real-source/data/order/exposure, training compute, tuning and evaluation gates.
  CPU time is efficiency metadata, never required to match, including for an
  inherited v2 schema label. No historical CPU observation is rewritten.
- `capability_development` requires `deployed_parameters` alongside its trainable
  count. Source-deferred registrations require `deployed_parameter_counts` for
  every arm. Final records additionally require actual runtime evidence.
- Any new or resumed fitting is prospective, regardless of study ID. Restoring
  an old optimizer without stepping it remains possible for original replay.
- `noncomparative_diagnostic` explicitly declares `no_new_model_fits: true` and
  cannot issue an optimizer update.

## Supported fitting API

```python
from tac_osm.parameter_budget import (
    budgeted_optimizer, checked_optimizer_step, save_training_checkpoint,
)

optimizer = budgeted_optimizer(
    model, torch.optim.AdamW, lr=learning_rate,
    budget_manifest="path/to/FAIR_COMPARISON.json",
    budget_receipt="path/to/native-seed-1.runtime.json",
    budget_seed=1, budget_architecture="native_tac",
)
# Existing forward/backward loop remains unchanged.
checked_optimizer_step(optimizer)
save_training_checkpoint({"model": model.state_dict(),
                          "optimizer": optimizer.state_dict()}, "original.pt")
```

The pre-update hook enumerates the whole model and checks all trainables belong
to the optimizer exactly once, without foreign parameters. It rejects ambiguous
arm/seed registration. It writes an atomic, fsynced pre-update receipt before
parameters can change. Subsequent updates recheck inventory and the immutable
registration digest. Closures that can change topology inside an update are
unsupported and rejected. Optimizer restoration alone does not generate proof.

Prefer the guarded launcher, which also rejects unregistered optimizer updates:

```sh
python scripts/guarded_research.py --manifest path/to/FAIR_COMPARISON.json \
  --script scripts/new_research_entrypoint.py -- --existing-runner-arguments
python scripts/audit_training_budget.py
```

All current default-branch PyTorch training constructors, optimizer-step calls
and checkpoint writes in `scripts/` and `src/` use these adapters. The AST audit
checks supported imports/aliases, dynamic optimizer construction and direct
updates/saves. Legacy fit workflows require explicit manual opt-in; a PR or
push marker no longer starts their fits. Opt-in still does not bypass live gates.
Unit and static checks remain automatic through root and scientific-policy CI.

## Final evidence

Preserve the original prefit JSON bytes and set `prefit_registration_sha256`
in a separate final registration. Each arm's `measured` record (or each
capability/deferred `completed_model_records` row) includes:

```json
{
  "runtime_evidence": {
    "receipt_path": "original.runtime.json",
    "receipt_sha256": "SHA256_OF_THE_RECEIPT",
    "checkpoint_path": "original.pt",
    "checkpoint_sha256": "SHA256_OF_ORIGINAL_CHECKPOINT"
  }
}
```

Paths resolve from the validation working directory; use repository-relative
paths for portable evidence. Run normal `--phase final`. The validator requires
proof preceding update zero, completed guarded updates, matching seed/arm and
prefit digest, declared/budgeted optimizer-step consistency, separate receipts per fit, actual checkpoint hashes, named tensor
shapes/dtypes/counts and weights matching the runtime snapshot. Parameter checks
alone never qualify architecture superiority or validate source data semantics.

## Historical preservation

`FROZEN_REPLAY_REGISTRY.json` pins original ME-071, ME-072 and ME-073 registrations
and five original checkpoints per study to commit
`166d60d8d897f1f8be5f48410e7d90b8af46048c`. It does not retroactively certify their
training under this new runtime guard. No original model, result, protocol or
registration bytes are modified. Other historical studies can be registered
only with independently verified immutable provenance, not a numeric ID cutoff.

```sh
git fetch origin 166d60d8d897f1f8be5f48410e7d90b8af46048c
python scripts/verify_frozen_replay.py --all
python scripts/validate_research_parity.py --manifest original-prefit.json \
  --operation replay --provenance me073-prefit
```

This verifies original Git bytes only. It never issues training permission,
recomputes scientific scores or declares a predictive improvement.

## Limits

This enforces supported repository PyTorch fitting entrypoints. It is not a
sandbox against hostile Python, hand-written tensor updates, modified guards,
external uninstrumented training frameworks or dishonest research metadata.
Receipts provide an auditable execution contract, not cryptographic attestation
of an untrusted machine. Repository review and CI remain compulsory. Historical
research branches do not acquire enforcement until the change is integrated
there; default-branch protection and merge requirements are separate settings.
