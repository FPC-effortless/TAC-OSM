"""One absolute parameter policy and live PyTorch optimizer/checkpoint guard.

No study-name exception authorizes a new optimizer update. Manifest validation
is declaration checking; runtime receipts and checkpoint verification are separate.
This protects supported research entrypoints, not arbitrary hostile Python code.
"""

from __future__ import annotations
from collections.abc import Mapping
from contextvars import ContextVar
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import tempfile
import uuid
import weakref

MIN_PARAMETERS, MAX_PARAMETERS = 95_000, 105_000
REPOSITORY = Path(__file__).resolve().parents[2]
_ACTIVE_MANIFEST = ContextVar("tac_training_manifest", default=None)
_GUARDS = weakref.WeakSet()
_MODEL_GUARDS = weakref.WeakKeyDictionary()


class ParameterBudgetError(ValueError):
    pass


def check_parameter_budget(trainable, deployed, *, label="model"):
    for name, n in (("trainable", trainable), ("deployed", deployed)):
        if (
            isinstance(n, bool)
            or not isinstance(n, int)
            or not MIN_PARAMETERS <= n <= MAX_PARAMETERS
        ):
            raise ParameterBudgetError(
                f"{label}: 100k target requires 95,000..105,000 {name} parameters, got {n}"
            )
    if deployed < trainable:
        raise ParameterBudgetError(
            f"{label}: deployed learned parameters cannot be below trainable count"
        )
    return {"trainable_parameters": trainable, "deployed_parameters": deployed}


def canonical_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def declared_budgets(manifest):
    """Normalize all supported fitting registrations without bypass branches."""
    kind = manifest.get("study_type")
    if manifest.get("source_based_deferred_registration") is True:
        trained = manifest.get("parameter_counts", {})
        deployed = manifest.get("deployed_parameter_counts", {})
        if not trained or set(trained) != set(deployed):
            raise ParameterBudgetError(
                "source-deferred registration needs explicit trainable AND deployed counts for every arm"
            )
        return [
            dict(
                architecture=k,
                trainable_parameters=n,
                deployed_parameters=deployed[k],
                seed=s,
            )
            for s in manifest.get("seeds", [])
            for k, n in trained.items()
        ]
    if kind == "architecture_comparison":
        return manifest.get("arms", [])
    if kind == "single_model_training":
        return manifest.get("models", [])
    if kind == "capability_development":
        return [
            dict(
                seed=s,
                trainable_parameters=manifest.get("trainable_parameters"),
                deployed_parameters=manifest.get("deployed_parameters"),
            )
            for s in manifest.get("seeds", [])
        ]
    if kind == "noncomparative_diagnostic":
        if manifest.get("no_new_model_fits") is not True:
            raise ParameterBudgetError(
                "no-fit diagnostics must explicitly forbid new model fits"
            )
        return []
    raise ParameterBudgetError("unsupported training registration type")


def check_declared_budgets(manifest):
    rows = declared_budgets(manifest)
    if manifest.get("study_type") != "noncomparative_diagnostic" and not rows:
        raise ParameterBudgetError(
            "all fitting paths require model/seed parameter budgets"
        )
    for i, row in enumerate(rows):
        check_parameter_budget(
            row.get("trainable_parameters"),
            row.get("deployed_parameters"),
            label=f"model[{i}]",
        )
    return rows


def live_inventory(model):
    rows = []
    for name, p in model.named_parameters():
        try:
            n = p.numel()
            shape = list(p.shape)
        except (ValueError, RuntimeError) as error:
            raise ParameterBudgetError(
                "initialize lazy parameters before training"
            ) from error
        rows.append(
            dict(
                name=name,
                shape=shape,
                numel=n,
                requires_grad=bool(p.requires_grad),
                dtype=str(p.dtype),
            )
        )
    counts = dict(
        trainable_parameters=sum(r["numel"] for r in rows if r["requires_grad"]),
        deployed_parameters=sum(r["numel"] for r in rows),
    )
    check_parameter_budget(
        **{
            "trainable": counts["trainable_parameters"],
            "deployed": counts["deployed_parameters"],
        }
    )
    return counts, rows


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        temp = Path(f.name)
        try:
            json.dump(value, f, sort_keys=True, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def parameter_values_sha(state, inventory):
    import torch

    digest = hashlib.sha256()
    for row in inventory:
        value = state.get(row["name"])
        if (
            not isinstance(value, torch.Tensor)
            or list(value.shape) != row["shape"]
            or str(value.dtype) != row["dtype"]
        ):
            raise ParameterBudgetError(
                "checkpoint parameter inventory differs from the live model: "
                + row["name"]
            )
        digest.update(row["name"].encode())
        digest.update(
            bytes(
                value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).tolist()
            )
        )
    return digest.hexdigest()


def _find_state(obj, inventory, path=()):
    if not isinstance(obj, Mapping):
        return None
    if all(r["name"] in obj for r in inventory):
        return path, obj
    for name, value in obj.items():
        if isinstance(name, str) and isinstance(value, Mapping):
            found = _find_state(value, inventory, path + (name,))
            if found is not None:
                return found
    return None


class _LiveGuard:
    def __init__(
        self, model, optimizer, manifest_path, receipt_path, seed, architecture
    ):
        self.model = weakref.ref(model)
        self.optimizer = weakref.ref(optimizer)
        self.manifest_path = Path(manifest_path).resolve() if manifest_path else None
        self.path = Path(receipt_path) if receipt_path else None
        self.seed = seed
        self.architecture = architecture
        self.steps = 0
        self.inventory = None
        self.receipt = {
            "schema": "tac.live-parameter-budget.v1",
            "run_id": uuid.uuid4().hex,
            "status": "NOT_STARTED",
            "successful_optimizer_steps": 0,
            "checkpoints": [],
        }
        _GUARDS.add(self)

    def _model(self):
        model = self.model()
        if model is None:
            raise ParameterBudgetError("the guarded whole model is no longer available")
        return model

    def before(self, optimizer, args, kwargs):
        # A closure can change the model after a pre-step hook: unsupported
        # closures are blocked rather than claiming a check at the wrong time.
        if kwargs.get("closure") is not None or len(args) > 1 and args[1] is not None:
            raise ParameterBudgetError(
                "closure-based fitting requires a separately audited adapter"
            )
        model = self._model()
        counts, inventory = live_inventory(model)
        own = {id(p) for p in model.parameters()}
        active = {id(p) for p in model.parameters() if p.requires_grad}
        optimized = [p for group in optimizer.param_groups for p in group["params"]]
        ids = {id(p) for p in optimized}
        if len(optimized) != len(ids) or not active <= ids or not ids <= own:
            raise ParameterBudgetError(
                "optimizer must contain the whole model trainable parameters exactly once and no foreign parameters"
            )
        if self.inventory is None:
            if self.manifest_path is None:
                raise ParameterBudgetError(
                    "new fitting needs an explicit preregistered manifest; old IDs do not grant permission"
                )
            manifest = json.loads(self.manifest_path.read_text())
            # Import only at execution time; the CLI validator stays Torch-free.
            import sys

            if str(REPOSITORY) not in sys.path:
                sys.path.insert(0, str(REPOSITORY))
            from scripts.validate_research_parity import check_manifest

            check_manifest(manifest, "preflight")
            if manifest.get("study_type") == "noncomparative_diagnostic":
                raise ParameterBudgetError(
                    "no-fit diagnostic cannot execute an optimizer update"
                )
            declared = check_declared_budgets(manifest)
            candidates = [
                r
                for r in declared
                if r["trainable_parameters"] == counts["trainable_parameters"]
                and r["deployed_parameters"] == counts["deployed_parameters"]
            ]
            if self.seed is not None:
                candidates = [r for r in candidates if r.get("seed") == self.seed]
            if self.architecture is not None:
                candidates = [
                    r
                    for r in candidates
                    if r.get("architecture", self.architecture) == self.architecture
                ]
            if not candidates:
                raise ParameterBudgetError(
                    "live model count/seed/arm differs from its frozen registration"
                )
            if self.seed is None:
                seeds = {r.get("seed") for r in candidates}
                if len(seeds) != 1:
                    raise ParameterBudgetError(
                        "ambiguous model seed; supply budget_seed before fitting"
                    )
                self.seed = seeds.pop()
            if self.architecture is None:
                arms = {r.get("architecture") for r in candidates}
                if len(arms) > 1:
                    raise ParameterBudgetError(
                        "ambiguous architecture; supply budget_architecture before fitting"
                    )
                self.architecture = arms.pop()
            self.inventory = inventory
            if self.path is None:
                self.path = (
                    self.manifest_path.parent
                    / "runtime-parameter-receipts"
                    / (self.receipt["run_id"] + ".json")
                )
            self.receipt.update(
                study_id=manifest["study_id"],
                manifest_sha256=file_sha(self.manifest_path),
                manifest_path=str(self.manifest_path),
                seed=self.seed,
                architecture=self.architecture,
                preflight={
                    **counts,
                    "optimizer_steps_before_check": 0,
                    "parameter_inventory": inventory,
                },
                model_class=type(model).__module__ + "." + type(model).__qualname__,
            )
        if inventory != self.inventory:
            raise ParameterBudgetError(
                "live model inventory changed after the original pre-update check"
            )
        if file_sha(self.manifest_path) != self.receipt["manifest_sha256"]:
            raise ParameterBudgetError("frozen preregistration changed during fitting")
        self.receipt.update(
            status="PRE_UPDATE_VERIFIED",
            last_pre_update={**counts, "successful_optimizer_steps": self.steps},
        )
        atomic_json(self.path, self.receipt)  # failure stops BEFORE optimizer.step
        _MODEL_GUARDS[model] = self

    def after(self, optimizer, args, kwargs):
        counts, inventory = live_inventory(self._model())
        if inventory != self.inventory:
            raise ParameterBudgetError(
                "optimizer changed model topology or trainability during update"
            )
        self.steps += 1
        self.receipt.update(
            status="POST_UPDATE_VERIFIED",
            successful_optimizer_steps=self.steps,
            last_post_update=counts,
        )
        atomic_json(self.path, self.receipt)

    def checkpoint(self, path, state_path, state):
        model = self._model()
        counts, inventory = live_inventory(model)
        if self.steps < 1 or inventory != self.inventory:
            raise ParameterBudgetError(
                "checkpoint has no qualifying guarded update or changed model inventory"
            )
        actual = parameter_values_sha(model.state_dict(), inventory)
        if parameter_values_sha(state, inventory) != actual:
            raise ParameterBudgetError(
                "saved checkpoint does not contain the guarded live model weights"
            )
        row = dict(
            path=str(Path(path)),
            sha256=file_sha(path),
            state_key_path=list(state_path),
            parameter_values_sha256=actual,
            successful_optimizer_steps=self.steps,
            **counts,
        )
        self.receipt["checkpoints"] = [
            r for r in self.receipt["checkpoints"] if r["path"] != row["path"]
        ] + [row]
        self.receipt["status"] = "CHECKPOINT_VERIFIED"
        atomic_json(self.path, self.receipt)


def budgeted_optimizer(
    model,
    optimizer_type,
    parameters=None,
    *args,
    budget_manifest=None,
    budget_receipt=None,
    budget_seed=None,
    budget_architecture=None,
    **kwargs,
):
    """Attach non-optional pre/post hooks; optimizer restore alone is not a fit.

    Counts are checked at the first/every actual update, not constructor time,
    so old optimizers can be restored for inference without a training exception.
    """
    caller_frame = inspect.currentframe().f_back
    if budget_seed is None and isinstance(caller_frame.f_locals.get("seed"), int):
        budget_seed = caller_frame.f_locals["seed"]
    if budget_architecture is None:
        budget_architecture = getattr(model, "architecture", None)
    if budget_manifest is None:
        budget_manifest = _ACTIVE_MANIFEST.get()
    if budget_manifest is None:
        # Only locate a file; never exempt a study based on its name.
        import re

        caller = Path(inspect.currentframe().f_back.f_code.co_filename)
        match = re.match(r"(me\d{2,3}|dev\d{2,3})(?:_|\.)", caller.name, re.I)
        if match:
            possible = (
                REPOSITORY
                / "research/mathematical-engineering"
                / match.group(1).lower()
                / "FAIR_COMPARISON.json"
            )
            if possible.exists():
                budget_manifest = possible
    optimizer = optimizer_type(
        model.parameters() if parameters is None else parameters, *args, **kwargs
    )
    guard = _LiveGuard(
        model,
        optimizer,
        budget_manifest,
        budget_receipt,
        budget_seed,
        budget_architecture,
    )
    optimizer._tac_parameter_budget_guard = guard
    guard.pre_handle = optimizer.register_step_pre_hook(guard.before)
    guard.post_handle = optimizer.register_step_post_hook(guard.after)
    return optimizer


def checked_optimizer_step(optimizer, *args, **kwargs):
    guard = getattr(optimizer, "_tac_parameter_budget_guard", None)
    if (
        not isinstance(guard, _LiveGuard)
        or guard.optimizer() is not optimizer
        or guard.pre_handle.id not in optimizer._optimizer_step_pre_hooks
    ):
        raise ParameterBudgetError(
            "optimizer update has no whole-model parameter guard"
        )
    return optimizer.step(*args, **kwargs)


def save_training_checkpoint(obj, path, *args, **kwargs):
    """Preserve actual bytes, then bind matching trained models to their receipt.

    BytesIO/non-model serializations remain ordinary outputs, not proof of a fit.
    """
    import torch

    if not isinstance(path, (str, Path)):
        return torch.save(obj, path, *args, **kwargs)
    matches = []
    for guard in list(_GUARDS):
        if (
            guard.steps
            and guard.model() is not None
            and _MODEL_GUARDS.get(guard.model()) is guard
        ):
            found = _find_state(obj, guard.inventory)
            if found is not None:
                try:
                    if parameter_values_sha(
                        found[1], guard.inventory
                    ) == parameter_values_sha(
                        guard.model().state_dict(), guard.inventory
                    ):
                        matches.append((guard, found))
                except ParameterBudgetError:
                    pass
    torch.save(obj, path, *args, **kwargs)
    if isinstance(path, (str, Path)):
        for guard, (state_path, state) in matches:
            guard.checkpoint(path, state_path, state)


def validate_runtime_evidence(row):
    """Independently read the runtime receipt AND actual serialized tensors."""
    evidence = row.get("runtime_evidence")
    if not isinstance(evidence, dict):
        raise ParameterBudgetError(
            "final qualification requires live-model runtime evidence, not declared counts"
        )
    receipt_path = Path(evidence["receipt_path"])
    if file_sha(receipt_path) != evidence.get("receipt_sha256"):
        raise ParameterBudgetError("runtime receipt SHA mismatch")
    receipt = json.loads(receipt_path.read_text())
    pre = receipt.get("preflight", {})
    if row.get("_prefit_registration_sha256") != receipt.get("manifest_sha256"):
        raise ParameterBudgetError(
            "runtime proof is not bound to the original frozen preregistration"
        )
    if (
        row.get("seed") != receipt.get("seed")
        or row.get("architecture") is not None
        and row.get("architecture") != receipt.get("architecture")
    ):
        raise ParameterBudgetError("runtime proof seed/architecture differs")
    if receipt.get("status") != "CHECKPOINT_VERIFIED":
        raise ParameterBudgetError("runtime receipt has no final qualifying checkpoint")
    check_parameter_budget(
        pre.get("trainable_parameters"), pre.get("deployed_parameters")
    )
    if (
        pre.get("optimizer_steps_before_check") != 0
        or receipt.get("successful_optimizer_steps", 0) < 1
    ):
        raise ParameterBudgetError(
            "runtime proof does not precede first update or has no completed updates"
        )
    for k in ("trainable_parameters", "deployed_parameters"):
        if row.get(k) != pre[k] or receipt.get("last_post_update", {}).get(k) != pre[k]:
            raise ParameterBudgetError("registered/runtime pre/post counts differ")
    inventory = pre.get("parameter_inventory", [])
    if (
        not inventory
        or sum(r["numel"] for r in inventory) != pre["deployed_parameters"]
        or sum(r["numel"] for r in inventory if r["requires_grad"])
        != pre["trainable_parameters"]
    ):
        raise ParameterBudgetError("runtime named-parameter inventory/count mismatch")
    checkpoint = Path(evidence["checkpoint_path"])
    checkpoint_hash = file_sha(checkpoint)
    if row.get("checkpoint_sha256", checkpoint_hash) != checkpoint_hash:
        raise ParameterBudgetError("declared checkpoint SHA differs from actual bytes")
    if checkpoint_hash != evidence.get("checkpoint_sha256"):
        raise ParameterBudgetError("actual checkpoint SHA mismatch")
    records = [
        r for r in receipt.get("checkpoints", []) if r["sha256"] == checkpoint_hash
    ]
    if not records:
        raise ParameterBudgetError("checkpoint is not bound to a guarded live model")
    saved = __import__("torch").load(checkpoint, map_location="cpu", weights_only=False)
    state = saved
    for key in records[-1]["state_key_path"]:
        state = state[key]
    if parameter_values_sha(state, inventory) != records[-1]["parameter_values_sha256"]:
        raise ParameterBudgetError(
            "checkpoint tensors differ from the guarded model snapshot"
        )
    for r in inventory:
        if state[r["name"]].numel() != r["numel"]:
            raise ParameterBudgetError("actual checkpoint parameter size differs")
    return receipt["run_id"]


def install_unbound_optimizer_guard():
    """Backstop a controlled training process, including dynamically made opts."""
    from torch.optim.optimizer import register_optimizer_step_pre_hook

    def before(optimizer, args, kwargs):
        guard = getattr(optimizer, "_tac_parameter_budget_guard", None)
        if (
            not isinstance(guard, _LiveGuard)
            or guard.optimizer() is not optimizer
            or guard.pre_handle.id not in optimizer._optimizer_step_pre_hooks
        ):
            raise ParameterBudgetError(
                "unregistered optimizer update in a guarded research process"
            )

    return register_optimizer_step_pre_hook(before)
