"""ME106 fixed fresh-fit panel for protected median split sensor forecasts.

This module deliberately exposes prepare/fit/score/replay as separate commands.
Importing it and running the unit tests never performs an optimizer update.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import random
import resource
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import me100_sensor_data as data
from scripts import me102_gap_data as gap
from scripts import me101_belief_sensor as me101
from scripts import me102_reconstruction_panel as me102
from scripts.me106_gradient_roles import objective_parts, assign_priority_gradients
from scripts.validate_research_parity import check_manifest
from tac_osm.parameter_budget import (
    budgeted_optimizer,
    checked_optimizer_step,
    save_training_checkpoint,
    validate_runtime_evidence,
)
from tac_osm.variants.point_priority_sensor_world import PointPrioritySensorWorld

OUT = ROOT / "research/replay-bundles/ME106"
DOC = ROOT / "research/mathematical-engineering/me106"
REG = DOC / "FAIR_COMPARISON.json"
PARENT = ROOT / "research/replay-bundles/ME103"
PARENT_DOC = ROOT / "research/mathematical-engineering/me103"
SEEDS = tuple(range(106001, 106006))
PARENT_SEEDS = tuple(range(102001, 102006))
STEPS = 1536
BATCH = 16
CONTEXTS = (8, 64, 256)
RECIPES = {
    "balanced": {"point": 0.5, "interval": 0.25},
    "point_guard": {"point": 1.0, "interval": 0.25},
    "point_strong": {"point": 2.0, "interval": 0.25},
}
# Include the whole inherited model closure and the ME106 additions.
SOURCES = list(dict.fromkeys(
    list(me102.SOURCES)
    + ["scripts/me101_belief_sensor.py",
       "src/tac_osm/variants/median_split_sensor_world.py",
       "src/tac_osm/variants/protected_sensor_world.py",
       "src/tac_osm/variants/point_priority_sensor_world.py",
       "scripts/me105_protected_panel.py", "scripts/me105_graph_reference.py",
       "scripts/me106_gradient_roles.py",
       "scripts/me102_graph_reference.py", "scripts/me104_graph_reference.py",
       "scripts/me106_graph_reference.py", "scripts/me106_priority_panel.py"]
))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


@lru_cache(maxsize=1)
def fit_data():
    """Read the immutable FIT partition once; all augmentation uses derived data."""
    return data.load("fit")


def tensor_batch(d, origins, length, rng):
    return [torch.from_numpy(a) for a in gap.augmentation(d, origins, length, rng)]


def make_plan(seed, d):
    """Build fixed half-gap/half-general batches shared by all three recipes."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(STEPS):
        rows.append(
            [d["gap"][i] for i in rng.integers(len(d["gap"]), size=BATCH // 2)]
            + [d["general"][i] for i in rng.integers(len(d["general"]), size=BATCH // 2)]
        )
    return np.asarray(rows, dtype=np.int64)


def initialize(d, seed):
    """Create a 100,349-parameter trainable model with normalized target stats."""
    torch.manual_seed(seed)
    model = PointPrioritySensorWorld(
        torch.tensor(gap.floor_rates(d)["rates"], dtype=torch.float32)
    )
    with torch.no_grad():
        model.target_mean.copy_(torch.from_numpy(d["mean"][:6]))
        model.target_scale.copy_(torch.from_numpy(d["std"][:6]))
    return model


def optimizer_parameter_groups(model):
    """Return fixed inherited/new-head groups with complete parameter accounting."""
    named = list(model.named_parameters())
    inherited = [parameter for name, parameter in named
                 if not name.startswith(("width_head.", "median_head."))]
    additions = [parameter for name, parameter in named
                 if name.startswith(("width_head.", "median_head."))]
    grouped = inherited + additions
    if len(grouped) != len(named) or len({id(parameter) for parameter in grouped}) != len(named):
        raise RuntimeError("optimizer groups must account for every model parameter exactly once")
    if {id(p) for p in additions} != {id(p) for name, p in named if name.startswith(("width_head.", "median_head."))}:
        raise RuntimeError("new heads are not assigned to the high-rate optimizer group")
    return [dict(params=inherited, lr=0.000003), dict(params=additions, lr=0.0003)]


def load_parent(model, checkpoint):
    """Load the complete ME103 parent; allow only the new zero width and median heads."""
    result = model.load_state_dict(checkpoint["model"], strict=False)
    if result.missing_keys != ["width_head.weight", "width_head.bias", "median_head.weight", "median_head.bias"] or result.unexpected_keys:
        raise RuntimeError(f"parent checkpoint architecture mismatch: {result}")
    if not torch.equal(model.width_head.weight, torch.zeros_like(model.width_head.weight)):
        raise RuntimeError("new width head weight is not exactly zero")
    if not torch.equal(model.width_head.bias, torch.zeros_like(model.width_head.bias)):
        raise RuntimeError("new width head bias is not exactly zero")
    for head in (model.width_head, model.median_head):
        if not torch.equal(head.weight, torch.zeros_like(head.weight)) or not torch.equal(head.bias, torch.zeros_like(head.bias)):
            raise RuntimeError("new protected-sensor heads are not exactly zero")
    return result


def objective(model, forecast, trace, batch, rawscale, recipe):
    """Return the registered objective and its independently projected roles."""
    return objective_parts(model, forecast, trace, batch, rawscale, RECIPES[recipe])


def objective_component_gradient_probe(model, parts):
    """Measure each actual objective term's gradients into model parameter blocks.

    This is a prefit diagnostic based on ``autograd.grad`` over the instantiated
    graph; it does not step an optimizer or infer influence symbolically.
    """
    params = list(model.named_parameters())
    blocks = {
        "inherited": [i for i, (name, _) in enumerate(params)
                      if not name.startswith(("width_head.", "median_head."))],
        "width_head": [i for i, (name, _) in enumerate(params) if name.startswith("width_head.")],
        "median_head": [i for i, (name, _) in enumerate(params) if name.startswith("median_head.")],
    }
    values = tuple(parameter for _, parameter in params)
    result = {}
    for term_name, term in parts.items():
        grads = torch.autograd.grad(term, values, retain_graph=True, allow_unused=True)
        result[term_name] = {}
        for block, indices in blocks.items():
            tensors = [grads[i] for i in indices if grads[i] is not None]
            result[term_name][block] = dict(
                gradient_tensors=len(tensors),
                nonzero_tensors=sum(bool(torch.count_nonzero(grad)) for grad in tensors),
                norm=float(torch.sqrt(sum(grad.detach().square().sum() for grad in tensors))) if tensors else 0.0,
            )
    return result


def profile(model, batch, rawscale, recipe):
    model.zero_grad(set_to_none=True)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True, record_shapes=True
    ) as prof:
        forecast, _, trace = model(*batch[:6], return_trace=True)
        loss, parts = objective(model, forecast, trace, batch, rawscale, recipe)
        assign_priority_gradients(model, parts, RECIPES[recipe]["point"], RECIPES[recipe]["interval"])
    events = prof.key_averages(group_by_input_shape=True)
    ops = sorted((event.key, event.count, str(event.input_shapes)) for event in events if event.key.startswith("aten::"))
    return dict(dense_flops=sum(event.flops for event in events), aten_signature=digest(ops),
                aten_calls=sum(event.count for event in events if event.key.startswith("aten::")))


def _parent(recipe, seed):
    path = PARENT / recipe / f"seed-{seed}" / "final.pt"
    row = next(r for r in json.loads((PARENT / "FIT_INDEX.json").read_text())["models"]
               if r["architecture"] == recipe and r["seed"] == seed)
    if sha(path) != row["checkpoint_sha256"]:
        raise RuntimeError("ME103 parent checkpoint hash mismatch")
    cp = torch.load(path, map_location="cpu", weights_only=False)
    if cp.get("steps") != 3072 or cp.get("recipe") != recipe or cp.get("seed") != seed:
        raise RuntimeError("unexpected ME103 parent lineage")
    return path, cp, row


def _frozen(paths):
    for path in paths:
        rel = Path(path).relative_to(ROOT)
        raw = subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=ROOT, stderr=subprocess.DEVNULL)
        if raw != Path(path).read_bytes():
            raise RuntimeError(f"file is not Git-frozen: {rel}")


def prepare():
    if OUT.exists() or any((DOC / name).exists() for name in (
        "FIT_PLANS.npz", "PREFIT_PROFILES.json", "PREFIT_DESIGN_RECEIPT.json"
    )):
        raise RuntimeError("ME106 artifacts already exist; no overwrite or retry")
    if not REG.exists():
        raise RuntimeError("main-owned ME106 preregistration is required before prepare")
    manifest = json.loads(REG.read_text())
    d = fit_data()
    plans = {str(seed): make_plan(seed, d) for seed in SEEDS}
    np.savez_compressed(DOC / "FIT_PLANS.npz", **plans)
    plan_hashes = {seed: hashlib.sha256(plans[str(seed)].tobytes()).hexdigest() for seed in SEEDS}
    rawscale = torch.tensor(np.nanstd(d["raw"][..., :6].astype(np.float64), axis=(0, 1)).clip(1e-4), dtype=torch.float32)
    profiles = {}
    parent_hashes = {}
    init_audits = []
    for recipe in RECIPES:
        profiles[recipe] = {}
        for seed, parent_seed in zip(SEEDS, PARENT_SEEDS):
            path, cp, row = _parent("strong_reconstruction", parent_seed)
            parent_folder = PARENT / "strong_reconstruction" / f"seed-{parent_seed}"
            parent_hashes[str(seed)] = dict(
                checkpoint_sha256=sha(path),
                fit_receipt_sha256=sha(parent_folder / "FIT_RECEIPT.json"),
                runtime_receipt_sha256=sha(parent_folder / "live-budget.json"),
            )
            model = initialize(d, seed)
            load_parent(model, cp)
            parent_model = me102.initialize(d, "strong_reconstruction", parent_seed)
            parent_model.load_state_dict(cp["model"], strict=True)
            parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            deployed_count = sum(p.numel() for p in model.parameters())
            if parameter_count != 100349 or deployed_count != 100349 or len(list(model.parameters())) != 33:
                raise RuntimeError("instantiated ME106 model failed the prefit parameter guard")
            if not all(p.requires_grad for p in model.parameters()):
                raise RuntimeError("ME106 contains a frozen model tensor")
            first_rng = np.random.default_rng(seed + 10000)
            first_batch = tensor_batch(d, plans[str(seed)][0], CONTEXTS[0], first_rng)
            model.zero_grad(set_to_none=True)
            first_forecast, _, first_trace = model(*first_batch[:6], return_trace=True)
            parent_forecast, _ = parent_model(*first_batch[:6])
            uncertainty_view = model.uncertainty_forecast(first_forecast, first_trace)
            forward_identity = torch.equal(model.raw_quantiles(first_forecast), model.raw_quantiles(uncertainty_view))
            if not forward_identity:
                raise RuntimeError("gradient-role view changed the forecast law")
            if not torch.equal(model.raw_quantiles(first_forecast), parent_model.raw_quantiles(parent_forecast)):
                raise RuntimeError("zero-width initialization changed parent quantiles")
            first_loss, first_parts = objective(model, first_forecast, first_trace, first_batch, rawscale, recipe)
            if not torch.isfinite(first_loss):
                raise RuntimeError("nonfinite initial loss during no-step prefit audit")
            component_gradients = objective_component_gradient_probe(model, first_parts)
            gradient_audit = assign_priority_gradients(model, first_parts, RECIPES[recipe]["point"], RECIPES[recipe]["interval"])
            grads = {name: parameter.grad for name, parameter in model.named_parameters()}
            if len(grads) != 33 or any(g is None or not torch.isfinite(g).all() or not torch.count_nonzero(g) for g in grads.values()):
                raise RuntimeError("initial all-parameter gradient guard failed")
            init_audits.append(dict(
                recipe=recipe, seed=seed, parent_seed=parent_seed,
                trainable_parameters=parameter_count, deployed_parameters=deployed_count,
                parent_quantiles_exact=True, gradient_view_forward_identity=forward_identity,
                input_sha256=hashlib.sha256(b"".join(v.numpy().tobytes() for v in first_batch[:4])).hexdigest(),
                loss=float(first_loss.detach()), loss_parts={k: float(v.detach()) for k, v in first_parts.items()},
                objective_component_gradients=component_gradients,
                parameter_tensors=len(grads), nonzero_gradient_tensors=sum(bool(torch.count_nonzero(g)) for g in grads.values()),
                gradient_assignment=gradient_audit,
                gradient_sha256=digest({name: hashlib.sha256(g.detach().contiguous().numpy().tobytes()).hexdigest() for name, g in grads.items()}),
                no_optimizer_step=True,
            ))
            for length in CONTEXTS:
                x = tensor_batch(d, plans[str(seed)][0], length, np.random.default_rng(seed + 10000))
                if str(length) not in profiles[recipe]:
                    profiles[recipe][str(length)] = profile(model, x, rawscale, recipe)
    for length in CONTEXTS:
        rows = [profiles[r][str(length)] for r in RECIPES]
        if len({r["aten_signature"] for r in rows}) != 1 or len({r["dense_flops"] for r in rows}) != 1:
            raise RuntimeError("prefit graph/profile parity failed")
    for seed in SEEDS:
        rows = [r for r in init_audits if r["seed"] == seed]
        if len(rows) != len(RECIPES) or len({r["input_sha256"] for r in rows}) != 1:
            raise RuntimeError("same-seed prefit augmented input parity failed")
    write(DOC / "PREFIT_PROFILES.json", dict(profiles=profiles, no_optimizer_updates=True))
    write(DOC / "PREFIT_DESIGN_RECEIPT.json", dict(
        trainable_parameters=sorted({row["trainable_parameters"] for row in init_audits}),
        deployed_parameters=sorted({row["deployed_parameters"] for row in init_audits}),
        parameter_tensors=sorted({row["parameter_tensors"] for row in init_audits}), all_parameters_trainable=True,
        seeds=list(SEEDS), parent_hashes=parent_hashes, plan_sha256=plan_hashes,
        profiles=profiles, init_audits=init_audits, no_optimizer_updates=True,
    ))
    # Finalize the main-authored draft with measured plans, instantiated counts,
    # profiled compute, and the complete source closure before preflight passes.
    template = manifest["arms"][0]
    profile_flops = sum(profiles["balanced"][str(n)]["dense_flops"] for n in CONTEXTS) * (STEPS // len(CONTEXTS))
    common = dict(
        dataset_sha256=data.SHA,
        training_sample_order_sha256="",
    )
    new_arms = []
    for seed in SEEDS:
        for recipe in RECIPES:
            arm = json.loads(json.dumps(template))
            arm.update(common)
            arm.update(architecture=recipe, seed=seed,
                       trainable_parameters=100349, deployed_parameters=100349,
                       training_sample_order_sha256=plan_hashes[seed])
            budget = dict(arm["budget"])
            budget.update(steps=STEPS, examples_seen=STEPS * BATCH,
                          tokens_seen=512 * BATCH * sum(CONTEXTS), batch_size=BATCH,
                          history_events=max(CONTEXTS), planned_training_flops=profile_flops,
                          max_training_flops=profile_flops * 1.01,
                          tuning_trials=1, tuning_flops=0,
                          prefit_profile_training_flops=profile_flops,
                          prefit_profile_method="torch.profiler dense forward/backward estimate plus exact ATen operation/shape signature parity; unsupported scalar and optimizer FLOPs are partial")
            arm["budget"] = budget
            new_arms.append(arm)
    manifest.update(arms=new_arms, recipes=RECIPES,
                    source_files={rel: sha(ROOT / rel) for rel in SOURCES},
                    parent_artifacts=parent_hashes,
                    parent_fit_index_sha256=sha(PARENT / "FIT_INDEX.json"),
                    parent_registration_sha256=sha(PARENT_DOC / "FAIR_COMPARISON.json"),
                    prefit_design_sha256=sha(DOC / "PREFIT_DESIGN_RECEIPT.json"),
                    fit_plans_sha256=sha(DOC / "FIT_PLANS.npz"),
                    prefit_profiles_sha256=sha(DOC / "PREFIT_PROFILES.json"))
    manifest["protocol"].update(
        seeds=list(SEEDS), updates=STEPS, batch=BATCH, contexts=list(CONTEXTS),
        optimizer="new AdamW per model; inherited group lr=0.000003, width_head+median_head group lr=0.0003, weight_decay=0.001; no parent optimizer state",
        parent="all five ME103 strong_reconstruction final checkpoints (102001..102005); load all parent model tensors and zero-initialize new width_head and median_head",
        plan="one default_rng(seed) plan per new seed, eight halfgap and eight halfgeneral origins per batch, shared byte-for-byte across recipes",
    )
    write(REG, manifest)
    check_manifest(manifest, "preflight")
    print("ME106_PREFIT_PANEL_PASS")


def fit_one(task):
    recipe, seed = task
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    registration = json.loads(REG.read_text())
    check_manifest(registration, "preflight")
    if registration.get("fit_plans_sha256") != sha(DOC / "FIT_PLANS.npz") or registration.get("prefit_profiles_sha256") != sha(DOC / "PREFIT_PROFILES.json"):
        raise RuntimeError("ME106 plan or prefit profiles changed after registration")
    if registration.get("prefit_design_sha256") != sha(DOC / "PREFIT_DESIGN_RECEIPT.json"):
        raise RuntimeError("ME106 actual prefit count/gradient audit changed")
    if registration.get("parent_fit_index_sha256") != sha(PARENT / "FIT_INDEX.json") or registration.get("parent_registration_sha256") != sha(PARENT_DOC / "FAIR_COMPARISON.json"):
        raise RuntimeError("ME103 parent registration/index changed after ME106 prepare")
    if registration.get("source_files") != {p: sha(ROOT / p) for p in SOURCES}:
        raise RuntimeError("ME106 source hashes differ from frozen registration")
    d = fit_data()
    parent_seed = seed - 4000
    parent_path, parent, _ = _parent("strong_reconstruction", parent_seed)
    parent_receipt = json.loads((DOC / "PREFIT_DESIGN_RECEIPT.json").read_text())["parent_hashes"][str(seed)]
    if sha(parent_path) != parent_receipt["checkpoint_sha256"]:
        raise RuntimeError("ME103 parent checkpoint differs from prefit registration")
    model = initialize(d, seed)
    load_parent(model, parent)
    folder = OUT / recipe / f"seed-{seed}"
    folder.mkdir(parents=True, exist_ok=False)
    groups = optimizer_parameter_groups(model)
    if sum(len(group["params"]) for group in groups) != 33:
        raise RuntimeError("optimizer groups must contain all 33 model tensors")
    optimizer = budgeted_optimizer(
        model, torch.optim.AdamW,
        parameters=groups,
        lr=0.000003, weight_decay=0.001,
        budget_manifest=REG, budget_receipt=folder / "live-budget.json",
        budget_seed=seed, budget_architecture=recipe,
    )
    if optimizer.state:
        raise RuntimeError("new AdamW must start with an empty state")
    plan = np.load(DOC / "FIT_PLANS.npz")[str(seed)]
    rng = np.random.default_rng(seed + 10000)
    rawscale = torch.tensor(np.nanstd(d["raw"][..., :6].astype(np.float64), axis=(0, 1)).clip(1e-4), dtype=torch.float32)
    input_hash = hashlib.sha256()
    gradient_receipt = None
    losses = []
    start_cpu, start_wall = time.process_time(), time.monotonic()
    for step, origins in enumerate(plan):
        length = CONTEXTS[step % len(CONTEXTS)]
        x = tensor_batch(d, origins, length, rng)
        for value in x[:4]:
            input_hash.update(value.numpy().tobytes())
        optimizer.zero_grad(set_to_none=True)
        forecast, _, trace = model(*x[:6], return_trace=True)
        loss, parts = objective(model, forecast, trace, x, rawscale, recipe)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite loss; no retry")
        gradient_audit = assign_priority_gradients(model, parts, RECIPES[recipe]["point"], RECIPES[recipe]["interval"])
        gradients = dict(model.named_parameters())
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in gradients.values()):
            raise RuntimeError("invalid or missing parameter gradient")
        if step == 0:
            gradient_receipt = {"parameters": {name: dict(norm=float(p.grad.norm()), nonzero=bool(torch.count_nonzero(p.grad)))
                                for name, p in gradients.items()},
                                "assignment": gradient_audit}
            if len(gradient_receipt["parameters"]) != 33 or any(not row["nonzero"] for row in gradient_receipt["parameters"].values()):
                raise RuntimeError("first-batch gradient guard failed")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        checked_optimizer_step(optimizer)
        losses.append({k: float(v.detach()) for k, v in parts.items()})
        if (step + 1) % 512 == 0:
            print(json.dumps(dict(recipe=recipe, seed=seed, step=step + 1)), flush=True)
    if len(optimizer.state) != 33:
        raise RuntimeError("optimizer did not retain state for all 33 tensors")
    profiles, profile_inputs = {}, {}
    for length in CONTEXTS:
        x = tensor_batch(d, plan[0], length, np.random.default_rng(seed + 10000))
        profile_inputs[str(length)] = hashlib.sha256(b"".join(v.numpy().tobytes() for v in x[:4])).hexdigest()
        profiles[str(length)] = profile(model, x, rawscale, recipe)
    path = folder / "final.pt"
    save_training_checkpoint(dict(
        model=model.state_dict(), optimizer=optimizer.state_dict(), recipe=recipe, seed=seed,
        steps=STEPS, plan=plan, parent_checkpoint_sha256=sha(parent_path), losses=losses,
        augmentation_rng_final=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
        source_sha256=data.SHA, source_files={p: sha(ROOT / p) for p in SOURCES},
    ), path)
    row = dict(architecture=recipe, seed=seed, trainable_parameters=sum(p.numel() for p in model.parameters()),
               deployed_parameters=sum(p.numel() for p in model.parameters()), actual_steps=STEPS,
               checkpoint_sha256=sha(path), parent_checkpoint_sha256=sha(parent_path),
               actual_sample_order_sha256=hashlib.sha256(plan.tobytes()).hexdigest(),
               augmented_inputs_sha256=input_hash.hexdigest(), optimizer_state_tensors=len(optimizer.state),
               actual_examples=STEPS * BATCH, actual_tokens=512 * BATCH * sum(CONTEXTS),
               parameter_tensors=len(gradient_receipt["parameters"]), nonzero_gradient_tensors_step1=sum(v["nonzero"] for v in gradient_receipt["parameters"].values()),
               gradient_receipt=gradient_receipt, postfit_profiles=profiles, profile_inputs_sha256=profile_inputs,
               training_flops=sum(profiles[str(n)]["dense_flops"] for n in CONTEXTS) * (STEPS // len(CONTEXTS)),
               flops_profiler="torch.profiler dense forward/backward estimate plus exact ATen operation/shape signature audit; unsupported scalar/optimizer FLOPs remain partial",
               optimizer_cpu_seconds=time.process_time() - start_cpu, wall_seconds=time.monotonic() - start_wall,
               peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    write(folder / "FIT_RECEIPT.json", row)
    write(folder / "GRADIENTS.json", gradient_receipt)
    return row


def fit():
    if OUT.exists():
        raise RuntimeError("ME106 output exists; original attempts cannot be overwritten or retried")
    check_manifest(json.loads(REG.read_text()), "preflight")
    parent_paths = [PARENT / "strong_reconstruction" / f"seed-{s}" / filename
                    for s in PARENT_SEEDS for filename in ("final.pt", "FIT_RECEIPT.json", "live-budget.json")]
    freeze_paths = [REG, DOC / "FIT_PLANS.npz", DOC / "PREFIT_PROFILES.json", DOC / "PREFIT_DESIGN_RECEIPT.json",
                    DOC / "PROTOCOL.md", DOC / "MATHEMATICS.md",
                    ROOT / ".github/workflows/me106-point-priority-panel.yml",
                    PARENT / "FIT_INDEX.json"] + parent_paths + [ROOT / p for p in SOURCES]
    _frozen(freeze_paths)
    OUT.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        rows = list(pool.map(fit_one, [(r, s) for s in SEEDS for r in RECIPES]))
    for seed in SEEDS:
        group = [r for r in rows if r["seed"] == seed]
        if len({r["augmented_inputs_sha256"] for r in group}) != 1:
            raise RuntimeError("same-seed input parity failed")
        for length in map(str, CONTEXTS):
            profs = [r["postfit_profiles"][length] for r in group]
            inputs = [r["profile_inputs_sha256"][length] for r in group]
            if len({p["aten_signature"] for p in profs}) != 1 or len({p["dense_flops"] for p in profs}) != 1:
                raise RuntimeError("same-seed graph parity failed")
            if len(set(inputs)) != 1:
                raise RuntimeError("same-seed profile input parity failed")
    write(OUT / "FIT_INDEX.json", dict(models=rows, source_files={p: sha(ROOT / p) for p in SOURCES}))


def budget_audit():
    index = json.loads((OUT / "FIT_INDEX.json").read_text())
    _frozen([OUT / "FIT_INDEX.json"] + [p for r in RECIPES for s in SEEDS for p in (OUT / r / f"seed-{s}").glob("*")])
    expected = {(recipe, seed) for recipe in RECIPES for seed in SEEDS}
    got = {(row.get("architecture"), row.get("seed")) for row in index["models"]}
    if got != expected or len(index["models"]) != 15:
        raise RuntimeError("FIT_INDEX must contain exactly the 15 unique recipe/seed pairs")
    if index["source_files"] != {p: sha(ROOT / p) for p in SOURCES}:
        raise RuntimeError("source hash changed after fit")
    receipt = json.loads((DOC / "PREFIT_DESIGN_RECEIPT.json").read_text())
    parent_index = json.loads((PARENT / "FIT_INDEX.json").read_text())
    if sha(PARENT / "FIT_INDEX.json") != json.loads(REG.read_text()).get("parent_fit_index_sha256"):
        raise RuntimeError("ME103 parent fit index changed since prefit")
    for row in index["models"]:
        folder = OUT / row["architecture"] / f"seed-{row['seed']}"
        cp = torch.load(folder / "final.pt", map_location="cpu", weights_only=False)
        audited_model = initialize(fit_data(), row["seed"])
        audited_model.load_state_dict(cp["model"], strict=True)
        live_count = sum(p.numel() for p in audited_model.parameters() if p.requires_grad)
        deployed_count = sum(p.numel() for p in audited_model.parameters())
        if live_count != 100349 or deployed_count != 100349 or len(list(audited_model.parameters())) != 33:
            raise RuntimeError("postfit instantiated model count differs from prefit")
        parent_seed = row["seed"] - 4000
        parent = receipt["parent_hashes"][str(row["seed"])]
        parent_folder = PARENT / "strong_reconstruction" / f"seed-{parent_seed}"
        if row["actual_steps"] != STEPS or cp.get("steps") != STEPS:
            raise RuntimeError("checkpoint and receipt must report exactly 1,536 new updates")
        if cp.get("parent_checkpoint_sha256") != parent["checkpoint_sha256"] or row.get("parent_checkpoint_sha256") != parent["checkpoint_sha256"]:
            raise RuntimeError("new model does not bind to its registered ME103 parent")
        states = cp["optimizer"]["state"]
        if len(states) != 33 or {int(v["step"]) for v in states.values()} != {STEPS}:
            raise RuntimeError("native fresh AdamW must hold all 33 states at step 1,536")
        parent_row = next(r for r in parent_index["models"]
                          if r["architecture"] == "strong_reconstruction" and r["seed"] == parent_seed)
        if sha(PARENT / "strong_reconstruction" / f"seed-{parent_seed}" / "final.pt") != parent_row["checkpoint_sha256"]:
            raise RuntimeError("parent checkpoint source hash changed")
        if cp.get("source_files") != {p: sha(ROOT / p) for p in SOURCES}:
            raise RuntimeError("checkpoint source closure hash mismatch")
        parent_files = {
            "checkpoint_sha256": sha(parent_folder / "final.pt"),
            "fit_receipt_sha256": sha(parent_folder / "FIT_RECEIPT.json"),
            "runtime_receipt_sha256": sha(parent_folder / "live-budget.json"),
        }
        if parent_files != parent:
            raise RuntimeError("ME103 parent artifact hashes differ from the prefit audit")
        validate_runtime_evidence(dict(row, _prefit_registration_sha256=sha(REG), _expected_optimizer_steps=STEPS,
            runtime_evidence=dict(receipt_path=str(folder / "live-budget.json"), receipt_sha256=sha(folder / "live-budget.json"),
                                  checkpoint_path=str(folder / "final.pt"), checkpoint_sha256=sha(folder / "final.pt"))))
    return index


def score(replay=False):
    index = budget_audit()
    # Scoring is intentionally implemented against the frozen ME101 cohort artifacts.
    fit = fit_data()
    parent_forecasts_path = ROOT / "research/replay-bundles/ME101/FORECASTS.npz"
    me101_results_path = ROOT / "research/replay-bundles/ME101/RESULTS.json"
    me103_results_path = ROOT / "research/replay-bundles/ME103/RESULTS.json"
    _frozen([parent_forecasts_path, me101_results_path, me103_results_path])
    me101_results = json.loads(me101_results_path.read_text())
    me103_results = json.loads(me103_results_path.read_text())
    report = data.load("report")
    parent_forecasts = np.load(parent_forecasts_path)
    arrays = {name + "_" + suffix: parent_forecasts[name + "_" + suffix]
              for name in ("gap", "general")
              for suffix in ("origins", "truth", "observed", "locf", "ridge")}
    cohorts = {"gap": report["gap"], "general": report["general"]}
    results = {name: dict(count=len(origins), models=[]) for name, origins in cohorts.items()}
    model_rows = []
    rawscale = torch.tensor(np.nanstd(fit["raw"][..., :6].astype(np.float64), axis=(0, 1)).clip(1e-4), dtype=torch.float32)
    with torch.no_grad():
        for row in index["models"]:
            recipe, seed = row["architecture"], row["seed"]
            cp = torch.load(OUT / recipe / f"seed-{seed}" / "final.pt", map_location="cpu", weights_only=False)
            model = initialize(fit, seed)
            model.load_state_dict(cp["model"], strict=True)
            model.eval()
            scored = {}
            for name, origins in cohorts.items():
                qs, nll_sum, nll_count = [], 0.0, 0
                for i in range(0, len(origins), 32):
                    x = [torch.from_numpy(v) for v in data.batch(report, origins[i:i + 32], 256)]
                    forecast, _ = model(*x[:6])
                    qs.append(model.raw_quantiles(forecast).numpy())
                    nll_sum += float(model.nll(forecast, x[6], x[7])) * int(x[7].sum())
                    nll_count += int(x[7].sum())
                q = np.concatenate(qs)
                arrays[f"{name}_{recipe}_{seed}"] = q
                values = me101.metrics(q, arrays[f"{name}_truth"], arrays[f"{name}_observed"])
                values["raw_density_nll"] = nll_sum / nll_count
                scored[name] = values
                results[name]["models"].append(dict(recipe=recipe, seed=seed, **values))
            model_rows.append(dict(recipe=recipe, seed=seed, scores=scored))
    gap_reference = {r["seed"]: r for r in me101_results["cohorts"]["gap"]["models"] if r.get("context") == 256}
    daily_reference = {r["seed"]: r for r in me103_results["cohorts"]["general"]["models"]
                       if r["recipe"] == "strong_reconstruction"}
    gap_parent = {r["seed"]: r for r in me103_results["cohorts"]["gap"]["models"]
                  if r["recipe"] == "strong_reconstruction"}
    daily_locf = float(np.abs(arrays["general_locf"][:, 0] - arrays["general_truth"][:, 0]).mean())
    if not np.array_equal(np.asarray(cohorts["gap"]), arrays["gap_origins"]):
        raise RuntimeError("ME106 gap report origins differ from the frozen original cohort")
    if not np.array_equal(np.asarray(cohorts["general"]), arrays["general_origins"]):
        raise RuntimeError("ME106 daily report origins differ from the frozen original cohort")
    gate_rows = []
    for row in model_rows:
        recipe, seed = row["recipe"], row["seed"]
        old_seed, parent_seed = seed - 5000, seed - 4000
        gap_score, daily_score = row["scores"]["gap"], row["scores"]["general"]
        old, parent_gap, parent_daily = gap_reference[old_seed], gap_parent[parent_seed], daily_reference[parent_seed]
        gates = dict(
            coverage=0.90 <= gap_score["coverage_by_channel"][0] <= 0.98,
            gap_point=gap_score["pm_mae"] <= min(1.05 * old["pm_mae"], 1.05 * parent_gap["pm_mae"]),
            daily_point=daily_score["pm_mae"] <= min(1.05 * daily_locf, 1.05 * parent_daily["pm_mae"]),
            gap_nll=gap_score["raw_density_nll"] <= min(6.234504261016846, 1.05 * parent_gap["raw_density_nll"]),
            interval_score=gap_score["interval_score_by_channel"][0] <= min(
                1.05 * old["interval_score_by_channel"][0], 0.90 * parent_gap["interval_score_by_channel"][0]
            ),
        )
        gate_rows.append(dict(recipe=recipe, seed=seed, gates=gates, full_gate=all(gates.values())))
    result = dict(status="EXPOSED2015_ME106_SPLIT_PANEL", models=model_rows, cohorts=results,
                  model_gates=gate_rows,
                  recipe_all_seed_gates={recipe: all(r["full_gate"] for r in gate_rows if r["recipe"] == recipe) for recipe in RECIPES},
                  independent_qualification=False, protected_test_scored=False)
    if replay:
        prior = np.load(OUT / "FORECASTS.npz")
        for key, value in arrays.items():
            np.testing.assert_allclose(value, prior[key], rtol=1e-5, atol=1e-6)
        expected = json.loads((OUT / "RESULTS.json").read_text())
        _compare_numeric(result, expected)
        if result["model_gates"] != expected["model_gates"] or result["recipe_all_seed_gates"] != expected["recipe_all_seed_gates"]:
            raise RuntimeError("ME106 score replay mismatch")
    else:
        post_path = DOC / "POSTFIT.json"
        if (OUT / "RESULTS.json").exists() or post_path.exists():
            raise RuntimeError("score artifacts already exist")
        np.savez_compressed(OUT / "FORECASTS.npz", **arrays)
        write(OUT / "RESULTS.json", result)
        postfit = json.loads(REG.read_text())
        for arm in postfit["arms"]:
            row = next(r for r in index["models"] if r["architecture"] == arm["architecture"] and r["seed"] == arm["seed"])
            arm["measured"] = dict(row, evaluation_split_sha256=arm["test_split_sha256"])
        write(post_path, postfit)
        check_manifest(postfit, "final")


def _compare_numeric(actual, expected, path="result"):
    if isinstance(actual, dict) and isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise RuntimeError(f"ME106 replay key mismatch at {path}")
        for key in actual:
            _compare_numeric(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise RuntimeError(f"ME106 replay length mismatch at {path}")
        for i, (left, right) in enumerate(zip(actual, expected)):
            _compare_numeric(left, right, f"{path}[{i}]")
    elif isinstance(actual, (int, float, np.number)) and isinstance(expected, (int, float, np.number)):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6, err_msg=path)
    elif actual != expected:
        raise RuntimeError(f"ME106 replay mismatch at {path}")


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "fit", "score", "replay"))
    mode = parser.parse_args().mode
    {"prepare": prepare, "fit": fit, "score": score, "replay": lambda: score(True)}[mode]()


if __name__ == "__main__":
    main()
