"""ME108 fixed single-candidate role-specialized research runner.

Importing this module is read-only. Optimizer work is reachable only through
the explicit ``fit`` command, after a frozen, main-authored preflight.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import resource
import subprocess
import sys
import time
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
from scripts import me106_priority_panel as me106
from scripts.diaggrad001_loss_flow import BLOCKS as ROLE_BLOCKS, block_name
from scripts.me106_gradient_roles import objective_parts
from scripts.validate_research_parity import check_manifest
from tac_osm.parameter_budget import (budgeted_optimizer, checked_optimizer_step,
                                      save_training_checkpoint, validate_runtime_evidence)
from tac_osm.role_optimizer import BlockProtectedAdamW
from tac_osm.variants.role_specialized_sensor_world import RoleSpecializedSensorWorld

OUT = ROOT / "research/replay-bundles/ME108"
DOC = ROOT / "research/mathematical-engineering/me108-role-trained-2026-10-26"
REG = DOC / "FAIR_COMPARISON.json"
PARENT = ROOT / "research/replay-bundles/ME103"
PARENT_DOC = ROOT / "research/mathematical-engineering/me103"
SEEDS = tuple(range(108001, 108006))
PARENT_SEEDS = tuple(range(102001, 102006))
STEPS, BATCH, CONTEXTS = 4096, 16, (8, 64, 256)
ARMS = ("role_guarded",)
SOURCE_FILES = list(dict.fromkeys(me106.SOURCES + [
    "scripts/me101_belief_sensor.py", "scripts/me106_gradient_roles.py",
    "scripts/me108_role_training.py", "src/tac_osm/parameter_budget.py",
    "scripts/me108_optimizer_reference.py",
    "src/tac_osm/role_optimizer.py", "src/tac_osm/role_update_contract.py",
    "scripts/diaggrad001_loss_flow.py", "scripts/me107_graph_reference.py",
    "src/tac_osm/variants/median_split_sensor_world.py",
    "src/tac_osm/variants/protected_sensor_world.py",
    "src/tac_osm/variants/point_priority_sensor_world.py",
    "src/tac_osm/variants/role_specialized_sensor_world.py",
]))


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
    return data.load("fit")


def tensor_batch(d, origins, length, rng):
    return [torch.from_numpy(x) for x in gap.augmentation(d, origins, length, rng)]


def make_plan(seed, d):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(STEPS):
        gap_rows = rng.integers(len(d["gap"]), size=BATCH // 2)
        general_rows = rng.integers(len(d["general"]), size=BATCH // 2)
        rows.append([d["gap"][i] for i in gap_rows] + [d["general"][i] for i in general_rows])
    return np.asarray(rows, dtype=np.int64)


def initialize(d, seed):
    torch.manual_seed(seed)
    model = RoleSpecializedSensorWorld(torch.tensor(gap.floor_rates(d)["rates"], dtype=torch.float32))
    with torch.no_grad():
        model.target_mean.copy_(torch.from_numpy(d["mean"][:6]))
        model.target_scale.copy_(torch.from_numpy(d["std"][:6]))
    return model


def load_parent(model, checkpoint):
    """Copy the ME103 backbone, replacing its current linear head with fresh FFNs."""
    state = checkpoint["model"]
    dropped = {key for key in state if key.startswith(("current_head.", "width_head.", "median_head."))}
    expected_parent_heads = {"current_head.weight", "current_head.bias"}
    if dropped != expected_parent_heads:
        raise RuntimeError(f"ME103 strong_reconstruction must have exactly its original current linear head; got {sorted(dropped)}")
    transferred = {key: value for key, value in state.items() if key not in dropped}
    result = model.load_state_dict(transferred, strict=False)
    expected = sorted(key for key in model.state_dict()
                      if key.startswith(("current_head.", "width_head.", "median_head.")))
    if sorted(result.missing_keys) != expected or result.unexpected_keys:
        raise RuntimeError(f"parent backbone transfer mismatch: {result}")
    return sorted(dropped)


def parent_artifacts(seed):
    parent_seed = seed - 6000
    path = PARENT / "strong_reconstruction" / f"seed-{parent_seed}" / "final.pt"
    row = next(r for r in json.loads((PARENT / "FIT_INDEX.json").read_text())["models"]
               if r["architecture"] == "strong_reconstruction" and r["seed"] == parent_seed)
    if sha(path) != row["checkpoint_sha256"]:
        raise RuntimeError("ME103 parent checkpoint hash mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (checkpoint.get("steps") != 3072 or checkpoint.get("seed") != parent_seed
            or checkpoint.get("recipe") != "strong_reconstruction"):
        raise RuntimeError("unexpected ME103 checkpoint lineage")
    folder = path.parent
    return path, checkpoint, {"checkpoint_sha256": sha(path),
                              "fit_receipt_sha256": sha(folder / "FIT_RECEIPT.json"),
                              "runtime_receipt_sha256": sha(folder / "live-budget.json")}


def role_objective(model, forecast, trace, batch, rawscale):
    return objective_parts(model, forecast, trace, batch, rawscale,
                           {"point": .5, "interval": .25})


def blocks(model):
    named = list(model.named_parameters())
    indices = {name: [i for i, (parameter_name, _) in enumerate(named)
                      if block_name(parameter_name) == name] for name in ROLE_BLOCKS}
    if any(not indices[name] for name in ROLE_BLOCKS) or sum(map(len, indices.values())) != len(named):
        raise RuntimeError("13 mathematical role blocks must partition all model parameters")
    return named, indices


def parameter_groups(model):
    named = list(model.named_parameters())
    new_prefixes = ("current_head.", "width_head.", "median_head.")
    inherited = [p for n, p in named if not n.startswith(new_prefixes)]
    new_heads = [p for n, p in named if n.startswith(new_prefixes)]
    if len(inherited) + len(new_heads) != len(named) or len({id(p) for p in inherited + new_heads}) != len(named):
        raise RuntimeError("optimizer groups must cover every parameter exactly once")
    if [id(p) for p in inherited + new_heads] != [id(p) for _, p in named]:
        raise RuntimeError("optimizer flattened groups must preserve named-parameter order")
    return [dict(params=inherited, lr=3e-5), dict(params=new_heads, lr=3e-4)]


def gradient_norms(grads, indices):
    result = {}
    for name, ix in indices.items():
        present = [grads[i] for i in ix if grads[i] is not None]
        squared_norm = sum(float(g.detach().double().square().sum()) for g in present)
        result[name] = dict(tensors=len(present),
                            nonzero=sum(bool(torch.count_nonzero(g)) for g in present),
                            norm=squared_norm ** .5)
    return result


def profile(model, x, rawscale):
    model.zero_grad(set_to_none=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                with_flops=True, record_shapes=True) as prof:
        forecast, _, trace = model(*x[:6], return_trace=True)
        loss, parts = role_objective(model, forecast, trace, x, rawscale)
        params = tuple(model.parameters())
        # Fit uses this same two-VJP path: point role first, combined loss second.
        torch.autograd.grad(.5 * parts["point"], params, retain_graph=True, allow_unused=True)
        loss.backward()
    events = prof.key_averages(group_by_input_shape=True)
    ops = sorted((e.key, e.count, str(e.input_shapes)) for e in events if e.key.startswith("aten::"))
    return dict(dense_flops=sum(e.flops for e in events), flops_scope="partial_forward_backward; optimizer and unsupported operators excluded",
                aten_signature=digest(ops),
                aten_calls=sum(e.count for e in events if e.key.startswith("aten::")))


def prepare():
    if OUT.exists() or any((DOC / name).exists() for name in ("FIT_PLANS.npz", "PREFIT_PROFILES.json", "PREFIT_DESIGN_RECEIPT.json")):
        raise RuntimeError("ME108 artifacts exist; no overwrite or retry")
    if not REG.exists():
        raise RuntimeError("main-owned ME108 preregistration is required before prepare")
    registration = json.loads(REG.read_text())
    check_manifest(registration, "preflight")
    if registration.get("study_type") != "single_model_training" or len(registration.get("models", [])) != 5:
        raise RuntimeError("ME108 preregistration must contain exactly five single_model_training seed arms")
    if [(m.get("architecture"), m.get("seed")) for m in registration["models"]] != [
            ("role_guarded", seed) for seed in SEEDS]:
        raise RuntimeError("ME108 preregistered models must bind exactly the five fixed role_guarded seeds")
    d = fit_data()
    plans = {str(seed): make_plan(seed, d) for seed in SEEDS}
    np.savez_compressed(DOC / "FIT_PLANS.npz", **plans)
    plan_hashes = {str(s): hashlib.sha256(plans[str(s)].tobytes()).hexdigest() for s in SEEDS}
    rawscale = torch.tensor(np.nanstd(d["raw"][..., :6].astype(np.float64), axis=(0, 1)).clip(1e-4), dtype=torch.float32)
    parent_hashes, audits, profiles = {}, [], {}
    for seed in SEEDS:
        ppath, cp, hashes = parent_artifacts(seed)
        parent_hashes[str(seed)] = hashes
        model = initialize(d, seed)
        dropped = load_parent(model, cp)
        nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if nparams != 102713 or len(list(model.parameters())) != 39 or not all(p.requires_grad for p in model.parameters()):
            raise RuntimeError("role model failed 102,713 parameter / 39 tensor guard")
        x = tensor_batch(d, plans[str(seed)][0], CONTEXTS[0], np.random.default_rng(seed + 10000))
        model.zero_grad(set_to_none=True)
        forecast, _, trace = model(*x[:6], return_trace=True)
        loss, parts = role_objective(model, forecast, trace, x, rawscale)
        point = torch.autograd.grad(.5 * parts["point"], tuple(model.parameters()), retain_graph=True, allow_unused=True)
        loss.backward()
        grads = [p.grad for p in model.parameters()]
        if not torch.isfinite(loss) or any(g is None or not torch.isfinite(g).all() for g in grads):
            raise RuntimeError("prefit all-parameter finite gradient audit failed")
        named, block_indices = blocks(model)
        audits.append(dict(seed=seed, parent_seed=seed-6000, parent_checkpoint_sha256=sha(ppath),
                           dropped_parent_tensors=dropped, trainable_parameters=nparams,
                           deployed_parameters=sum(p.numel() for p in model.parameters()), parameter_tensors=len(grads),
                           loss=float(loss.detach()), point_gradient=gradient_norms(point, block_indices),
                           total_gradient=gradient_norms(grads, block_indices), no_optimizer_step=True))
        for length in CONTEXTS:
            sample = tensor_batch(d, plans[str(seed)][0], length,
                                  np.random.default_rng(seed + 10000))
            if str(length) not in profiles:
                profiles[str(length)] = profile(model, sample, rawscale)
    write(DOC / "PREFIT_DESIGN_RECEIPT.json", dict(parent_hashes=parent_hashes, plan_sha256=plan_hashes,
         audits=audits, trainable_parameters=102713, deployed_parameters=102713, parameter_tensors=39,
         profiles=profiles, no_optimizer_updates=True))
    write(DOC / "PREFIT_PROFILES.json", dict(profiles=profiles, no_optimizer_updates=True))


def frozen(paths):
    for path in paths:
        rel = Path(path).relative_to(ROOT)
        raw = subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=ROOT, stderr=subprocess.DEVNULL)
        if raw != Path(path).read_bytes():
            raise RuntimeError(f"file is not Git-frozen: {rel}")


def fit_one(seed):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    registration = json.loads(REG.read_text())
    check_manifest(registration, "preflight")
    if seed not in SEEDS:
        raise RuntimeError("unregistered ME108 seed")
    for key, path in (("fit_plans_sha256", DOC / "FIT_PLANS.npz"),
                      ("prefit_design_sha256", DOC / "PREFIT_DESIGN_RECEIPT.json"),
                      ("prefit_profiles_sha256", DOC / "PREFIT_PROFILES.json")):
        if registration.get(key) != sha(path):
            raise RuntimeError(f"registered ME108 {key} changed")
    if registration.get("source_files") != {p: sha(ROOT / p) for p in SOURCE_FILES}:
        raise RuntimeError("ME108 source closure differs from frozen registration")
    d = fit_data()
    ppath, parent, parent_hashes = parent_artifacts(seed)
    expected_parent = json.loads((DOC / "PREFIT_DESIGN_RECEIPT.json").read_text())["parent_hashes"][str(seed)]
    if expected_parent != parent_hashes:
        raise RuntimeError("ME103 parent artifacts changed since prefit")
    model = initialize(d, seed)
    dropped = load_parent(model, parent)
    folder = OUT / "role_guarded" / f"seed-{seed}"
    folder.mkdir(parents=True, exist_ok=False)
    named, block_indices = blocks(model)
    groups = parameter_groups(model)
    # Block indices are in model.named_parameters order; BlockProtectedAdamW
    # maps these using its documented flattened group ordering.
    optimizer = budgeted_optimizer(model, BlockProtectedAdamW, parameters=groups,
        blocks=list(block_indices.values()), lr=3e-5, weight_decay=.001,
        betas=(.9, .999), eps=1e-8, budget_manifest=REG,
        budget_receipt=folder / "live-budget.json", budget_seed=seed,
        budget_architecture="role_guarded")
    if optimizer.state:
        raise RuntimeError("fresh ME108 AdamW must start without inherited optimizer state")
    plan = np.load(DOC / "FIT_PLANS.npz")[str(seed)]
    if plan.shape != (STEPS, BATCH): raise RuntimeError("registered ME108 plan must contain 4096 batches of 16 origins")
    rng = np.random.default_rng(seed + 10000)
    rawscale = torch.tensor(np.nanstd(d["raw"][..., :6].astype(np.float64), axis=(0, 1)).clip(1e-4), dtype=torch.float32)
    input_hash, losses, first_grad = hashlib.sha256(), [], None
    start_cpu, start_wall = time.process_time(), time.monotonic()
    guard_counts = {}
    for step, origins in enumerate(plan):
        x = tensor_batch(d, origins, CONTEXTS[step % len(CONTEXTS)], rng)
        for v in x[:4]: input_hash.update(v.numpy().tobytes())
        optimizer.zero_grad(set_to_none=True)
        forecast, _, trace = model(*x[:6], return_trace=True)
        loss, parts = role_objective(model, forecast, trace, x, rawscale)
        if not torch.isfinite(loss): raise RuntimeError("nonfinite ME108 loss; no retry")
        params = tuple(p for _, p in named)
        point_loss = .5 * parts["point"]
        point_grads = torch.autograd.grad(point_loss, params, retain_graph=True, allow_unused=True)
        loss.backward()
        for p in params:
            if p.grad is None: p.grad = torch.zeros_like(p)
        if any(not torch.isfinite(p.grad).all() for p in params): raise RuntimeError("nonfinite full objective gradient")
        if step == 0:
            if len(params) != 39:
                raise RuntimeError("first-step gradient receipt requires all 39 parameter tensors")
            first_grad = dict(parameters={n: dict(norm=float(p.grad.norm()), nonzero=bool(torch.count_nonzero(p.grad))) for n,p in named},
                              point=gradient_norms(point_grads, block_indices))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.set_point_gradients([torch.zeros_like(p) if g is None else g.detach() for p, g in zip(params, point_grads)])
        guard_start = time.monotonic()
        checked_optimizer_step(optimizer)
        guard_elapsed = time.monotonic() - guard_start
        receipt = getattr(optimizer, "last_guard_receipt", None)
        if not receipt: raise RuntimeError("optimizer omitted compulsory role guard receipt")
        if len(receipt) != len(ROLE_BLOCKS): raise RuntimeError("optimizer guard must receipt all 13 role blocks")
        for block_name_value, block_receipt in zip(ROLE_BLOCKS, receipt):
            for key in ("projected", "rounding_fallback", "applied_rounding_fallback"):
                counter = f"{block_name_value}.{key}"
                guard_counts[counter] = guard_counts.get(counter, 0) + int(bool(block_receipt[key]))
            guard_counts[f"{block_name_value}.max_applied_dot"] = max(
                guard_counts.get(f"{block_name_value}.max_applied_dot", float("-inf")),
                float(block_receipt["applied_dot"]))
            if float(block_receipt["applied_dot"]) > 0:
                raise RuntimeError(f"ME108 applied point guard failed for {block_name_value}")
        guard_counts["guard_seconds"] = guard_counts.get("guard_seconds", 0.0) + guard_elapsed
        losses.append({k: float(v.detach()) for k, v in parts.items()})
        if (step+1)%512==0:
            print(json.dumps(dict(seed=seed,step=step+1,fit_point=losses[-1]['point'],
                                  fit_nll=losses[-1]['nll'],cpu_seconds=time.process_time()-start_cpu)),flush=True)
    if len(optimizer.state) != 39: raise RuntimeError("optimizer did not retain state for all 39 tensors")
    profile_receipt = json.loads((DOC / "PREFIT_PROFILES.json").read_text())["profiles"]
    counts = {8: sum(step % 3 == 0 for step in range(STEPS)),
              64: sum(step % 3 == 1 for step in range(STEPS)),
              256: sum(step % 3 == 2 for step in range(STEPS))}
    profiled_flops = sum(int(profile_receipt[str(context)]["dense_flops"]) * count
                         for context, count in counts.items())
    tokens_seen = BATCH * sum(context * count for context, count in counts.items())
    path = folder / "final.pt"
    save_training_checkpoint(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
        architecture="role_guarded", seed=seed, steps=STEPS, plan=plan,
        parent_checkpoint_sha256=sha(ppath), dropped_parent_tensors=dropped, losses=losses,
        augmentation_rng_initial_seed=seed + 10000, python_rng=random.getstate(),
        augmentation_rng_final=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
        source_sha256=data.SHA, source_files={p: sha(ROOT / p) for p in SOURCE_FILES}), path)
    row = dict(architecture="role_guarded", seed=seed, trainable_parameters=sum(p.numel() for p in model.parameters()),
        deployed_parameters=sum(p.numel() for p in model.parameters()), actual_steps=STEPS,
        checkpoint_sha256=sha(path), parent_checkpoint_sha256=sha(ppath), dropped_parent_tensors=dropped,
        actual_sample_order_sha256=hashlib.sha256(plan.tobytes()).hexdigest(), augmented_inputs_sha256=input_hash.hexdigest(),
        optimizer_state_tensors=len(optimizer.state), parameter_tensors=39, first_gradient_receipt=first_grad,
        guard_aggregate=guard_counts, optimizer_cpu_seconds=time.process_time()-start_cpu,
        wall_seconds=time.monotonic()-start_wall, peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        actual_examples=STEPS*BATCH, actual_tokens=tokens_seen, training_flops=profiled_flops,
        flops_scope="partial model forward/backward profiler estimate; excludes optimizer and unsupported operators")
    write(folder / "FIT_RECEIPT.json", row)
    return row


def fit():
    if OUT.exists(): raise RuntimeError("ME108 fit output exists; no overwrite or retry")
    registration = json.loads(REG.read_text())
    check_manifest(registration, "preflight")
    if registration.get("study_type") != "single_model_training":
        raise RuntimeError("ME108 preflight must authorize single_model_training")
    parents = [PARENT / "strong_reconstruction" / f"seed-{s}" / name
               for s in PARENT_SEEDS for name in ("final.pt", "FIT_RECEIPT.json", "live-budget.json")]
    frozen([REG, DOC / "FIT_PLANS.npz", DOC / "PREFIT_DESIGN_RECEIPT.json", DOC / "PREFIT_PROFILES.json",
            PARENT / "FIT_INDEX.json", PARENT_DOC / "FAIR_COMPARISON.json", *parents,
            *[ROOT / p for p in SOURCE_FILES]])
    OUT.mkdir(parents=True, exist_ok=False)
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        rows = list(pool.map(fit_one, SEEDS))
    write(OUT / "FIT_INDEX.json", dict(models=rows, source_files={p: sha(ROOT / p) for p in SOURCE_FILES}))


def budget_audit():
    index = json.loads((OUT / "FIT_INDEX.json").read_text())
    expected = {("role_guarded", s) for s in SEEDS}
    if {(r.get("architecture"), r.get("seed")) for r in index["models"]} != expected:
        raise RuntimeError("ME108 FIT_INDEX must contain exactly five registered models")
    if index["source_files"] != {p: sha(ROOT / p) for p in SOURCE_FILES}:
        raise RuntimeError("ME108 source closure changed after fit")
    frozen([REG, OUT / "FIT_INDEX.json", DOC / "FIT_PLANS.npz",
            DOC / "PREFIT_DESIGN_RECEIPT.json", DOC / "PREFIT_PROFILES.json",
            PARENT / "FIT_INDEX.json", PARENT_DOC / "FAIR_COMPARISON.json",
            *[ROOT / p for p in SOURCE_FILES],
            *[PARENT / "strong_reconstruction" / f"seed-{s}" / filename
              for s in PARENT_SEEDS for filename in ("final.pt", "FIT_RECEIPT.json", "live-budget.json")],
            *[p for r in index["models"] for p in (OUT / "role_guarded" / f"seed-{r['seed']}").glob("*")]])
    for row in index["models"]:
        folder = OUT / "role_guarded" / f"seed-{row['seed']}"
        validate_runtime_evidence(dict(row, _prefit_registration_sha256=sha(REG), _expected_optimizer_steps=STEPS,
            runtime_evidence=dict(receipt_path=str(folder / "live-budget.json"), receipt_sha256=sha(folder / "live-budget.json"),
                                  checkpoint_path=str(folder / "final.pt"), checkpoint_sha256=sha(folder / "final.pt"))))
    return index


def score(replay=False):
    index = budget_audit()
    d, fit = data.load("report"), fit_data()
    forecasts_path = ROOT / "research/replay-bundles/ME101/FORECASTS.npz"
    old_results_path = ROOT / "research/replay-bundles/ME101/RESULTS.json"
    parent_results_path = ROOT / "research/replay-bundles/ME103/RESULTS.json"
    arrays0 = np.load(forecasts_path)
    frozen([forecasts_path, old_results_path, parent_results_path])
    arrays = {name + "_" + suffix: arrays0[name + "_" + suffix]
              for name in ("gap", "general") for suffix in ("origins", "truth", "observed", "locf", "ridge")}
    cohorts = {"gap": d["gap"], "general": d["general"]}
    for name, origins in cohorts.items():
        if not np.array_equal(np.asarray(origins), arrays[name + "_origins"]):
            raise RuntimeError(f"ME108 {name} REPORT origins differ from the frozen original evaluation cohort")
    results = {name: dict(count=len(origins), models=[]) for name, origins in cohorts.items()}
    model_rows = []
    rawscale = torch.tensor(np.nanstd(fit["raw"][..., :6].astype(np.float64), axis=(0, 1)).clip(1e-4), dtype=torch.float32)
    with torch.no_grad():
        for row in index["models"]:
            seed = row["seed"]
            cp = torch.load(OUT / "role_guarded" / f"seed-{seed}" / "final.pt", map_location="cpu", weights_only=False)
            model = initialize(fit, seed); model.load_state_dict(cp["model"], strict=True); model.eval()
            scored = {}
            for name, origins in cohorts.items():
                qs, nll_sum, nll_count = [], 0., 0
                for i in range(0, len(origins), 32):
                    x = [torch.from_numpy(v) for v in data.batch(d, origins[i:i+32], 256)]
                    forecast, _ = model(*x[:6]); qs.append(model.raw_quantiles(forecast).numpy())
                    nll_sum += float(model.nll(forecast, x[6], x[7])) * int(x[7].sum()); nll_count += int(x[7].sum())
                q = np.concatenate(qs); arrays[f"{name}_role_guarded_{seed}"] = q
                metrics = me101.metrics(q, arrays[f"{name}_truth"], arrays[f"{name}_observed"])
                metrics["raw_density_nll"] = nll_sum / nll_count; scored[name] = metrics
                results[name]["models"].append(dict(architecture="role_guarded", seed=seed, **metrics))
            model_rows.append(dict(architecture="role_guarded", seed=seed, scores=scored))
    old = json.loads(old_results_path.read_text()); parent = json.loads(parent_results_path.read_text())
    old_gap = {r["seed"]: r for r in old["cohorts"]["gap"]["models"] if r.get("context") == 256}
    parent_gap = {r["seed"]: r for r in parent["cohorts"]["gap"]["models"] if r["recipe"] == "strong_reconstruction"}
    parent_daily = {r["seed"]: r for r in parent["cohorts"]["general"]["models"] if r["recipe"] == "strong_reconstruction"}
    daily_locf = float(np.abs(arrays["general_locf"][:, 0] - arrays["general_truth"][:, 0]).mean())
    gates = []
    for row in model_rows:
        seed = row["seed"]; oldrow, pg, pd = old_gap[seed-7000], parent_gap[seed-6000], parent_daily[seed-6000]
        g, n = row["scores"]["gap"], row["scores"]["general"]
        checks = dict(coverage=.90 <= g["coverage_by_channel"][0] <= .98,
            gap_point=g["pm_mae"] <= min(1.05*oldrow["pm_mae"],1.05*pg["pm_mae"]),
            daily_point=n["pm_mae"] <= min(1.05*daily_locf,1.05*pd["pm_mae"]),
            gap_nll=g["raw_density_nll"] <= min(6.234504261016846,1.05*pg["raw_density_nll"]),
            interval_score=g["interval_score_by_channel"][0] <= min(1.05*oldrow["interval_score_by_channel"][0],.90*pg["interval_score_by_channel"][0]))
        gates.append(dict(architecture="role_guarded", seed=seed, gates=checks, full_gate=all(checks.values())))
    result = dict(status="EXPOSED2015_ME108_ROLE_TRAINING", models=model_rows, cohorts=results, model_gates=gates,
        all_seed_gate=all(r["full_gate"] for r in gates), independent_qualification=False, protected_test_scored=False)
    if replay:
        prior = np.load(OUT / "FORECASTS.npz")
        for k, v in arrays.items(): np.testing.assert_allclose(v, prior[k], rtol=1e-5, atol=1e-6)
        expected = json.loads((OUT / "RESULTS.json").read_text())
        if result["model_gates"] != expected["model_gates"] or result["all_seed_gate"] != expected["all_seed_gate"]:
            raise RuntimeError("ME108 replay gate booleans differ from original score")
        _compare_numeric(result, expected)
    else:
        if (OUT / "RESULTS.json").exists() or (DOC / "POSTFIT.json").exists(): raise RuntimeError("ME108 scores already exist")
        np.savez_compressed(OUT / "FORECASTS.npz", **arrays); write(OUT / "RESULTS.json", result)
        post = json.loads(REG.read_text())
        for model_record in post["models"]:
            row = next(r for r in index["models"]
                       if r["architecture"] == model_record["architecture"] and r["seed"] == model_record["seed"])
            model_record["measured"] = row
        write(DOC / "POSTFIT.json", post); check_manifest(post, "final")


def _compare_numeric(actual, expected, path="result"):
    if isinstance(actual, dict) and isinstance(expected, dict):
        if actual.keys() != expected.keys(): raise RuntimeError(f"replay key mismatch at {path}")
        for key in actual: _compare_numeric(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected): raise RuntimeError(f"replay length mismatch at {path}")
        for i, (a, b) in enumerate(zip(actual, expected)): _compare_numeric(a, b, f"{path}[{i}]")
    elif isinstance(actual, (int, float, np.number)) and isinstance(expected, (int, float, np.number)):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6, err_msg=path)
    elif actual != expected: raise RuntimeError(f"replay mismatch at {path}")


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("prepare", "fit", "score", "replay"))
    mode = parser.parse_args().mode
    {"prepare": prepare, "fit": fit, "score": score, "replay": lambda: score(True)}[mode]()


if __name__ == "__main__": main()
