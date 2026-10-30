"""Strict same-graph objective comparisons without invented exact FLOP totals.

This gate is separate from architecture parity. It requires identical actual
ATen signatures/shapes/call counts for every trained forward/backward step and
paired first/warm optimizer ledgers. Identical opaque operators cancel only
within the same source-pinned model. It does not certify hardware work, energy,
generalization or a cross-architecture compute match.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import re

HEX64 = re.compile(r"^[0-9a-f]{64}$")
SCHEMA = "tac-osm.objective-operator-profile.v1"
CONTRACTS = ("dataset_sha256", "train_split_sha256", "dev_split_sha256",
             "training_sample_order_sha256", "feature_contract_sha256",
             "target_contract_sha256", "optimizer_schedule_sha256",
             "tuning_policy_sha256", "selection_policy_sha256",
             "evaluation_contract_sha256")
BUDGETS = ("steps", "examples_seen", "tokens_seen", "batch_size",
           "history_events", "dense_compute_tokens", "tuning_trials", "tuning_flops")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _require(value, reason):
    if not value:
        raise ValueError(reason)


def _sha(value, name):
    _require(isinstance(value, str) and HEX64.fullmatch(value), name + " requires SHA256")


def _integer(value, name, *, zero=False):
    _require(type(value) is int and value >= (0 if zero else 1), name + " requires a positive integer")


def _parameter_count(row):
    for field in ("trainable_parameters", "deployed_parameters"):
        _integer(row.get(field), field)
        _require(95000 <= row[field] <= 105000, "prospective 100K model band required")
    _require(row["deployed_parameters"] >= row["trainable_parameters"], "deployed parameters below active count")


def ledger_counter(row):
    entries = row["operators"] if isinstance(row, dict) else row
    _require(isinstance(entries, list) and entries, "complete nonempty operator ledger required")
    result = Counter()
    for entry in entries:
        _require(isinstance(entry, dict), "operator row must be an object")
        stage, signature, calls = entry.get("stage"), entry.get("signature"), entry.get("calls")
        _require(isinstance(stage, str) and bool(stage), "operator stage missing")
        _integer(calls, "operator calls")
        _require(isinstance(signature, str), "operator signature missing")
        decoded = json.loads(signature)
        _require(isinstance(decoded, list) and len(decoded) == 3 and
                 isinstance(decoded[0], str) and decoded[0].startswith("aten::"),
                 "actual ATen signature required")
        for tensors in decoded[1:]:
            _require(isinstance(tensors, list), "input/output tensor signature missing")
            for dtype, shape in tensors:
                _require(isinstance(dtype, str) and isinstance(shape, list) and
                         all(type(d) is int and d >= 0 for d in shape), "invalid tensor shape signature")
        key = stage, signature
        _require(key not in result, "duplicate operator ledger row")
        result[key] = calls
    return result


def verify_same_ledger(left, right, label, *, require_matrix=True):
    a, b = ledger_counter(left), ledger_counter(right)
    _require(a == b, label + ": complete operator signatures/shapes/counts differ")
    if require_matrix:
        _require(any(json.loads(sig)[0].split(".")[0] in ("aten::mm", "aten::bmm", "aten::addmm")
                     for _, sig in a), label + ": actual matrix computation missing")


def source_bindings(root, model_paths):
    """Fingerprint executable sources, excluding profiles/config values."""
    root = Path(root).resolve()
    names = set(model_paths)
    names.update(("research/mathematical-engineering/langacquire097/" + name)
                 for name in ("models.py", "objective.py", "train.py", "resources.py", "profile097.py", "data.py"))
    names.update(("src/tac_osm/research_objective_budget.py", "scripts/validate_research_parity.py"))
    result = {}
    for name in sorted(names):
        path = root / name
        _require(not path.is_symlink() and path.resolve().is_relative_to(root), "source path escapes repository")
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def state_dict_digest(state):
    """Same tensor-byte convention as the source runner, without model forward."""
    import torch
    _require(isinstance(state, dict) and bool(state), "original initial model tensors required")
    result = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        _require(isinstance(name, str) and isinstance(tensor, torch.Tensor) and
                 tensor.device.type == "cpu" and tensor.layout == torch.strided and
                 bool(torch.isfinite(tensor).all()), "initial model state must contain finite CPU tensors")
        result.update(name.encode())
        result.update(str(tuple(tensor.shape)).encode())
        result.update(tensor.detach().contiguous().numpy().tobytes())
    return result.hexdigest()


def verify_initial_state(profile_path, row, family):
    """Authenticate the zero-update weights, not a seed reproducibility claim."""
    import torch
    name = row.get("initial_state_file")
    _require(isinstance(name, str) and name == f"INIT_{family}_{row['seed']}.pt" and
             re.fullmatch(r"INIT_[a-z0-9_]+_[0-9]+\.pt", name), "source initial-state filename differs")
    _sha(row.get("initial_state_sha256"), "original initial-state archive")
    parent = Path(profile_path).resolve().parent
    path = parent / name
    _require(not path.is_symlink() and path.resolve().is_relative_to(parent), "initial-state archive escapes profile directory")
    _require(path.stat().st_size <= 32 * 1024 * 1024, "initial-state archive exceeds bounded model storage")
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == row["initial_state_sha256"], "source initial-state archive changed")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    _require(isinstance(payload, dict) and set(payload) ==
             {"schema", "family", "seed", "model", "model_sha256", "optimizer_steps"} and
             payload["schema"] == "langacquire097.initial-state.v1" and payload["family"] == family and
             payload["seed"] == row["seed"] and payload["optimizer_steps"] == 0,
             "original zero-optimizer initial-state payload required")
    actual = state_dict_digest(payload["model"])
    _require(actual == payload["model_sha256"] == row["weights_before"],
             "actual original initial tensors differ from the profile")
    return actual


def _profile(manifest, root):
    root = Path(root).resolve()
    relative = manifest.get("operator_profile_path")
    _require(isinstance(relative, str) and not Path(relative).is_absolute(), "repository-owned operator profile required")
    path = root / relative
    _require(not path.is_symlink() and path.resolve().is_relative_to(root), "operator profile escapes repository")
    _sha(manifest.get("operator_profile_sha256"), "operator_profile_sha256")
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == manifest["operator_profile_sha256"], "frozen operator profile changed")
    profile = json.loads(raw)
    _require(profile.get("schema") == SCHEMA, "original objective operator profile required")
    _require(profile.get("family") == manifest["architecture_family"], "profile model family differs")
    bindings = {"source_sha256": manifest["real_world_source"]["sha256"],
                "recipe_sha256": manifest["training_recipe_sha256"],
                "implementation_sha256": manifest["implementation_sha256"]}
    for name, expected in bindings.items():
        _sha(expected, name)
        _require(profile.get(name) == expected, "operator profile binding differs: " + name)
    _require(isinstance(profile.get("training_recipe"), dict) and
             digest(profile["training_recipe"]) == profile["recipe_sha256"], "actual profiled training recipe differs")
    sources = profile.get("source_hashes")
    _require(isinstance(sources, dict) and bool(sources), "actual executable source hashes required")
    _require(digest(sources) == profile["implementation_sha256"], "implementation digest does not bind sources")
    for relative, expected in sources.items():
        _sha(expected, "source executable")
        path = root / relative
        _require(not Path(relative).is_absolute() and not path.is_symlink() and path.resolve().is_relative_to(root),
                 "source executable escapes repository")
        _require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, "profile executable source changed: " + relative)
    return profile


def validate_comparison(manifest, phase="preflight", *, root=None):
    _require(phase in ("preflight", "final"), "unknown objective-budget phase")
    _require(manifest.get("execution_level") == "internal_research",
             "objective operator accounting is internal research, not independent qualification")
    _require(manifest.get("registered_before_first_fit") is True, "objective comparison must be preregistered")
    _require(manifest.get("no_architecture_superiority_claim") is True, "same-family objective comparison cannot rank architectures")
    _require(manifest.get("compute_accounting") == "within_family_complete_operator_ledger", "complete objective operator convention required")
    family = manifest.get("architecture_family")
    _require(isinstance(family, str) and len(family) >= 2, "one actual architecture family required")
    source = manifest.get("real_world_source", {})
    _require(source.get("observed_real_outcomes") is True, "authentic primary observations required")
    _sha(source.get("sha256"), "original source")
    for field in ("name", "license"):
        _require(isinstance(source.get(field), str) and len(source[field]) >= 8, "real source " + field + " missing")
    literature = manifest.get("literature_gap")
    _require(isinstance(literature, dict), "prior literature and an unresolved objective question required")
    for key in ("unresolved_TAConly_question", "what_is_not_being_rediscovered"):
        _require(isinstance(literature.get(key), str) and len(literature[key].strip()) >= 8, "literature gap missing: " + key)
    prior = literature.get("established_work")
    _require(isinstance(prior, list) and len(prior) >= 2, "at least two specific prior works required")
    for row in prior:
        _require(isinstance(row, dict) and all(isinstance(row.get(k), str) and len(row[k].strip()) >= 8
                 for k in ("paper", "established_result")) and str(row.get("url", "")).startswith("https://"),
                 "published prior-work reference required")
    rows = manifest.get("models")
    _require(isinstance(rows, list) and len(rows) >= 4, "full/query paired model matrix required")
    seeds = manifest.get("seeds")
    _require(isinstance(seeds, list) and len(seeds) >= 2 and len(set(seeds)) == len(seeds), "at least two explicit independent seeds required")
    for seed in seeds:
        _integer(seed, "seed")
    _require(len(rows) == 2 * len(seeds), "every seed needs exactly two objectives")
    profile = _profile(manifest, root or Path.cwd())
    profiles = profile.get("profiles")
    _require(isinstance(profiles, list) and len(profiles) == len(rows), "profile seed/objective matrix incomplete")
    expected = {(seed, objective) for seed in seeds for objective in ("full", "query")}
    _require({(p.get("seed"), p.get("objective")) for p in profiles} == expected, "profile seed/objective identities differ")
    _require({(r.get("seed"), r.get("name")) for r in rows} == expected, "model seed/objective identities differ")
    optimizer = manifest.get("optimizer_contract")
    _require(isinstance(optimizer, dict) and optimizer.get("name") == "AdamW" and optimizer.get("foreach") is False,
             "explicit identical dense AdamW optimizer contract required")
    _require(profile.get("optimizer_contract") == optimizer, "profile optimizer contract differs")
    recipe = profile["training_recipe"]
    expected_optimizer = {"name": "AdamW", "learning_rate": recipe.get("learning_rate"),
                          "weight_decay": recipe.get("weight_decay"), "grad_clip": recipe.get("grad_clip"),
                          "betas": recipe.get("betas", [0.9, 0.999]), "eps": recipe.get("eps", 1e-8),
                          "amsgrad": False, "foreach": False, "clip_foreach": False, "fused": False}
    _require(optimizer == expected_optimizer, "optimizer differs from the frozen profiled training recipe")
    _require(profile.get("optimizer_steps") == 0, "prefit profiling must not execute optimizer updates")
    _require(profile.get("hardware", {}).get("torch_threads") == 1 and
             profile.get("hardware", {}).get("cuda_used") is False,
             "registered single-thread CPU profile required")
    by_seed = defaultdict(list)
    verified_initials = {}
    for row in rows:
        _parameter_count(row)
        _require(row.get("architecture_family", family) == family and row.get("objective", row["name"]) == row["name"],
                 "objective pair changed architecture or objective identity")
        for field in CONTRACTS:
            _sha(row.get(field), field)
        _require(row["dataset_sha256"] == source["sha256"], "arm original source differs")
        budget = row.get("budget")
        _require(isinstance(budget, dict), "actual fixed training budget required")
        for field in BUDGETS:
            _integer(budget.get(field), field, zero=field in ("tuning_trials", "tuning_flops"))
        _require(budget["examples_seen"] == budget["steps"] * budget["batch_size"], "training examples inconsistent")
        _require(budget["dense_compute_tokens"] == budget["examples_seen"] * budget["history_events"], "dense geometry inconsistent")
        _require(budget["tokens_seen"] <= budget["dense_compute_tokens"], "observed tokens exceed compute positions")
        _require(budget["tuning_trials"] == 0 and budget["tuning_flops"] == 0, "no unregistered tuning allowed")
        p = next(p for p in profiles if p["seed"] == row["seed"] and p["objective"] == row["name"])
        _require(p.get("trainable_parameters") == row["trainable_parameters"] and
                 p.get("deployed_parameters") == row["deployed_parameters"], "profile actual parameter counts differ")
        _require(p.get("fit_sha256") == row["train_split_sha256"], "profile original FIT bytes differ")
        _require(p.get("schedule_sha256") == row["training_sample_order_sha256"], "profile frozen sampler differs")
        for profile_field, budget_field in (("planned_steps", "steps"), ("planned_examples_seen", "examples_seen"),
                                            ("planned_tokens_seen", "tokens_seen"), ("planned_dense_compute_tokens", "dense_compute_tokens")):
            _require(p.get(profile_field) == budget[budget_field], "profile actual scheduled exposure differs: " + budget_field)
        recipe = profile["training_recipe"]
        for recipe_field, budget_field in (("steps", "steps"), ("batch_size", "batch_size"), ("context", "history_events")):
            _require(recipe.get(recipe_field) == budget[budget_field], "profile training recipe geometry differs")
        _require(p.get("weights_before") == p.get("weights_after"), "prefit profile changed model weights")
        _require(p.get("family") == family and p.get("optimizer_steps") == 0,
                 "prefit row changed model family or executed an optimizer")
        _sha(p.get("weights_before"), "initial model weights")
        _require(p.get("initial_state_file") == f"INIT_{family}_{p['seed']}.pt",
                 "profile row initial-state identity differs")
        _sha(p.get("initial_state_sha256"), "profile row initial-state archive")
        initial_key = p.get("initial_state_file"), p.get("initial_state_sha256")
        if initial_key not in verified_initials:
            profile_path = Path(root or Path.cwd()) / manifest["operator_profile_path"]
            verified_initials[initial_key] = verify_initial_state(profile_path, p, family)
        _require(verified_initials[initial_key] == p["weights_before"], "objective initial archive/tensors differ")
        _require(p.get("all_gradients_connected_and_finite") is True, "prefit active gradients incomplete/nonfinite")
        _require(p.get("input_shape") == [budget["batch_size"], budget["history_events"]], "actual profiled geometry differs")
        ledger_counter(p)
        _require(p.get("operator_sha256") == digest(p["operators"]), "profile operator fingerprint differs")
        if phase == "final":
            measured = row.get("measured")
            _require(isinstance(measured, dict), "postfit actual operator/resource receipt missing")
            for field in ("trainable_parameters", "deployed_parameters"):
                _require(measured.get(field) == row[field], "postfit active/deployed model count changed")
            _require(measured.get("initial_model_sha256") == p["weights_before"],
                     "actual training initialization differs from the frozen original profile")
            _require(measured.get("actual_profile_initial_state_sha256") == p["initial_state_sha256"],
                     "actual training initial archive differs from the frozen original profile")
            _sha(measured.get("checkpoint_sha256"), "replayable checkpoint")
            for actual, frozen in (("actual_steps", "steps"), ("actual_examples", "examples_seen"),
                                   ("actual_tokens", "tokens_seen"), ("actual_dense_compute_tokens", "dense_compute_tokens")):
                _require(measured.get(actual) == budget[frozen], "actual training exposure differs: " + actual)
            _require(measured.get("actual_sample_order_sha256") == row["training_sample_order_sha256"], "actual sampler deviated")
            fb = measured.get("forward_backward_operator_step_sha256")
            _require(isinstance(fb, list) and len(fb) == budget["steps"] and all(v == p["operator_sha256"] for v in fb),
                     "every actual forward/backward step must match the frozen complete operator ledger")
            opt = measured.get("optimizer_operator_step_sha256")
            first, warm = measured.get("optimizer_operators_first"), measured.get("optimizer_operators_warm")
            ledger_counter(first)
            ledger_counter(warm)
            _require(isinstance(opt, list) and len(opt) == budget["steps"] and opt[0] == digest(first) and
                     all(v == digest(warm) for v in opt[1:]), "actual optimizer first/warm operator coverage incomplete")
        by_seed[row["seed"]].append((row, p))
    for seed, pair in by_seed.items():
        _require(len(pair) == 2, "duplicate objective row")
        (a, pa), (b, pb) = pair
        for field in CONTRACTS:
            _require(a[field] == b[field], "objective data/selection contract differs: " + field)
        for field in BUDGETS:
            _require(a["budget"][field] == b["budget"][field], "objective training budget differs: " + field)
        _require(a["trainable_parameters"] == b["trainable_parameters"] and a["deployed_parameters"] == b["deployed_parameters"],
                 "same-graph objective pair must have identical parameter counts")
        _require(pa["weights_before"] == pb["weights_before"], "objective pair initial weights differ")
        _require(pa["initial_state_file"] == pb["initial_state_file"] and
                 pa["initial_state_sha256"] == pb["initial_state_sha256"], "objective pair initial archives differ")
        verify_same_ledger(pa, pb, "prefit full/query")
        if phase == "final":
            _require(a["measured"]["initial_model_sha256"] == b["measured"]["initial_model_sha256"],
                     "actual full/query training initializations differ")
            for kind in ("first", "warm"):
                verify_same_ledger(a["measured"]["optimizer_operators_" + kind], b["measured"]["optimizer_operators_" + kind],
                                   "actual full/query optimizer " + kind, require_matrix=False)
    return {"study_id": manifest["study_id"], "phase": phase, "status": "PASS_WITHIN_FAMILY_OPERATOR_PARITY",
            "architecture_family": family, "matched_seeds": sorted(seeds), "matched_objectives": ["full", "query"],
            "eligible_for_architecture_superiority": False, "absolute_training_flops": "INCONCLUSIVE_NOT_CLAIMED",
            "scope": "same executable model; complete mathematical ATen signatures/shapes/counts, not hardware instructions"}


def validate_program(manifest, phase="preflight", *, root=None):
    """A finite four-family index; each objective comparison must pass itself."""
    root = Path(root or Path.cwd()).resolve()
    _require(manifest.get("registered_before_first_fit") is True and
             manifest.get("no_architecture_superiority_claim") is True and
             manifest.get("execution_level") == "internal_research",
             "objective program must be prospective internal research without architecture ranking")
    paths = manifest.get("family_manifests")
    _require(isinstance(paths, dict) and set(paths) == {"tac", "gru", "ssm", "transformer"},
             "all four declared objective-comparison families required")
    seeds = manifest.get("seeds")
    _require(isinstance(seeds, list) and len(seeds) >= 2 and len(set(seeds)) == len(seeds), "complete program seed cohort required")
    from tac_osm.research_execution import validate_internal_research
    reports, children = {}, {}
    for family, relative in sorted(paths.items()):
        _require(isinstance(relative, str) and not Path(relative).is_absolute(), "repository-owned family manifest required")
        path = root / relative
        _require(not path.is_symlink() and path.resolve().is_relative_to(root), "family manifest escapes repository")
        child = json.loads(path.read_bytes())
        _require(child.get("study_type") == "within_family_objective_comparison" and
                 child.get("architecture_family") == family, "objective-program child family/type differs")
        _require(child.get("seeds") == seeds, "family seed cohort differs from complete program")
        _require(child.get("real_world_source", {}).get("sha256") == manifest.get("real_world_source", {}).get("sha256"),
                 "family original source differs from program")
        validate_internal_research(child)
        reports[family] = validate_comparison(child, phase, root=root)
        children[family] = child
    for seed in seeds:
        models = [r for child in children.values() for r in child["models"] if r["seed"] == seed]
        for field in CONTRACTS:
            _require(all(r[field] == models[0][field] for r in models), "cross-family data/exposure contract differs: " + field)
        for field in BUDGETS:
            _require(all(r["budget"][field] == models[0]["budget"][field] for r in models), "cross-family prescribed budget differs: " + field)
        for field in ("trainable_parameters", "deployed_parameters"):
            values = [r[field] for r in models]
            _require((max(values) - min(values)) / max(values) <= .01, "trained-family parameter gap exceeds 1%")
    return {"study_id": manifest["study_id"], "phase": phase,
            "status": "PASS_WITHIN_FAMILY_OBJECTIVE_PROGRAM", "families": reports,
            "models": len(seeds) * 8, "matched_seeds": seeds,
            "eligible_for_architecture_superiority": False,
            "cross_family_compute_parity": "INELIGIBLE_NOT_MEASURED_OR_MATCHED",
            "absolute_training_flops": "INCONCLUSIVE_NOT_CLAIMED"}


class OperatorLedger:
    """Lazy TorchDispatchMode wrapper; importing a gate does not import Torch."""
    def __init__(self, stage):
        self.stage = stage
        self.counter = Counter()
        self.mode = None

    def __enter__(self):
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode
        owner = self
        tensor_type = torch.Tensor
        names = {}
        counter = self.counter
        counter_get = counter.get
        def tensors(values):
            # Most ATen arguments are a flat tuple of tensors/scalars. Avoid
            # a recursive Python call for every scalar and tensor argument.
            result = []
            append = result.append
            for value in values:
                if isinstance(value, tensor_type):
                    append((value.dtype, value.shape))
                elif isinstance(value, (tuple, list)):
                    result.extend(tensors(value))
                elif isinstance(value, dict):
                    result.extend(tensors(value[key] for key in sorted(value)))
            return result
        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                output = func(*args, **kwargs) if kwargs else func(*args)
                name = names.get(func)
                if name is None:
                    name = names[func] = func._schema.name
                inputs = tensors(args)
                if kwargs:
                    inputs.extend(tensors(kwargs[key] for key in sorted(kwargs)))
                outputs = ((output.dtype, output.shape),) if isinstance(output, tensor_type) else tuple(tensors((output,)))
                # Dtypes and torch.Size are immutable/hashable metadata. Never
                # cache by tensor identity: every actual shape is read again.
                key = name, tuple(inputs), outputs
                counter[key] = counter_get(key, 0) + 1
                return output
        self.mode = Mode()
        self.mode.__enter__()
        return self

    def __exit__(self, *args):
        return self.mode.__exit__(*args)

    def rows(self):
        names = {}
        def normalized(tensors):
            result = []
            for dtype, shape in tensors:
                if dtype not in names:
                    names[dtype] = str(dtype)
                result.append((names[dtype], tuple(shape)))
            return tuple(result)
        # Canonical names/JSON are materialized only once per distinct signature,
        # preserving the original serialized ledger bytes and sort convention.
        entries = [((name, normalized(inputs), normalized(outputs)), calls)
                   for (name, inputs, outputs), calls in self.counter.items()]
        return [{"stage": self.stage, "signature": canonical(signature).decode(), "calls": calls}
                for signature, calls in sorted(entries)]

    def sha256(self):
        return digest(self.rows())
