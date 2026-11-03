"""Bounded, preregistered read-only checkpoint diagnosis; NEVER fits a model."""
import argparse
from collections import Counter
from contextlib import contextmanager
import dataclasses
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
COUNTS = {"model_constructors": 0, "checkpoint_loads": 0, "neural_forwards": 0, "neural_backwards": 0, "optimizer_constructors": 0, "optimizer_updates": 0, "autograd_grad_API_calls": 0}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def json_save(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main(args):
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    resource.setrlimit(resource.RLIMIT_CPU, (285, 290))
    resource.setrlimit(resource.RLIMIT_AS, (8*1024**3, 8*1024**3))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8*1024**2, 8*1024**2))
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    json_save(out/"STARTED.json", {"status": "NO_FIT_DIAGNOSIS_STARTED", "counts": COUNTS, "wall_limit_seconds": 300, "CPU_limit_seconds": 290})
    protocol = read(HERE/"PROTOCOL.json")
    require(protocol["optimizer_updates_allowed"] == 0, "this runner cannot fit")
    for name, h in protocol["code_sha256"].items():
        require(sha(HERE/name) == h, "diagnosis source changed: "+name)
    old = REPO/"research/local_binding_owner_fixed_20261103"
    manifest = read(old/"SOURCE_MANIFEST.json")
    require(sha(old/"SOURCE_MANIFEST.json") == protocol["source_manifest_sha256"], "source manifest changed")
    for name, row in manifest["files"].items():
        require(sha(REPO/name) == row["sha256"], "original model source changed: "+name)
    artifact = args.artifact.resolve()
    child = artifact/"runs/fixed/default/output/child"
    original_result = child/"RESULT.json"
    require(sha(original_result) == protocol["original_result_sha256"], "wrong trial result")
    original = read(original_result)
    for name, h in protocol["checkpoint_sha256"].items():
        require(sha(child/name) == h, "wrong checkpoint: "+name)
    fixed = load_module("fixed_cases", old/"code/fixtures.py")
    # Constructed FIT examples only. No held-out fixture or protected TEST read.
    cases = [fixed.case(f"fixed-{i}", [("aa", bytes(fixed.ALPHABET[(i+k)%8] for k in (0, 1, 3, 4)))], "aa", 112, 0, random.Random(70001)) for i in range(8)]
    fixture_bytes = (json.dumps(cases, indent=2)+"\n").encode()
    require(hashlib.sha256(fixture_bytes).hexdigest() == protocol["fixed_fit_fixture_sha256"], "fixed FIT cases changed")
    require(len(cases) == 8, "eight fixed examples required")
    for case in cases:
        prefix, query = bytes.fromhex(case["prefix_hex"]), bytes.fromhex(case["query_hex"])
        target, slab = bytes.fromhex(case["target_hex"]), bytes.fromhex(case["slab_hex"])
        require(fixed.factual_answer(prefix, query) == target, "actual-byte target mismatch")
        require(slab[:113] == prefix+query and slab[113:117] == target and case["first_position"] == 112, "next-byte alignment defect")

    import torch
    require(torch.__version__ == "2.13.0+cpu" and torch.version.cuda is None, "pinned CPU Torch required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    def deny_optimizer(*a, **kw):
        COUNTS["optimizer_constructors"] += 1
        raise RuntimeError("NO_FIT: optimizer construction forbidden")
    torch.optim.Optimizer.__init__ = deny_optimizer
    old_grad = torch.autograd.grad
    def counted_grad(*a, **kw):
        COUNTS["autograd_grad_API_calls"] += 1
        return old_grad(*a, **kw)
    torch.autograd.grad = counted_grad
    sys.path.insert(0, str(HERE))
    core = load_module("frozen_core", old/"code/core.py")
    finder = core.PinnedFinder(REPO, manifest["files"])
    sys.meta_path.insert(0, finder)
    from collect018.common import build as build_legacy
    from collect018.model import tensors
    from collect018.probe import adapter as source_adapter
    from collect011.memory import write_step as legacy_write
    from model_repair import build as build_repaired
    import kernel_checks
    kernel = kernel_checks.run(legacy_write)
    json_save(out/"KERNEL_CHECKS.json", kernel)

    prefix = torch.tensor([list(bytes.fromhex(c["prefix_hex"])) for c in cases], dtype=torch.long)
    query = torch.tensor([list(bytes.fromhex(c["query_hex"])) for c in cases], dtype=torch.long)
    tokens = torch.cat((prefix, query), 1)
    whole = torch.tensor([list(bytes.fromhex(c["slab_hex"]))[:-1] for c in cases], dtype=torch.long)
    targets = torch.tensor([bytes.fromhex(c["target_hex"])[0] for c in cases], dtype=torch.long)
    value_start = bytes.fromhex(cases[0]["prefix_hex"]).index(bytes.fromhex(cases[0]["target_hex"]))
    owners = tuple(f"diagnosis-row-{i}" for i in range(8))

    def count_forward(model, inputs):
        COUNTS["neural_forwards"] += 1
        require(COUNTS["neural_forwards"] <= protocol["max_neural_forwards"], "forward budget exhausted")
        require(time.perf_counter()-start_wall < 290, "wall stop")
    def forward(model, value, **kwargs):
        return model(value, owners=owners, **kwargs)

    def stats(value):
        value = value.detach().double()
        return {"rms": float(value.square().mean().sqrt()), "max_abs": float(value.abs().max()), "finite": bool(torch.isfinite(value).all())}

    def difference(left, right):
        return {"max_abs": float((left-right).abs().max()), "bitwise_tensor_equal": bool(torch.equal(left.view(torch.uint8), right.view(torch.uint8))), "within_4e_minus5": bool(torch.allclose(left, right, atol=4e-5, rtol=4e-5)), "argmax_equal": bool(torch.equal(left.argmax(-1), right.argmax(-1)))}

    @contextmanager
    def trace(model):
        data, handles = {"layers": {}, "routers": {}, "admission": {}, "embedding": []}, []
        def embedding(module, inputs, output):
            if output.requires_grad:
                output.retain_grad()
            data["embedding"].append(output)
        handles.append(model.core.embedding.register_forward_hook(embedding))
        for index, block in enumerate(getattr(model.core, "blocks", [])):
            def layer_hook(module, inputs, output, index=index):
                data["layers"][index] = output[0]
            handles.append(block.register_forward_hook(layer_hook))
            def router_hook(module, inputs, output, index=index):
                data["routers"][index] = output
            def admission_hook(module, inputs, output, index=index):
                data["admission"][index] = output
            handles.append(block.phase_memory.router.register_forward_hook(router_hook))
            handles.append(block.phase_memory.admission.register_forward_hook(admission_hook))
        try:
            yield data
        finally:
            for handle in handles:
                handle.remove()

    def set_clocks(value, clock):
        if dataclasses.is_dataclass(value):
            return dataclasses.replace(value, **{f.name: clock if f.name == "clock" else set_clocks(getattr(value, f.name), clock) for f in dataclasses.fields(value)})
        if isinstance(value, tuple):
            return tuple(set_clocks(x, clock) for x in value)
        return value

    def clone(value):
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if dataclasses.is_dataclass(value):
            return dataclasses.replace(value, **{f.name: clone(getattr(value, f.name)) for f in dataclasses.fields(value)})
        if isinstance(value, tuple):
            return tuple(clone(x) for x in value)
        return value

    def checkpoint(model, family, step):
        p = child/f"{family}_checkpoint_{step:04}.pt"
        COUNTS["checkpoint_loads"] += 1
        saved = torch.load(p, map_location="cpu", weights_only=True)
        require(saved["step"] == step and saved["family"] == family, "checkpoint metadata mismatch")
        model.load_state_dict(saved["model"], strict=True)
        require(all(torch.equal(v, saved["model"][k]) for k, v in model.state_dict().items()), "live checkpoint differs")
        model.eval()

    diagnoses, models, saved_logits, saved_gradients = {}, {}, {}, {}
    for name in ("gru", "phase", "context", "phase_repaired", "context_repaired"):
        family = name.split("_")[0]
        repaired = name.endswith("_repaired")
        model = build_repaired(family, 18001) if repaired else build_legacy(family+"_first", 18001)
        COUNTS["model_constructors"] += 1
        model.register_forward_pre_hook(count_forward)
        expected = 100323 if family == "gru" else 99971
        require(sum(p.numel() for p in model.parameters()) == expected, "active count changed")
        models[name] = model
        row = {"active_parameters": expected, "repair": repaired, "checkpoint_family": family, "optimizer_updates": 0}
        if not repaired:
            checkpoint(model, family, 0)
            with torch.no_grad():
                z0, _ = forward(model, tokens)
            row["initial_first_logits"] = stats(z0[:, -1])
        checkpoint(model, family, 128)
        before = {n: p.detach().clone() for n, p in model.state_dict().items()}
        with torch.no_grad():
            with trace(model) as observed:
                all_logits, final_state = forward(model, tokens)
            logit = all_logits[:, -1].contiguous()
            row["first_predictions_ascii"] = bytes(logit.argmax(-1).tolist()).decode("ascii", errors="backslashreplace")
            row["first_correct_of_8"] = int((logit.argmax(-1) == targets).sum())
            row["first_CE"] = float(torch.nn.functional.cross_entropy(logit, targets))
            row["between_example_logit_rms"] = stats(logit-logit.mean(0, keepdim=True))["rms"]
            row["layers"] = {}
            whole_routes = {}
            for index, activation in observed["layers"].items():
                route = observed["routers"][index].softmax(-1)
                hard = route.argmax(-1)
                whole_routes[index] = hard.detach().clone()
                source_route = hard[:, value_start:value_start+4]
                query_route = hard[:, -1:]
                row["layers"][str(index)] = {
                    "last_activation": stats(activation[:, -1]),
                    "last_between_example_rms": stats(activation[:, -1]-activation[:, -1].mean(0, keepdim=True))["rms"],
                    "head_counts": torch.nn.functional.one_hot(hard, 4).sum((0, 1)).tolist(),
                    "last_query_routes": hard[:, -1].tolist(),
                    "record_value_routes": source_route.tolist(),
                    "record_bytes_in_last_query_bank": int((source_route == query_route).sum()),
                    "router_entropy_mean": float(-(route*route.clamp_min(1e-30).log()).sum(-1).mean()),
                    "admission_sigmoid_mean": float(observed["admission"][index].sigmoid().mean()),
                }
            z_whole, _ = forward(model, whole)
            row["future_suffix_invariance"] = difference(logit, z_whole[:, 112].contiguous())
            _, state = forward(model, prefix)
            prefix_hash = {k: core.tensor_hash(torch, t) for k, t in tensors(state).items()}
            with trace(model) as streamed:
                z_query, query_state = forward(model, query, state=clone(state))
            row["whole_vs_prefix_query"] = difference(logit, z_query[:, -1].contiguous())
            adapter = source_adapter(model)
            sample = {"history": prefix, "query": query, "owners": owners}
            adapter.validate(model, sample)
            adapter_logits = adapter.forecast(model, state, sample)["byte_logits"]
            row["probe_adapter"] = {"name": adapter.name, "source": "research/collect018/collect018/probe.py", "state_tensor_names": list(adapter.state_views(state)), "forecast_matches_direct": difference(z_query[:, -1].contiguous(), adapter_logits.contiguous()), "candidate_permutation": "INCONCLUSIVE_NO_CANDIDATE_AXIS", "whole_model_IEEE_AD_proof": "OPEN"}
            single, _ = model(tokens[:1], owners=owners[:1])
            row["batch_entity_isolation"] = difference(all_logits[:1].contiguous(), single.contiguous())
            row["query_route_mismatches_vs_whole"] = {str(i): int((streamed["routers"][i].argmax(-1)-whole_routes[i][:, -9:]).ne(0).sum()) for i in whole_routes}
            historical = next(x for x in original["training_probe"] if x["family"] == family and x["step"] == 128 and x["label"] == "training_probe")
            recorded = torch.tensor([historical["rows"][c["id"]]["first_logits"] for c in cases])
            row["saved_trial_logits_replay"] = difference(z_query[:, -1].contiguous(), recorded.contiguous())
            require(row["saved_trial_logits_replay"]["within_4e_minus5"], "saved trial first logits not reproduced")
            if family == "gru":
                reset = dataclasses.replace(state, hidden=torch.zeros_like(state.hidden))
                row["query_write_freeze"] = {"status": "INCONCLUSIVE_GRU_HAS_NO_OBSERVATION_FREEZE"}
            else:
                reset = set_clocks(model.start(owners), state.clock)
                frozen, _ = forward(model, query, state=clone(state), write_mask=torch.zeros_like(query, dtype=torch.bool))
                row["query_write_freeze"] = difference(z_query[:, -1].contiguous(), frozen[:, -1].contiguous())
                phase_reset = dataclasses.replace(state, phase=tuple(dataclasses.replace(p, memory=torch.zeros_like(p.memory)) for p in state.phase))
                no_phase, _ = forward(model, query, state=phase_reset)
                row["phase_memory_reset"] = difference(z_query[:, -1].contiguous(), no_phase[:, -1].contiguous())
            no_history, _ = forward(model, query, state=reset)
            row["all_memory_reset"] = difference(z_query[:, -1].contiguous(), no_history[:, -1].contiguous())
            require(prefix_hash == {k: core.tensor_hash(torch, t) for k, t in tensors(state).items()}, "diagnostic mutated prefix state")
            saved_logits[name] = logit.detach().clone()

        model.zero_grad(set_to_none=True)
        with trace(model) as traced:
            z, _ = forward(model, tokens)
            loss = torch.nn.functional.cross_entropy(z[:, -1], targets)
            COUNTS["neural_backwards"] += 1
            require(COUNTS["neural_backwards"] <= 5, "backward budget exhausted")
            loss.backward()
        require(len(traced["embedding"]) == 1 and traced["embedding"][0].grad is not None, "actual embedding gradient not captured")
        g = traced["embedding"][0].grad.detach().double().square().sum(-1).sqrt()
        row["input_gradient_mean_norm_by_position"] = g.mean(0).tolist()
        row["record_value_gradient_mean_norm"] = float(g[:, value_start:value_start+4].mean())
        row["last_query_gradient_mean_norm"] = float(g[:, -1].mean())
        row["gradient_source_to_last_query_ratio"] = row["record_value_gradient_mean_norm"] / max(row["last_query_gradient_mean_norm"], 1e-300)
        grads = {}
        for n, p in model.named_parameters():
            grads[n] = None if p.grad is None else {**stats(p.grad), "nonzero_elements": int(p.grad.ne(0).sum()), "elements": p.numel()}
        row["parameter_gradients_first_CE_only"] = grads
        saved_gradients[name] = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None and "phase_memory.router" in n}
        model.zero_grad(set_to_none=True)
        if hasattr(model, "router_logits"):
            model.router_logits = []
        require(all(torch.equal(v, before[n]) for n, v in model.state_dict().items()), "read-only diagnosis changed checkpoint weights")
        row["checkpoint_weights_unchanged"] = True
        diagnoses[name] = row
        json_save(out/(name+".json"), row)

    paired = {}
    for family in ("phase", "context"):
        a, b = saved_logits[family], saved_logits[family+"_repaired"]
        require(torch.equal(a.view(torch.uint8), b.view(torch.uint8)), "AD-only repair changed forward logits")
        ga, gb = saved_gradients[family], saved_gradients[family+"_repaired"]
        paired[family] = {"trained_checkpoint_forward_bitwise_equal": True, "router_gradient_change": {n: {"old": stats(ga[n]), "repaired": stats(gb[n]), "difference": stats(gb[n]-ga[n])} for n in ga}}
    require(COUNTS["model_constructors"] == 5 and COUNTS["checkpoint_loads"] == 8, "construction/load counts differ")
    require(COUNTS["optimizer_constructors"] == COUNTS["optimizer_updates"] == 0, "no-fit boundary violated")
    require(all(sha(REPO/n) == r["sha256"] for n, r in manifest["files"].items()), "original source mutated")
    result = {"status": "COMPLETE_NO_FIT_DIAGNOSIS", "counts": COUNTS, "data": "eight previously published fixed FIT cases only; no held-out or protected TEST reads", "source_manifest_sha256": sha(old/"SOURCE_MANIFEST.json"), "source_imports": finder.loaded, "value_source_positions": list(range(value_start, value_start+4)), "kernel": kernel, "paired_repair": paired, "models": diagnoses, "CPU_seconds_before_final_receipt": time.process_time()-start_cpu, "wall_seconds_before_final_receipt": time.perf_counter()-start_wall, "qualification": "Mechanism diagnosis and regression only. Repair changes adjoints, not stored weights or inference behavior. No learned improvement established; full graph/AD/IEEE proof remains OPEN."}
    json_save(out/"RESULT.json", result)
    print(json.dumps({k: result[k] for k in ("status", "counts", "paired_repair", "CPU_seconds_before_final_receipt", "wall_seconds_before_final_receipt")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except BaseException as error:
        if args.output.is_dir() and not (args.output/"FAILURE.json").exists():
            json_save(args.output/"FAILURE.json", {"status": "FAILED_RETAINED_NO_FIT", "error": type(error).__name__+": "+str(error), "counts": COUNTS})
        raise
