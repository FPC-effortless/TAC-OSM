"""No-fit source/reference audit of the repaired adjoint, FP32 and FP64."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(output):
    wall, cpu = time.perf_counter(), time.process_time()
    resource.setrlimit(resource.RLIMIT_CPU, (115, 120))
    resource.setrlimit(resource.RLIMIT_AS, (8*1024**3,)*2)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    output.mkdir(exist_ok=False, parents=True)
    protocol = json.loads((HERE/"REFERENCE_PROTOCOL.json").read_bytes())
    for name, h in protocol["code_sha256"].items():
        require(sha(HERE/name) == h, "source/reference audit source differs: "+name)
    old = REPO/"research/local_binding_owner_fixed_20261103"
    require(sha(old/"SOURCE_MANIFEST.json") == "5aa2c227e4ee3a58eadd96260fd74d7409a5ced56a1db1f86b99797794589787", "original manifest differs")
    manifest = json.loads((old/"SOURCE_MANIFEST.json").read_bytes())
    require(all(sha(REPO/n) == row["sha256"] for n, row in manifest["files"].items()), "original model source differs")
    core = load("frozen_core", old/"code/core.py")
    fixtures = load("frozen_fixtures", old/"code/fixtures.py")
    import torch
    require(torch.__version__ == "2.13.0+cpu" and torch.version.cuda is None, "pinned CPU runtime required")
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    def no_optimizer(*args, **kwargs):
        raise RuntimeError("NO_FIT: optimizer forbidden")
    torch.optim.Optimizer.__init__ = no_optimizer
    sys.path.insert(0, str(HERE))
    finder = core.PinnedFinder(REPO, manifest["files"])
    sys.meta_path.insert(0, finder)
    from model_repair import build
    from repaired_reference import whole_reference
    from collect018.model import run_chunks, tensors
    from collect018.train import objective
    cases = [fixtures.case(f"fixed-{i}", [("aa", bytes(fixtures.ALPHABET[(i+k)%8] for k in (0, 1, 3, 4)))], "aa", 112, 0, random.Random(70001)) for i in (0, 2, 4)]
    slabs = [fixtures.NATURAL]+[bytes.fromhex(c["slab_hex"]) for c in cases]
    slab = torch.tensor([list(s) for s in slabs], dtype=torch.long)
    x, y = slab[:, :-1], slab[:, 1:]
    mask = torch.zeros_like(y, dtype=torch.bool); first = mask.clone()
    for row in (1, 2, 3):
        mask[row, 112:116] = True; first[row, 112] = True
    owners = tuple(f"independent-audit-row-{i}" for i in range(4))
    rows = []
    def compare(a, b, label, tolerance):
        finite = a is not None and b is not None and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
        ok = finite and torch.allclose(a, b, atol=tolerance, rtol=tolerance)
        rows.append({"label": label, "pass": bool(ok), "tolerance": tolerance, "max_abs": float((a-b).detach().abs().max()) if finite and a.numel() else None})
    models = backwards = 0
    for family in ("phase", "context"):
        model, reference = build(family, 18001), build(family, 18001); models += 2
        require(core.parameter_hash(torch, model) == core.parameter_hash(torch, reference), "initial parameters differ")
        require(sum(p.numel() for p in model.parameters()) == 99971, "wrong live parameter count")
        before = {n: p.detach().clone() for n, p in model.state_dict().items()}
        for precision in (torch.float32, torch.float64):
            model.to(precision); reference.to(precision)
            tag = family+" "+str(precision)
            tolerance = 4e-5 if precision == torch.float32 else 1e-9
            grad_tolerance = 4e-4 if precision == torch.float32 else 1e-8
            with core.capture(torch, model) as (source_routes, embeddings):
                z, state, balance, _ = run_chunks(model, x, chunk=128, owners=owners)
            with core.capture(torch, reference) as (reference_routes, embeddings2):
                rz, rstate = whole_reference(reference, x, owners=owners)
            rb = core.reference_balance(torch, reference_routes)[0]
            compare(z, rz, tag+" logits", tolerance)
            for n, t in tensors(state).items():
                compare(t, tensors(rstate)[n], tag+" state "+n, tolerance)
            for i in range(4):
                compare(source_routes[i].argmax(-1), reference_routes[i].argmax(-1), tag+f" hard route {i}", 0.)
            losses = {"first_ce": torch.nn.functional.cross_entropy(z[first], y[first]), "joint": objective(z, y, mask, first, balance, False)[0], "first": objective(z, y, mask, first, balance, True)[0]}
            refs = core.reference_losses(torch, rz, y, mask, first, rb)
            for loss_name in ("first_ce", "joint", "first"):
                model.zero_grad(set_to_none=True); reference.zero_grad(set_to_none=True)
                compare(losses[loss_name], refs[loss_name], tag+" loss "+loss_name, tolerance)
                losses[loss_name].backward(retain_graph=loss_name != "first")
                refs[loss_name].backward(retain_graph=loss_name != "first"); backwards += 2
                for (n, p), (rn, rp) in zip(model.named_parameters(), reference.named_parameters()):
                    require(n == rn, "parameter names differ")
                    compare(p.grad, rp.grad, tag+" "+loss_name+" gradient "+n, grad_tolerance)
            poison = x.clone(); poison[1:, 113:] = (poison[1:, 113:]+137)%256
            poisoned, _ = model(poison, owners=owners)
            compare(z[1:, :113], poisoned[1:, :113], tag+" future poison", 0.)
            model.zero_grad(set_to_none=True); reference.zero_grad(set_to_none=True)
        require(all(torch.equal(v.double(), model.state_dict()[n]) for n, v in before.items()), "audit changed parameter values")
    result = {"status": "PASS_NO_FIT_REPAIRED_REFERENCE" if all(x["pass"] for x in rows) else "HOLD_REPAIRED_REFERENCE_FAILURES_RETAINED", "comparisons": rows, "comparison_count": len(rows), "failures": sum(not x["pass"] for x in rows), "model_constructors": models, "source_neural_forwards": 8, "reference_equation_forwards": 4, "backward_calls": backwards, "optimizer_constructors": 0, "optimizer_updates": 0, "checkpoint_loads": 0, "CPU_seconds_before_final_receipt": time.process_time()-cpu, "wall_seconds_before_final_receipt": time.perf_counter()-wall, "qualification": "Finite source/reference audit of declared surrogate. Full Lean/IEEE/AD proof and learned improvement remain OPEN."}
    (output/"RESULT.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "comparisons"}, indent=2))
    return result["failures"] == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output) else 2)
