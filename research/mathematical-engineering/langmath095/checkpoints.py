"""Operator-supplied checkpoint diagnostics: CPU FP32, no optimizer or downloads.

Receipts bind bytes to a reviewed source snapshot. They are not independent
publisher authentication, training evidence, or authorization to open TEST.
Run scan substitution in an isolated process: it temporarily replaces module
globals, then restores them even on failure. CUDA/FP16 and carried-state AD
remain OPEN. Importing this module does not import optional PyTorch.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import re
import sys

import numpy as np

from scans import diagonal_scan, matrix_scan

ROOT = Path(__file__).resolve().parents[3]
BASE_COMMIT = "d08fbba33990c503446c2a7d5073219a767f2f62"
SOURCE_HASHES = {
    "research/mathematical-engineering/context026/context_transition.py": "c88f3f7c1b628cb733a4a58dcd168a75411d8bb3d80e382068593713e887510a",
    "research/mathematical-engineering/langlayer038/recurrence_only.py": "e1b52fca3e06b1440d1080924b847aa9649d04cf70ae033287cde37a6dad56f3",
    "research/mathematical-engineering/langscale070/scaled_recurrence.py": "fb51763f1d3f85f2a44bd7bd768b7e34ac1d9ce8f06db90a40ca1abaa76b93a0",
    "research/language-scaling/lang10m/model.py": "ae62392d6d454ceaa96d403c17fc1d6f8436a20a6055a7aca77a5f29a1ad4252",
}
FACTORIES = {"500k": ("research/mathematical-engineering/langscale070/scaled_recurrence.py", "ScaledRecurrenceTAC070", 256),
             "10m": ("research/language-scaling/lang10m/model.py", "TACLanguage10M001", 4096)}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024*1024), b""):
            digest.update(part)
    return digest.hexdigest()


def required_source_hashes(model_kind):
    if model_kind not in FACTORIES:
        raise ValueError("model_kind must be 500k or 10m")
    paths = list(SOURCE_HASHES)[:2] + [FACTORIES[model_kind][0]]
    return {p: SOURCE_HASHES[p] for p in paths}


def validate_receipt(checkpoint, inputs, receipt, model_kind="500k"):
    """Validate all provenance/input boundaries before optional torch loading."""
    expected = required_source_hashes(model_kind)
    if Path(receipt).stat().st_size > 65536:
        raise ValueError("receipt too large")
    record = json.loads(Path(receipt).read_text())
    if not isinstance(record, dict) or record.get("schema") != "langmath095-checkpoint-v1":
        raise ValueError("unsupported receipt schema")
    if record.get("model_kind") != model_kind or record.get("source_commit") != BASE_COMMIT:
        raise ValueError("receipt model/source differs from reviewed snapshot")
    if record.get("partition") not in ("FIT", "DEV", "SELECT_DEV", "TRANSFER_DEV"):
        raise ValueError("only explicitly registered FIT/DEV partitions allowed; TEST forbidden")
    for key in ("publisher", "license"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError("recorded publisher and license required")
    if Path(checkpoint).stat().st_size > 256*1024*1024:
        raise ValueError("checkpoint exceeds bounded no-fit load limit")
    for path, key in ((checkpoint, "checkpoint_sha256"), (inputs, "inputs_sha256")):
        digest = record.get(key)
        if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest) or sha256(path) != digest:
            raise ValueError(key + " does not bind supplied bytes")
    if record.get("source_sha256") != expected:
        raise ValueError("exact reviewed model-source hash map required")
    for path, digest in expected.items():
        if sha256(ROOT / path) != digest:
            raise ValueError("reviewed model source differs: " + path)
    if Path(inputs).stat().st_size > 65536:
        raise ValueError("inputs exceed bounded diagnostic window")
    tokens = np.load(inputs, allow_pickle=False)
    if not isinstance(tokens, np.ndarray):
        tokens.close()
        raise ValueError("inputs must be one NPY array")
    if tokens.dtype != np.int64 or tokens.ndim != 2 or not (1 <= tokens.shape[0] <= 4 and 2 <= tokens.shape[1] <= 1025):
        raise ValueError("inputs must be int64 [B,T+1], B<=4 and 1<=T<=1024")
    if tokens.min() < 0 or tokens.max() >= FACTORIES[model_kind][2]:
        raise ValueError("input token outside registered vocabulary")
    return record, tokens


def _factory(model_kind):
    path, name, _ = FACTORIES[model_kind]
    for folder in ("context026", "langlayer038"):
        location = str(ROOT / "research/mathematical-engineering" / folder)
        if location not in sys.path:
            sys.path.insert(0, location)
    for name_part, folder, filename in (("context_transition", "context026", "context_transition.py"),
                                        ("recurrence_only", "langlayer038", "recurrence_only.py")):
        module = importlib.import_module(name_part)
        if Path(module.__file__).resolve() != (ROOT / "research/mathematical-engineering" / folder / filename).resolve():
            raise ValueError("preloaded source module is outside reviewed snapshot")
    spec = importlib.util.spec_from_file_location("langmath095_factory_" + model_kind, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, name)()


@contextmanager
def scan_substitution(algorithm, modules):
    """Temporarily substitute only scan operators in reviewed Python modules."""
    saved = []
    try:
        for module in modules:
            for name, scan in (("affine_scan", matrix_scan), ("scalar_affine_scan", diagonal_scan)):
                if hasattr(module, name):
                    saved.append((module, name, getattr(module, name)))
                    def replacement(p, q, scan=scan):
                        return scan(p, q, algorithm=algorithm, chunk_size=32)
                    setattr(module, name, replacement)
        yield
    finally:
        for module, name, original in reversed(saved):
            setattr(module, name, original)


def _distribution(value):
    x = value.detach().float().flatten()
    finite = bool(x.isfinite().all())
    if not finite or not x.numel():
        return {"finite": finite, "count": x.numel()}
    return dict(finite=True, count=x.numel(), minimum=float(x.min()), maximum=float(x.max()),
                mean=float(x.mean()), median=float(x.median()))


def _error(a, b, torch, *, atol=3e-5, rtol=1e-4):
    if a.shape != b.shape or a.dtype != b.dtype:
        return {"pass": False, "reason": "shape/dtype differs"}
    if not a.is_floating_point():
        return {"pass": bool(torch.equal(a, b))}
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        return {"pass": False, "reason": "nonfinite tensors"}
    delta = (a-b).abs()
    scale = atol + rtol * torch.maximum(a.abs(), b.abs())
    return {"pass": bool((delta <= scale).all()), "max_abs": float(delta.max()),
            "max_scaled": float((delta/scale).max()), "atol": atol, "rtol": rtol}


def _vjps(model, logits, torch, prefix=""):
    names, params = zip(*model.named_parameters())
    index = torch.arange(logits.numel(), dtype=logits.dtype).reshape_as(logits)
    outputs = []
    for i, coefficient in enumerate((index.sin(), (index*.731 + .3).cos())):
        gradients = torch.autograd.grad((logits*coefficient).mean(), params,
                                        retain_graph=(i == 0), allow_unused=True)
        outputs.append({name.removeprefix(prefix): None if g is None else g.detach()
                        for name, g in zip(names, gradients)})
    return outputs


def _parity(model, tokens, torch):
    from candidate import TACMathCandidate095
    reference, traces = model(tokens, return_trace=True)
    reference = reference.detach() if not reference.requires_grad else reference
    gradients = _vjps(model, reference, torch)
    reference = reference.detach()
    traces = {k: v.detach() for k, v in traces.items()}
    candidate = TACMathCandidate095(model).eval()
    modules = [sys.modules["context_transition"], sys.modules["recurrence_only"], sys.modules["candidate"]]
    results = {}
    for algorithm in ("original", "sequential", "tree", "chunked"):
        @contextmanager
        def unchanged():
            yield
        with unchanged() if algorithm == "original" else scan_substitution(algorithm, modules):
            output, tr = candidate(tokens, return_trace=True)
            vjps = _vjps(candidate, output, torch, "model.")
        forward = {"logits": _error(reference, output.detach(), torch)}
        forward.update({k: _error(value, tr[k].detach(), torch) for k, value in traces.items()})
        backward = []
        for expected, actual in zip(gradients, vjps):
            comparison = {}
            for name, value in expected.items():
                got = actual[name]
                comparison[name] = ({"pass": value is None and got is None, "missing": True}
                                    if value is None or got is None else
                                    _error(value, got, torch, atol=2e-6, rtol=5e-3))
            backward.append(comparison)
        passed = all(v["pass"] for v in forward.values()) and all(v["pass"] for row in backward for v in row.values())
        results[algorithm] = {"status": "PASS_BOUNDED_FIXTURES" if passed else "FAIL", "forward": forward, "parameter_vjps": backward}
    return results


def _diagnostics(model, tokens, labels, torch):
    captured, handles = {}, []
    for name, module in model.named_modules():
        if name in ("native_gate", "aligned_gate", "wave_decay", "concept_gate", "evidence", "embedding", "pos") or name.endswith(".forget"):
            def capture(module, args, output, name=name):
                captured[name] = output.detach()
            handles.append(module.register_forward_hook(capture))
    try:
        logits, trace = model(tokens, return_trace=True)
    finally:
        for handle in handles:
            handle.remove()
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, model.vocab), labels.reshape(-1))
    gradients = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
    named = {}
    for (name, parameter), grad in zip(model.named_parameters(), gradients):
        finite = bool(grad.isfinite().all()) if grad is not None else None
        named[name] = {"missing": grad is None, "finite": finite,
                       "norm": float(grad.double().norm()) if finite else None,
                       "zero_fraction": float((grad == 0).float().mean()) if grad is not None else None,
                       "all_zero": not bool(grad.any()) if grad is not None else None}
    retention = {"native": _distribution(.995*captured["native_gate"].sigmoid()[..., 0]),
                 "aligned": _distribution(captured["aligned_gate"].sigmoid().reshape(*tokens.shape, 2, 3)[..., 0]),
                 "wave": _distribution(.97+.029*captured["wave_decay"].sigmoid()),
                 "concept": _distribution(1-.05*captured["concept_gate"].sigmoid())}
    retention.update({name: _distribution(.995*x.sigmoid()) for name, x in captured.items() if name.endswith(".forget")})
    keys = trace["native_query_key"].detach()
    gram = keys.transpose(-1, -2) @ keys / keys.shape[1]
    evidence = captured["evidence"].tanh().reshape(*tokens.shape, 6, 16)
    pooling = evidence.mean(-1).softmax(-1)
    H = 1.25*torch.eye(6) + .25*trace["concept_laplacian"].detach()
    positional = (captured["embedding"] + captured["pos"]).tanh()
    return {"nll": float(loss.detach()), "full_named_gradients": named, "retention": retention,
            "native_key_gram_eigenvalues": torch.linalg.eigvalsh(gram).tolist(),
            "native_key_rank": torch.linalg.matrix_rank(keys).tolist(),
            "state_frobenius": {k: _distribution(v.detach().flatten(-2).norm(dim=-1))
                                for k, v in trace.items() if k in ("native_memory", "aligned_prewrite", "concept_prewrite")},
            "concept_pooling_uniform_deviation": _distribution((pooling-1/6).abs()),
            "concept_H_eigenvalues": _distribution(torch.linalg.eigvalsh(H)),
            "position_embedding_saturation_abs_gt_099": float((positional.abs() > .99).float().mean())}


def audit_checkpoint(checkpoint, inputs, receipt, model_kind="500k"):
    """Return a JSON-safe result; validation failures never call torch.load."""
    result = {"schema": "langmath095-checkpoint-audit-v1", "model_kind": model_kind,
              "optimizer_steps": 0, "provenance": "operator_provided_not_independently_authenticated",
              "execution_precision": "CPU_FP32", "full_carried_state_ad": "OPEN",
              "CUDA_FP16": "OPEN", "capability_claim": False}
    try:
        record, raw = validate_receipt(checkpoint, inputs, receipt, model_kind)
    except (ValueError, TypeError, KeyError, OSError) as error:
        return {**result, "status": "BLOCKED", "reason": str(error)}
    result["validated_receipt"] = record
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return {**result, "status": "INCONCLUSIVE_TORCH_UNAVAILABLE"}
    try:
        saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
        if not isinstance(saved, dict) or not isinstance(saved.get("model"), dict):
            raise ValueError("checkpoint must contain the registered model state dictionary")
        if not all(isinstance(v, torch.Tensor) for v in saved["model"].values()):
            raise ValueError("model state dictionary must contain only tensors")
        model = _factory(model_kind).float().eval()
        model.load_state_dict(saved["model"], strict=True)
        if not all(bool(p.isfinite().all()) for p in model.parameters()):
            raise ValueError("checkpoint contains nonfinite parameters")
        tokens = torch.from_numpy(raw[:, :-1].copy())
        labels = torch.from_numpy(raw[:, 1:].copy())
        diagnostics = _diagnostics(model, tokens, labels, torch)
        parity = _parity(model, tokens, torch)
        gradient_health = all(not g["missing"] and g["finite"] is True
                              for g in diagnostics["full_named_gradients"].values())
        ok = all(p["status"] != "FAIL" for p in parity.values()) and gradient_health
        step = saved.get("step", saved.get("steps"))
        report = {**result, "status": "PASS_BOUNDED_CPU_FP32_FIXTURES" if ok else "FAIL",
                  "checkpoint_step_metadata_unverified": step if type(step) is int else None,
                  "diagnostics": diagnostics, "gradient_finite_and_connected": gradient_health,
                  "scan_source_parity": parity}
        json.dumps(report, allow_nan=False)
        return report
    except Exception as error:
        return {**result, "status": "BLOCKED_EXECUTION", "reason": str(error)}


def make_window_probe_adapter(model_kind="500k"):
    """Actual reset-window graph; embedding WEIGHT AD is not token-input AD."""
    import torch
    from tac_osm.research_probe import ResearchProbeAdapter
    if model_kind not in FACTORIES:
        raise ValueError("unknown registered model")
    for path, digest in required_source_hashes(model_kind).items():
        if sha256(ROOT / path) != digest:
            raise ValueError("adapter source differs from reviewed snapshot")
    def validate(model, sample):
        x = sample["tokens"]
        owner = sys.modules.get(model.__class__.__module__)
        source_bound = (owner is not None and getattr(owner, "__file__", None) is not None and
                        Path(owner.__file__).resolve() == (ROOT / FACTORIES[model_kind][0]).resolve() and
                        getattr(owner, FACTORIES[model_kind][1], None) is type(model))
        if not source_bound or not isinstance(x, torch.Tensor) or x.dtype != torch.long or x.ndim != 2 or not (1 <= x.shape[0] <= 4 and 1 <= x.shape[1] <= 1024):
            raise ValueError("registered source model and bounded observed window required")
        if x.min() < 0 or x.max() >= FACTORIES[model_kind][2]:
            raise ValueError("observed token outside registered vocabulary")
    def forecast(model, state, sample):
        if state is not None:
            raise ValueError("window adapter has no carried state")
        return {"logits": model(sample["tokens"])}
    def unsupported(*args):
        raise ValueError("no carried-state operation is defined")
    def graph(model, state, sample):
        return forecast(model, state, sample), {"embedding_weight_AD_not_token_input_AD": model.embedding.weight}
    return ResearchProbeAdapter(name="LANGMATH095_"+model_kind+"_WINDOW_RESET_WEIGHT_AD",
        validate=validate, steps=lambda sample: 0, start=lambda model, sample: None,
        advance=unsupported, forecast=forecast, state_views=lambda state: {},
        patch_state=unsupported, batch_size=lambda sample: int(sample["tokens"].shape[0]),
        batch_item=lambda sample, i: {"tokens": sample["tokens"][i:i+1]}, gradient_graph=graph)
