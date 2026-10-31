"""LANGCAP103 instantiated-model engineering preflight. No optimizer or dataset.

Constructed byte inputs are shape/gradient tests, never capability evidence.
Counts come from real model instances; RAM arithmetic excludes activations.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys

HERE = Path(__file__).resolve().parent


def checked_sources():
    pins = json.loads((HERE / "SOURCE_PINS.json").read_text())
    for name, expected in pins["sha256"].items():
        actual = hashlib.sha256((HERE / "vendor" / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("Source pin mismatch: " + name)
    return pins


def load_native():
    checked_sources()
    sys.path.insert(0, str(HERE / "vendor"))
    spec = importlib.util.spec_from_file_location("langcap103_native", HERE / "vendor/models.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def counts(model):
    return {"trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "deployed_parameters": sum(p.numel() for p in model.parameters())}


def parameter_digest(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def comparator_classes(torch):
    nn = torch.nn

    class CausalTransformer(nn.Module):
        """Independent pre-norm causal blocks, GELU FFN, sinusoidal positions.

        Standard multilayer reference, not a pretrained frontier model.
        No KV cache is implemented; no competitive decode-speed claim is made.
        """
        def __init__(self, width, depth, ffn):
            super().__init__()
            self.width = width
            self.embedding = nn.Embedding(256, width)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(width, 8, ffn, dropout=0.,
                                           activation="gelu", batch_first=True, norm_first=True)
                for _ in range(depth)])
            self.norm = nn.LayerNorm(width)
            self.head = nn.Linear(width, 256)

        def forward(self, tokens):
            n = tokens.shape[1]
            x = self.embedding(tokens) * math.sqrt(self.width)
            positions = torch.arange(n, device=x.device, dtype=x.dtype)[:, None]
            frequencies = torch.exp(torch.arange(0, self.width, 2, device=x.device,
                dtype=x.dtype) * (-math.log(10000.) / self.width))
            phase = positions * frequencies
            x = x + torch.stack((phase.sin(), phase.cos()), -1).flatten(-2)[None]
            mask = torch.ones(n, n, dtype=torch.bool, device=x.device).triu(1)
            for layer in self.layers:
                x = layer(x, src_mask=mask, is_causal=True)
            return self.head(self.norm(x))

    class StackedGRU(nn.Module):
        """Four-layer PyTorch GRU; not called a Mamba/SSM reproduction."""
        def __init__(self, width, depth=4):
            super().__init__()
            self.embedding = nn.Embedding(256, width)
            self.recurrent = nn.GRU(width, width, depth, batch_first=True)
            self.head = nn.Linear(width, 256)

        def forward(self, tokens):
            x, state = self.recurrent(self.embedding(tokens))
            return self.head(x), state

    return CausalTransformer, StackedGRU


def choose_configs(target, torch, native):
    # Deterministic geometry selection uses counts only, no source or outcomes.
    options = []
    native.TARGET = target
    with torch.device("meta"):
        for width in range(64, 1537, 8):
            model = native.StackedCandidate(depth=8, d=width)
            n = counts(model)["trainable_parameters"]
            core = sum(p.numel() for name, p in model.named_parameters() if not name.startswith("head."))
            # Avoid achieving size merely through a disproportionately large readout.
            if .70 * target <= core <= .95 * target and abs(n-target) <= .01 * target:
                options.append((abs(n-target), -core, width))
    if not options:
        raise ValueError("No eligible native geometry")
    width = min(options)[2]
    d, depth = (128, 4) if target == 1_000_000 else (256, 8)
    fixed = 514*d + 256 + depth*(4*d*d + 9*d)
    ffn = round((target-fixed)/(depth*(2*d+1)))
    if ffn < d:
        raise ValueError("Transformer FFN would be underprovisioned")
    gru_width = min(range(32, 1025), key=lambda h: abs(24*h*h+536*h+256-target))
    return [
        {"arm": "tac_associative_stack", "width": width, "depth": 8},
        {"arm": "causal_transformer", "width": d, "depth": depth, "ffn": ffn},
        {"arm": "stacked_gru", "width": gru_width, "depth": 4},
    ]


def build(config, target, native, classes):
    args = {k: v for k, v in config.items() if k != "arm"}
    if config["arm"] == "tac_associative_stack":
        native.TARGET = target
        return native.StackedCandidate(depth=args["depth"], d=args["width"])
    return classes[0 if config["arm"] == "causal_transformer" else 1](**args)


def run():
    import torch
    torch.set_num_threads(2)
    pins = checked_sources()
    native = load_native()
    classes = comparator_classes(torch)
    rows = []
    for target in (1_000_000, 10_000_000):
        cohort = []
        for config in choose_configs(target, torch, native):
            torch.manual_seed(1729)
            model = build(config, target, native, classes).float().eval()
            n = counts(model)
            if any(abs(v-target)/target > .05 for v in n.values()):
                raise ValueError("Instantiated size outside declared candidate band")
            before = parameter_digest(model)
            # Engineering input only. No training loss, source labels or optimizer.
            tokens = torch.arange(8, dtype=torch.long)[None]
            output = model(tokens)
            logits = output[0] if isinstance(output, tuple) else output
            if logits.shape != (1, 8, 256) or not bool(torch.isfinite(logits).all()):
                raise ValueError("Invalid output shape/nonfinite logits")
            logits.square().mean().backward()
            missing = [name for name, p in model.named_parameters() if p.grad is None]
            nonfinite = [name for name, p in model.named_parameters()
                         if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
            if missing or nonfinite or parameter_digest(model) != before:
                raise ValueError("Disconnected/nonfinite pathway or unexpected parameter mutation")
            # Perturb only future fixture bytes; earlier predictions must stay fixed.
            with torch.no_grad():
                changed = tokens.clone(); changed[:, 4:] = 255
                changed_output = model(changed)
                changed_logits = changed_output[0] if isinstance(changed_output, tuple) else changed_output
            if not torch.allclose(logits[:, :4], changed_logits[:, :4], atol=1e-6, rtol=1e-5):
                raise ValueError("Future-byte leakage in constructed prefix probe")
            state_bytes = (model.start(("fixture",)).tensor_bytes()
                           if config["arm"] == "tac_associative_stack" else
                           config["depth"]*config["width"]*4
                           if config["arm"] == "stacked_gru" else None)
            head = sum(p.numel() for name, p in model.named_parameters() if name.startswith("head."))
            cohort.append({"target": target, "configuration": config, **n,
                "fp32_weight_bytes": 4*n["deployed_parameters"],
                "fp32_parameter_gradient_two_adam_moments_bytes": 16*n["trainable_parameters"],
                "memory_estimate_excludes": ["activations", "workspaces", "allocator", "input", "cache"],
                "persistent_recurrent_bytes_per_stream": state_bytes,
                "readout_parameter_fraction": head/n["trainable_parameters"],
                "fixture_forward_backward_and_causal_prefix": "PASS_ENGINEERING_ONLY",
                "untrained_weight_sha256": before,
                "optimizer_updates": 0})
            del model, logits, output, changed_output, changed_logits
        ns = [r["trainable_parameters"] for r in cohort]
        gap = (max(ns)-min(ns))/max(ns)
        if gap > .01:
            raise ValueError("Instantiated cross-arm parameter gap exceeds 1%")
        rows.append({"candidate_target": target, "relative_parameter_gap": gap, "arms": cohort})
    return {"schema": "langcap103.engineering-prefit.v1",
        "status": "MEASURED_UNTRAINED_CANDIDATES_NOT_FROZEN_REGISTRATION",
        "torch_version": torch.__version__, "python": platform.python_version(),
        "source_pins": pins, "cohorts": rows,
        "training_enabled": False, "real_data_accessed": False, "optimizer_updates": 0,
        "training_flops": None, "trained_latency": None, "capability_result": None,
        "training_blockers": ["authenticated training corpus and frozen splits",
            "independent source/reward/ACL prefit attestations",
            "PNDS source-equivalent formulation and actual graph/probe coverage",
            "frozen FAIR_COMPARISON manifest, training/tuning and measured FLOP budgets",
            "positive-control solvability and licensed evidence-dependent capability cohort",
            "authorized execution venue and quota"]}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    report = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "optimizer_updates": 0,
                      "cohorts": [{"target": c["candidate_target"],
                        "gap": c["relative_parameter_gap"],
                        "counts": [a["trainable_parameters"] for a in c["arms"]]}
                        for c in report["cohorts"]]}))
