"""Real-development-only native TAC input/readout optimization; no test access."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.me014b_exact_action_memory import read_early, batch, FIT_CUT, DEV_CUT, PIN_MD5
from scripts.me019_authentic_history import NativeExtended, original_128_windows, parameters
from scripts.me021_primary_interaction import (
    features, item_rates, metrics, train_logistic_or_mlp, predict_simple,
)

CONTRACT = Path(__file__).resolve().parents[1] / "research/mathematical-engineering/me022/contract.json"
C = json.loads(CONTRACT.read_text())


class ReadoutOptimizedNative(NativeExtended):
    """Only current duration scaling changes; native recurrence is inherited."""

    def __init__(self, mean: float = 0.0, scale: float = 1.0):
        super().__init__(C["history"])
        self.register_buffer("duration_mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("duration_scale", torch.tensor(scale, dtype=torch.float32))

    def forward(self, items: torch.Tensor, metadata: torch.Tensor,
                action: torch.Tensor, duration: torch.Tensor) -> torch.Tensor:
        return super().forward(items, metadata, action,
                               (duration-self.duration_mean)/self.duration_scale)


def scores(p, y):
    result = metrics(p, y)
    p, y = np.asarray(p).ravel(), np.asarray(y).ravel()
    bins = np.minimum((p*10).astype(int), 9)
    result["ECE10"] = float(sum(abs(float(p[bins == k].sum()-y[bins == k].sum()))
                                for k in range(10))/len(y))
    return result


def predictions(net, data):
    net.eval()
    with torch.no_grad():
        return torch.cat([torch.sigmoid(net(*batch(data, slice(i, i+384)))).flatten()
                          for i in range(0, len(data["y"]), 384)]).numpy()


def save_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True)+"\n")
    tmp.replace(path)


def fit(arm, seed, train, dev, mean, scale, root):
    torch.manual_seed(seed)
    net = ReadoutOptimizedNative(mean if "duration_norm" in arm else 0.,
                                 scale if "duration_norm" in arm else 1.)
    if parameters(net) != C["parameters"]:
        raise AssertionError("native active capacity changed")
    # Script only the inherited native module; wrapper's super forward is not
    # TorchScript compatible. Keep identical duration transform outside it.
    core = NativeExtended(C["history"])
    core.load_state_dict({k: v for k, v in net.state_dict().items()
                          if not k.startswith("duration_")})
    net_core = torch.jit.script(core)
    def forward(*args):
        return net_core(*args[:3], (args[3]-net.duration_mean)/net.duration_scale)
    body = [p for name, p in net_core.named_parameters() if not name.startswith("head.")]
    head = list(net_core.head.parameters())
    opt = torch.optim.AdamW([
        {"params": body, "lr": C["base_lr"]},
        {"params": head, "lr": C["base_lr"]*(6 if "head_lr6" in arm else 1)},
    ], weight_decay=C["weight_decay"])
    gen = torch.Generator().manual_seed(seed+300)
    # callable interface for the unchanged common scoring routine
    class Predictor:
        def eval(self): net_core.eval()
        def __call__(self, *args): return forward(*args)
    predictor = Predictor()
    cpu = wall = 0.
    trace = []
    name = f"{arm}-seed{seed}"
    checkpoint = root / f"{name}.pt"
    fit_diag = {k: (v[:1024] if isinstance(v, (torch.Tensor, list)) else v)
                for k, v in train.items()}
    def record(step, grad=None):
        p = predictions(predictor, dev)
        row = {"step": step, "train_cpu_seconds": cpu, "train_wall_seconds": wall,
               "dev": scores(p, dev["y"].numpy()),
               "fit_first1024": scores(predictions(predictor, fit_diag), fit_diag["y"].numpy()),
               "grad_norm_before_clip": grad,
               "duration_weight_in_model_units": float(net_core.head.weight.detach()[0, -1])}
        trace.append(row)
        save_json(root/f"{name}-trace.json", trace)
        torch.save({"state_dict": net_core.state_dict(), "optimizer": opt.state_dict(),
                    "rng": torch.get_rng_state(), "sampler_rng": gen.get_state(),
                    "step": step, "arm": arm, "seed": seed,
                    "duration_mean": float(net.duration_mean),
                    "duration_scale": float(net.duration_scale),
                    "source_sha256": C["source_sha256"], "contract": C}, checkpoint)
        return p
    record(0)
    for step in range(1, C["updates"]+1):
        t, w = time.process_time(), time.perf_counter()
        net_core.train()
        ids = torch.randint(len(train["y"]), (C["batch"],), generator=gen)
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(forward(*batch(train, ids)), train["y"][ids])
        if not torch.isfinite(loss):
            raise ArithmeticError("nonfinite genuine training loss")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(net_core.parameters(), C["clip_grad"])
        if not torch.isfinite(grad):
            raise ArithmeticError("nonfinite genuine gradient")
        opt.step()
        cpu += time.process_time()-t
        wall += time.perf_counter()-w
        if step in C["trace_steps"] or cpu > C["cpu_seconds_ceiling_per_fit"]:
            p = record(step, float(grad))
        if cpu > C["cpu_seconds_ceiling_per_fit"]:
            raise RuntimeError("registered compute ceiling exceeded; partial checkpoint retained")
    # Frozen readout reliance diagnostic, without optimizing the modified head.
    with torch.no_grad():
        net_core.head.weight[:, :64].zero_()
    knockout = scores(predictions(predictor, dev), dev["y"].numpy())
    # Restore final native coefficients for latency and evidence replay.
    net_core.load_state_dict(torch.load(checkpoint, weights_only=True)["state_dict"])
    values = batch(dev, slice(0, 128))
    times = []
    with torch.inference_mode():
        for j in range(10):
            t = time.perf_counter()
            forward(*values)
            if j >= 3: times.append(time.perf_counter()-t)
    np.savez_compressed(root/f"{name}-predictions.npz", probability=p)
    row = {"arm": arm, "seed": seed, "parameters": parameters(net_core),
           "steps": step, "presentations": step*C["batch"],
           "train_cpu_seconds": cpu, "train_wall_seconds": wall,
           "max_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
           "dev": trace[-1]["dev"], "state_readout_knockout_dev": knockout,
           "median_inference_seconds_batch128": float(np.median(times)),
           "checkpoint": checkpoint.name,
           "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
           "forward_affine_macs_per_batch": C["batch"]*(128*(11*32+32*32+43*32+2*64*32)+73),
           "mac_scope": "analytical dense affine forward only; excludes elementwise, backward, embedding and optimizer",
           "deployed_float32_weight_bytes": C["parameters"]*4,
           "recurrent_float32_state_bytes_per_example": 64*4,
           "parity": {"parameters": "PASS", "data_exposure": "PASS",
                      "tuning_selection": "PASS", "same_development_evaluation": "PASS",
                      "actual_training_compute": "UNESTABLISHED", "independent_test": "NOT_OPENED"},
           "trace": trace}
    print(json.dumps({k: row[k] for k in ("arm", "seed", "dev", "train_cpu_seconds")}), flush=True)
    return row


def preflight():
    torch.set_num_threads(1)
    torch.manual_seed(22022)
    base = NativeExtended(128)
    norm = ReadoutOptimizedNative(.4, .15)
    norm.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        base.head.weight.normal_(std=.1)
        base.head.bias.fill_(.2)
        norm.load_state_dict(base.state_dict(), strict=False)
        norm.head.weight[:, -1].mul_(.15)
        norm.head.bias.add_(base.head.weight[:, -1]*.4)
    args = (torch.randint(16000, (3, 128)), torch.rand(3, 128, 3),
            torch.randint(16000, (3,)), torch.rand(3, 1))
    torch.testing.assert_close(base(*args), norm(*args), atol=2e-6, rtol=2e-6)
    scripted = torch.jit.script(base)
    torch.testing.assert_close(base(*args), scripted(*args), atol=2e-6, rtol=2e-6)
    F.binary_cross_entropy_with_logits(norm(*args), torch.ones(3, 1)).backward()
    assert parameters(norm) == C["parameters"]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in norm.parameters())
    torch.manual_seed(123)
    a = NativeExtended(128)
    torch.manual_seed(123)
    b = ReadoutOptimizedNative(.4, .15)
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key])
    torch.testing.assert_close(a(*args), b(*args), atol=0, rtol=0)
    print("ME022 constructed-input algebra/capacity/gradient preflight PASS; no capability evidence", flush=True)


def run(root):
    preflight()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    users, source = read_early()
    if source["zip_sha256"] != C["source_sha256"] or source["zip_md5"] != PIN_MD5:
        raise ValueError("authenticated source mismatch")
    assert not source["future_test_labels_parsed"]
    assert source["later_rows_skipped_before_label_parsing"] > 0
    train = original_128_windows(users, 0, FIT_CUT, C["fit_max"])
    dev = original_128_windows(users, FIT_CUT, DEV_CUT, C["dev_max"])
    assert (len(train["y"]), len(dev["y"])) == (10776, 5952)
    assert max(train["timestamps"]) < FIT_CUT <= min(dev["timestamps"])
    assert max(dev["timestamps"]) < DEV_CUT
    assert all(u % 16 == 0 for u in train["users"]+dev["users"])
    mean = float(train["duration"].mean())
    scale = max(float(train["duration"].std(unbiased=False)), 1e-5)
    torch.save({"train": train, "dev": dev, "source": source}, root/"data-snapshot.pt")
    result = {"study": C["study"], "contract": C, "source": source,
              "fitting_source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "contract_sha256": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
              "data_snapshot_sha256": hashlib.sha256((root/"data-snapshot.pt").read_bytes()).hexdigest(),
              "duration_fit_mean": mean, "duration_fit_std": scale,
              "torch": torch.__version__, "numpy": np.__version__,
              "platform": platform.platform(), "threads": torch.get_num_threads(),
              "source_build_wall_seconds": time.perf_counter()-start,
              "train_windows": len(train["y"]), "dev_windows": len(dev["y"]),
              "historical_development_only": True, "architecture_promoted": False,
              "full_resource_parity": False, "models": [], "unmatched_references": []}
    prior = (dev["metadata"][:, -32:, 1].sum(-1).numpy()+1)/34
    result["causal_prior32"] = scores(prior, dev["y"].numpy())
    save_json(root/"results.json", result)
    for i, seed in enumerate(C["seeds"]):
        arms = C["arms"][i:]+C["arms"][:i]
        for arm in arms:
            result["models"].append(fit(arm, seed, train, dev, mean, scale, root))
            save_json(root/"results.json", result)
    rates, _ = item_rates(train)
    x, xd = features(train, rates), features(dev, rates)
    for seed in C["seeds"]:
        ref = train_logistic_or_mlp(x, train["y"].numpy(), seed, True)
        prob = predict_simple(ref, xd)
        result["unmatched_references"].append({"seed": seed, "parameters": ref["parameters"],
            "train_seconds": ref["train_seconds"], "dev": scores(prob, dev["y"].numpy())})
        np.savez_compressed(root/f"unmatched-mlp-seed{seed}-predictions.npz", probability=prob)
        torch.save({"state_dict": ref["network"].state_dict(),
                    "mean": torch.tensor(ref["mean"]), "std": torch.tensor(ref["std"]),
                    "item_rates": torch.tensor(rates)}, root/f"unmatched-mlp-seed{seed}.pt")
    result["end_to_end_wall_seconds"] = time.perf_counter()-start
    save_json(root/"results.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/me022")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    preflight() if args.self_test else run(args.out)
