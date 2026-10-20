"""ME023 fixed real-development matrix. Restores only the exposed ME022 source."""
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
from torch.nn import functional as F
from tac_osm.predictive_native import PredictiveNative
from scripts.me014b_exact_action_memory import batch
from scripts.me022_native_readout import predictions, scores, save_json
from scripts.me022_replay import fingerprints

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT/"research/mathematical-engineering/me023/contract.json"
C = json.loads(CONTRACT.read_text())


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_source(parent):
    parent = Path(parent)
    record = json.loads((parent/"results.json").read_text())
    assert record["source"]["zip_sha256"] == C["source_sha256"]
    # Restore every original model hash before reading any cached response.
    for row in record["models"]:
        assert sha(parent/row["checkpoint"]) == row["checkpoint_sha256"]
    assert len(record["models"]) == 12
    assert sha(parent/"data-snapshot.pt") == C["parent_data_snapshot_sha256"] == record["data_snapshot_sha256"]
    data = torch.load(parent/"data-snapshot.pt", weights_only=True)
    expected = json.loads((ROOT/"research/mathematical-engineering/me022/data-fingerprints.json").read_text())
    assert fingerprints(data) == expected
    tr, dev = data["train"], data["dev"]
    assert (len(tr["y"]), len(dev["y"])) == (10776, 5952)
    assert all(uid % 16 == 0 for uid in tr["users"]+dev["users"])
    assert max(tr["timestamps"]) < 1596240000 <= min(dev["timestamps"])
    assert max(dev["timestamps"]) < 1597363200
    assert not data["source"]["future_test_labels_parsed"]
    return data


def make_model(arm, mean, scale):
    return PredictiveNative(arm.startswith("action"), mean, scale)


def atomic_checkpoint(path, record):
    temporary = path.with_suffix(".tmp")
    torch.save(record, temporary)
    temporary.replace(path)


def state_diagnostic(model, train, dev):
    model.eval()
    with torch.no_grad():
        # FIT-only reference removes varying state content while preserving a
        # typical learned state level; do not choose it using development error.
        states = [model.encode(train["items"][i:i+256], train["metadata"][i:i+256])
                  for i in range(0, 1024, 256)]
        mean = torch.cat(states).mean(0, keepdim=True)
        out = []
        for i in range(0, len(dev["y"]), 384):
            sl = slice(i, i+384)
            metadata = dev["metadata"][sl]
            prior = (metadata[:, -32:, 1].sum(-1, keepdim=True)+1.)/34.
            logit = model.readout(mean.expand(len(metadata), -1), dev["action"][sl],
                                  dev["duration"][sl], prior)
            out.append(torch.sigmoid(logit).flatten())
        p = torch.cat(out).numpy()
    return scores(p, dev["y"].numpy()), mean.cpu(), p


def fit(arm, seed, train, dev, mean, scale, root):
    torch.manual_seed(seed)
    model = torch.jit.script(make_model(arm, mean, scale))
    assert sum(p.numel() for p in model.parameters()) == C["parameters"]
    groups = [[], []]
    for name, p in model.named_parameters():
        groups[int(name.startswith("head.") or name == "interaction")].append(p)
    opt = torch.optim.AdamW([{"params": groups[0], "lr": C["base_lr"]},
                             {"params": groups[1], "lr": C["readout_lr"]}],
                            weight_decay=C["weight_decay"])
    sampler = torch.Generator().manual_seed(seed+300)
    name = f"{arm}-seed{seed}"
    trace, cpu, wall = [], 0., 0.
    ckpath = root/f"{name}.pt"
    def checkpoint(step, gradient=None):
        p = predictions(model, dev)
        trace.append({"step": step, "train_cpu_seconds": cpu, "train_wall_seconds": wall,
                      "dev": scores(p, dev["y"].numpy()), "grad_norm_before_clip": gradient,
                      "bounded_matrix_frobenius": float(model.bounded_matrix().detach().norm())})
        save_json(root/f"{name}-trace.json", trace)
        atomic_checkpoint(ckpath, {"state_dict": model.state_dict(), "optimizer": opt.state_dict(),
            "rng": torch.get_rng_state(), "sampler_rng": sampler.get_state(), "step": step,
            "arm": arm, "seed": seed, "duration_mean": mean, "duration_scale": scale,
            "contract": C, "train_cpu_seconds": cpu, "train_wall_seconds": wall})
        return p
    checkpoint(0)
    weights = torch.tensor(C["prefix_loss_weights"]).reshape(1, 4)
    for step in range(1, C["updates"]+1):
        t, w = time.process_time(), time.perf_counter()
        model.train()
        idx = torch.randint(len(train["y"]), (C["batch"],), generator=sampler)
        inputs = batch(train, idx)
        opt.zero_grad(set_to_none=True)
        if arm.endswith("prefix"):
            logits = model.prefix_logits(*inputs)
            y = torch.cat((train["metadata"][idx][:, C["prefix_depths"], 1], train["y"][idx]), -1)
            loss = (F.binary_cross_entropy_with_logits(logits, y, reduction="none")*weights).sum(-1).mean()
        else:
            loss = F.binary_cross_entropy_with_logits(model(*inputs), train["y"][idx])
        assert torch.isfinite(loss)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), C["clip_grad"])
        assert torch.isfinite(gradient)
        opt.step()
        cpu += time.process_time()-t
        wall += time.perf_counter()-w
        if step in C["trace_steps"] or cpu > C["cpu_seconds_ceiling_per_fit"]:
            p = checkpoint(step, float(gradient))
        if cpu > C["cpu_seconds_ceiling_per_fit"]:
            raise RuntimeError("compute ceiling exceeded; partial weights/optimizer/RNG retained")
    replaced, reference_state, rp = state_diagnostic(model, train, dev)
    np.savez_compressed(root/f"{name}-predictions.npz", probability=p, mean_state_probability=rp)
    atomic_checkpoint(root/f"{name}-mean-state.pt", {"state": reference_state})
    standalone = root/f"{name}.jit.pt"
    torch.jit.save(model.eval(), str(standalone))
    with torch.no_grad():
        durations = []
        for j in range(10):
            t = time.perf_counter(); model(*batch(dev, slice(0, 128)))
            if j >= 3: durations.append(time.perf_counter()-t)
    row = {"arm": arm, "seed": seed, "parameters": C["parameters"], "steps": step,
        "sampled_windows": step*C["batch"], "historical_event_presentations": step*C["batch"]*128,
        "supervised_response_presentations": step*C["batch"]*(4 if arm.endswith("prefix") else 1),
        "dev": trace[-1]["dev"], "mean_state_dev": replaced,
        "mean_state_brier_delta": replaced["Brier"]-trace[-1]["dev"]["Brier"],
        "train_cpu_seconds": cpu, "train_wall_seconds": wall,
        "process_lifetime_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "inference_median_seconds_batch128": float(np.median(durations)),
        "checkpoint": ckpath.name, "checkpoint_sha256": sha(ckpath),
        "standalone": standalone.name, "standalone_sha256": sha(standalone),
        "matrix_frobenius": float(model.bounded_matrix().detach().norm()), "trace": trace}
    print(json.dumps({k: row[k] for k in ("arm", "seed", "dev", "mean_state_brier_delta", "train_cpu_seconds")}), flush=True)
    return row


def run(parent, out):
    torch.set_num_threads(1)
    root = Path(out); root.mkdir(parents=True, exist_ok=False)
    start, start_cpu = time.perf_counter(), time.process_time()
    data = load_source(parent)
    train, dev = data["train"], data["dev"]
    mean = float(train["duration"].mean())
    scale = max(float(train["duration"].std(unbiased=False)), 1e-5)
    result = {"study": C["study"], "contract": C, "source": data["source"],
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "contract_sha256": sha(CONTRACT), "duration_mean": mean, "duration_scale": scale,
        "environment": {"torch": str(torch.__version__), "numpy": np.__version__, "platform": platform.platform(), "threads": 1},
        "parent_snapshot_sha256": C["parent_data_snapshot_sha256"],
        "historical_development_only": True, "new_test_outcomes": 0, "models": [],
        "full_compute_parity": False, "architecture_promoted": False}
    save_json(root/"results.json", result)
    for i, seed in enumerate(C["seeds"]):
        arms = C["arms"][i:]+C["arms"][:i]
        for arm in arms:
            result["models"].append(fit(arm, seed, train, dev, mean, scale, root))
            save_json(root/"results.json", result)
    result["execution_wall_seconds"] = time.perf_counter()-start
    result["execution_process_cpu_seconds"] = time.process_time()-start_cpu
    save_json(root/"results.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", default="results/me022")
    parser.add_argument("--out", default="results/me023")
    args = parser.parse_args()
    run(args.parent, args.out)
