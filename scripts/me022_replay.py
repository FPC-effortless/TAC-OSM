"""Recorded-only ME022 audit. No downloads, fitting or new test outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.me019_authentic_history import NativeExtended
from scripts.me022_native_readout import predictions, scores
from scripts.me021_primary_interaction import features, predict_simple


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprints(data):
    result = {}
    for split in ("train", "dev"):
        result[split] = {}
        for key in ("items", "metadata", "action", "duration", "y", "users", "timestamps"):
            value = data[split][key]
            raw = (value.contiguous().numpy().tobytes() if isinstance(value, torch.Tensor)
                   else json.dumps(value, separators=(",", ":")).encode())
            result[split][key] = hashlib.sha256(raw).hexdigest()
    return result


def audit(root, rebuild_source=False):
    root = Path(root)
    r = json.loads((root/"results.json").read_text())
    c = r["contract"]
    expected = {(arm, seed) for arm in c["arms"] for seed in c["seeds"]}
    assert {(row["arm"], row["seed"]) for row in r["models"]} == expected
    assert len(r["models"]) == len(expected) == 12
    for row in r["models"]:
        assert sha(root/row["checkpoint"]) == row["checkpoint_sha256"]
    # Only read stored development responses after every original weight hash.
    if (root/"data-snapshot.pt").exists():
        assert sha(root/"data-snapshot.pt") == r["data_snapshot_sha256"]
        data = torch.load(root/"data-snapshot.pt", weights_only=True)
    elif rebuild_source:
        from scripts.me014b_exact_action_memory import read_early, FIT_CUT, DEV_CUT
        from scripts.me019_authentic_history import original_128_windows
        users, source = read_early()
        assert source["zip_sha256"] == c["source_sha256"]
        data = {"train": original_128_windows(users, 0, FIT_CUT, c["fit_max"]),
                "dev": original_128_windows(users, FIT_CUT, DEV_CUT, c["dev_max"]),
                "source": source}
    else:
        raise FileNotFoundError("Provide original local data-snapshot.pt or --rebuild-source (official early source only)")
    expected_fingerprints = json.loads((root/"data-fingerprints.json").read_text())
    assert fingerprints(data) == expected_fingerprints
    tr, dev = data["train"], data["dev"]
    assert all(u % 16 == 0 for u in tr["users"]+dev["users"])
    assert max(tr["timestamps"]) < 1596240000 <= min(dev["timestamps"])
    assert max(dev["timestamps"]) < 1597363200
    assert not data["source"]["future_test_labels_parsed"]
    torch.set_num_threads(1)
    sampler = {}
    errors = []
    for row in r["models"]:
        ck = torch.load(root/row["checkpoint"], weights_only=True)
        assert ck["step"] == row["steps"] == c["updates"]
        assert row["parameters"] == c["parameters"]
        assert row["train_cpu_seconds"] <= c["cpu_seconds_ceiling_per_fit"]
        assert row["presentations"] == c["updates"]*c["batch"]
        assert [v["step"] for v in row["trace"]] == c["trace_steps"]
        assert ck["source_sha256"] == c["source_sha256"]
        assert ck["contract"] == c
        if row["seed"] in sampler:
            assert torch.equal(sampler[row["seed"]], ck["sampler_rng"])
        sampler[row["seed"]] = ck["sampler_rng"]
        if "duration_norm" in row["arm"]:
            assert ck["duration_mean"] == r["duration_fit_mean"]
            assert ck["duration_scale"] == r["duration_fit_std"]
        else:
            assert ck["duration_mean"] == 0 and ck["duration_scale"] == 1
        net = NativeExtended(128)
        net.load_state_dict(ck["state_dict"], strict=True)
        net = torch.jit.script(net).eval()
        class Predictor:
            def eval(self): net.eval()
            def __call__(self, *args):
                return net(*args[:3], (args[3]-ck["duration_mean"])/ck["duration_scale"])
        p = predictions(Predictor(), dev)
        old = np.load(root/f'{row["arm"]}-seed{row["seed"]}-predictions.npz')["probability"]
        error = float(np.max(np.abs(p-old)))
        assert error <= 1e-7
        errors.append(error)
        for metric, value in scores(p, dev["y"].numpy()).items():
            assert abs(value-row["dev"][metric]) <= 1e-9
        # Fold FIT normalization into the trained linear head. Deployment then
        # consumes original duration units with the original native equations.
        # No new fit, model selection, buffers or feature-building runtime.
        folded = NativeExtended(128)
        folded.load_state_dict(ck["state_dict"], strict=True)
        with torch.no_grad():
            w = folded.head.weight[:, -1].clone()
            folded.head.weight[:, -1].div_(ck["duration_scale"])
            folded.head.bias.sub_(w*ck["duration_mean"]/ck["duration_scale"])
        exported = torch.jit.script(folded).eval()
        export_path = root/f'{row["arm"]}-seed{row["seed"]}-standalone.jit.pt'
        torch.jit.save(exported, str(export_path))
        independent = torch.jit.load(str(export_path)).eval()
        raw_predictions = predictions(independent, dev)
        assert np.max(np.abs(raw_predictions-p)) <= 1e-6
    for row in r["unmatched_references"]:
        ck = torch.load(root/f'unmatched-mlp-seed{row["seed"]}.pt', weights_only=True)
        net = nn.Sequential(nn.Linear(12, 16), nn.Tanh(), nn.Linear(16, 1))
        net.load_state_dict(ck["state_dict"])
        ref = {"network": net, "mean": ck["mean"].numpy(), "std": ck["std"].numpy()}
        p = predict_simple(ref, features(dev, ck["item_rates"].numpy()))
        old = np.load(root/f'unmatched-mlp-seed{row["seed"]}-predictions.npz')["probability"]
        assert np.max(np.abs(p-old)) <= 1e-7
        for metric, value in scores(p, dev["y"].numpy()).items():
            assert abs(value-row["dev"][metric]) <= 1e-9
    report = {"status": "PASS", "native_checkpoints": 12, "reference_checkpoints": 3,
              "native_prediction_comparisons": 12*len(dev["y"]),
              "maximum_native_prediction_absolute_error": max(errors),
              "folded_standalone_exports": 12,
              "folded_export_probability_tolerance": 1e-6,
              "source_bounds_and_sampler_rng_match": True,
              "new_fits": 0, "new_test_outcomes": 0}
    (root/"replay.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/me022")
    parser.add_argument("--rebuild-source", action="store_true")
    args = parser.parse_args()
    audit(args.root, args.rebuild_source)
