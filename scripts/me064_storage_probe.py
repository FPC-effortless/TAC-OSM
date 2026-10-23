"""Preservation route probe: commits UNTRAINED toy initialized checkpoint bytes only.

No real observations, optimizer updates, or evidence of predictor quality.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch
from tests.test_me057_training_contract import identified
from scripts.me064_prefit_original_fit import CONFIG, model_for

OUTPUT = Path("research/replay-bundles/ME064")
NAME = "UNTRAINED_STORAGE_PROBE.pt"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_probe():
    torch.set_num_threads(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    models = {}
    for name in CONFIG:
        torch.manual_seed(64001)
        m = model_for(name, identified())
        models[name] = {
            "parameters": sum(p.numel() for p in m.parameters()),
            "weights": m.state_dict(),
        }
    path = OUTPUT / NAME
    torch.save({"is_trained": False, "authoritative_data_included": False,
                "source": "tests.test_me057_training_contract.identified",
                "models": models}, path)
    manifest = {"study": "ME064_REPLAY_STORAGE_TEST_ONLY",
                "artifact": str(path), "checkpoint_is_trained": False,
                "real_fit_opened": False, "optimizer_updates": 0,
                "sha256": digest(path), "bytes": path.stat().st_size,
                "claim": "verifies Git-backed checkpoint byte transport ONLY"}
    (OUTPUT / "STORAGE_PROBE_MANIFEST.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps(manifest, sort_keys=True), flush=True)


def check_probe():
    manifest = json.loads((OUTPUT / "STORAGE_PROBE_MANIFEST.json").read_text())
    path = OUTPUT / NAME
    if digest(path) != manifest["sha256"] or path.stat().st_size != manifest["bytes"]:
        raise RuntimeError("Git checkpoint bytes missing or hash changed")
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data["is_trained"] or data["authoritative_data_included"]:
        raise RuntimeError("storage probe must contain no real training or private data")
    for name, receipt in data["models"].items():
        torch.manual_seed(64001)
        m = model_for(name, identified())
        m.load_state_dict(receipt["weights"], strict=True)
        if receipt["parameters"] != sum(p.numel() for p in m.parameters()):
            raise RuntimeError("wrong model parameter count")
    print("ME064_CLEAN_CHECKOUT_STORAGE_PROBE_PASS", manifest["sha256"], flush=True)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--write"]: write_probe()
    elif sys.argv[1:] == ["--check"]: check_probe()
    else: raise SystemExit("use --write or --check")
