"""Read-only completion audit of the original LANGCAP105 trained artifacts.

No optimizer steps, DEV, TEST, checkpoint selection or capability ranking.
Run against a checkout of the exact original training commit.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys

TRAINING_COMMIT = "ba9220163c777a7d9b6f765378810408baf00b1f"
PROTOCOL_SHA = "51761e3642261a07cc4ea19b24e853154fd4dcfb8dc38a6f1a25de568c063442"
SOURCE_HASHES = {
    "research/langcap105/train_owner.py": "2cb681a9ec694bf4caaa750a695ffe34107fa2beb3a9b8ecb1d351bf7dcc79b0",
    "research/langcap103/preflight.py": "acf2aae721ab002d6ed01cc2eac8e08be3fe030951ab40be10e4e0569d953dc7",
    "research/langcap103/source.py": "1bbb53f5ba292018b899649096814c21601ee1dd1ace89cf28da6bc462805f71",
    "research/langcap103/vendor/models.py": "41b41b9ded06a0e1c6d3aa3f31ffc16508ddd4f2fbb678bfdc1015a878aeca5c",
    "research/langcap103/vendor/native098.py": "e59c0aec80a39d84a6f01a7ca662d0c8baf3e2c2d33901057f017cb23c999498",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def check_receipt(v, protocol, checkpoint):
    require(v["status"] == "FIT_TRAINING_COMPLETED_UNQUALIFIED" and
            v["smoke"] is False and v["actual_steps"] == protocol["fit"]["steps"],
            "Not a complete original full fit")
    require(v["protocol_sha256"] == PROTOCOL_SHA and
            v["source_sha256"] == protocol["source"]["source_sha256"], "Receipt provenance differs")
    require(all(v[k] is False for k in ("dev_scored", "protected_test_access",
            "independently_qualified", "architecture_superiority")), "Scientific label differs")
    require(v["checkpoint_sha256"] == digest(checkpoint), "Final checkpoint digest differs")
    losses = v["fit_loss_series"]
    require(len(losses) == v["actual_steps"] and all(isinstance(x, (int, float)) and
            math.isfinite(x) and x >= 0 for x in losses), "Missing/nonfinite loss series")
    require(losses[0] == v["fit_loss_first"] and losses[-1] == v["fit_loss_last"],
            "Loss endpoints differ")
    require(v["observed_source_bytes"] == v["actual_steps"] * protocol["fit"]["batch_size"] *
            protocol["fit"]["sequence_length"], "Exposure receipt differs")
    require(0 < v["elapsed_seconds"] < protocol["resource_bounds"]["per_job_soft_deadline_minutes"] * 60,
            "Invalid original worker time")
    return losses


def verify_one(path, protocol, data, preflight, owner, torch):
    v = json.loads(path.read_text())
    key = (v["scale"], v["arm"], v["seed"])
    spec = next(m for m in protocol["models"][key[0]]["configurations"] if m["arm"] == key[1])
    checkpoint = path.parent / "TRAINED.pt"
    losses = check_receipt(v, protocol, checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(all(payload[k] == v[k] for k in ("arm", "scale", "seed", "source_sha256", "protocol_sha256"))
            and payload["steps"] == v["actual_steps"], "Checkpoint identity differs")
    require(payload["scientific_status"] == "OWNER_AUTHORIZED_UNQUALIFIED_EXPLORATORY_FIT",
            "Checkpoint qualification label differs")
    torch.manual_seed(key[2])
    config = {k: val for k, val in spec.items() if k != "trainable_parameters"}
    model = preflight.build(config, protocol["models"][key[0]]["target_parameters"],
                            preflight.load_native(), preflight.comparator_classes(torch)).float().eval()
    initial = {name: p.detach().clone() for name, p in model.named_parameters()}
    # Replay the first original training loss without a backward or update.
    length = protocol["fit"]["sequence_length"]
    stream = owner.training_stream(owner.blocks_for(data, length), key[2])
    chosen = [next(stream) for _ in range(protocol["fit"]["batch_size"])]
    raw = [data[name][offset:offset+length+1] for name, offset in chosen]
    x = torch.tensor([list(b[:-1]) for b in raw], dtype=torch.long)
    y = torch.tensor([list(b[1:]) for b in raw], dtype=torch.long)
    model.train()
    with torch.no_grad():
        output = model(x)
        logits = output[0] if isinstance(output, tuple) else output
        replayed_first_loss = float(torch.nn.functional.cross_entropy(logits.reshape(-1, 256), y.reshape(-1)))
    require(math.isclose(replayed_first_loss, losses[0], rel_tol=1e-6, abs_tol=1e-6),
            "Original first loss/source-order/initialization cannot replay")
    model.eval()
    model.load_state_dict(payload["model"], strict=True)
    params = dict(model.named_parameters())
    require(sum(p.numel() for p in params.values()) == spec["trainable_parameters"], "Deployed count differs")
    require(all(p.dtype == torch.float32 and bool(torch.isfinite(p).all()) for p in params.values()),
            "Checkpoint parameter precision/finiteness differs")
    changed = [name for name, p in params.items() if not torch.equal(p, initial[name])]
    require(bool(changed), "Checkpoint equals fresh initialization")
    del initial
    groups = payload["optimizer"]["param_groups"]
    require(len(groups) == 1, "Unexpected optimizer groups")
    group = groups[0]
    require(group["lr"] == protocol["fit"]["lr"] and tuple(group["betas"]) == tuple(protocol["fit"]["betas"])
            and group["eps"] == protocol["fit"]["eps"] and group["weight_decay"] == protocol["fit"]["weight_decay"]
            and group["foreach"] is False, "AdamW configuration differs")
    ids = group["params"]
    states = payload["optimizer"]["state"]
    require(len(ids) == len(params) and len(set(ids)) == len(ids) and set(ids) == set(states),
            "Incomplete AdamW parameter coverage")
    for name, param, pid in zip(params, params.values(), ids):
        slot = states[pid]
        require(set(slot) == {"step", "exp_avg", "exp_avg_sq"}, "Unexpected AdamW state")
        require(slot["step"].numel() == 1 and slot["step"].item() == v["actual_steps"], "AdamW update count differs")
        for moment in ("exp_avg", "exp_avg_sq"):
            require(slot[moment].shape == param.shape and slot[moment].dtype == param.dtype and
                    bool(torch.isfinite(slot[moment]).all()), "Invalid moment tensor for " + name)
        require(bool((slot["exp_avg_sq"] >= 0).all()), "Negative AdamW second moment")
    rng = payload["rng_cpu"]
    require(rng.dtype == torch.uint8 and rng.ndim == 1, "Missing CPU RNG state")
    old_rng = torch.get_rng_state()
    try:
        torch.set_rng_state(rng)
        a = torch.rand(8)
        torch.set_rng_state(rng)
        require(torch.equal(a, torch.rand(8)), "RNG state cannot replay")
    finally:
        torch.set_rng_state(old_rng)
    # Two original FIT prefixes. No source IDs or labels go into the model.
    names = sorted(data)[:2]
    tokens = torch.tensor([list(data[n][:9]) for n in names], dtype=torch.long)
    def predict(x):
        output = model(x)
        return output[0] if isinstance(output, tuple) else output
    before = preflight.parameter_digest(model)
    with torch.no_grad():
        clean = predict(tokens)
        require(clean.shape == (2, 9, 256) and bool(torch.isfinite(clean).all()), "Invalid loaded predictions")
        require(torch.equal(clean, predict(tokens)), "Loaded inference is nondeterministic")
        poison = tokens.clone(); poison[:, -2:] = (poison[:, -2:] + 97) % 256
        require(torch.allclose(clean[:, :-2], predict(poison)[:, :-2], atol=1e-6, rtol=1e-5),
                "Trained prediction changes with future-only bytes")
        single = torch.cat([predict(tokens[i:i+1]) for i in range(2)])
        require(torch.allclose(clean, single, atol=1e-5, rtol=1e-5), "Cross-entity batch interference")
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream); stream.seek(0)
        model.load_state_dict(torch.load(stream, weights_only=True), strict=True)
        require(torch.equal(clean, predict(tokens)), "Model serialization changes actual outputs")
    targets = torch.tensor([list(data[n][1:10]) for n in names], dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(predict(tokens).reshape(-1, 256), targets.reshape(-1))
    loss.backward()
    require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params.values()),
            "Loaded checkpoint has missing/nonfinite gradients")
    require(preflight.parameter_digest(model) == before and digest(checkpoint) == v["checkpoint_sha256"],
            "Read-only verification changed model weights")
    return {"scale":key[0], "arm":key[1], "seed":key[2], "steps":v["actual_steps"],
        "trainable_and_deployed_parameters":spec["trainable_parameters"],
        "checkpoint_sha256":v["checkpoint_sha256"], "receipt_sha256":digest(path),
        "parameter_tensors":len(params), "parameter_tensors_changed_from_seeded_initialization":len(changed),
        "adamw_state_tensors_verified":len(states), "observed_byte_presentations":v["observed_source_bytes"],
        "elapsed_seconds":v["elapsed_seconds"], "fit_loss_first":losses[0], "fit_loss_last":losses[-1],
        "fit_loss_first_128_mean":statistics.mean(losses[:128]), "fit_loss_last_128_mean":statistics.mean(losses[-128:]),
        "original_first_loss_replayed":replayed_first_loss,
        "checkpoint_load_and_prediction_roundtrip":"PASS", "optimizer_moments_steps_and_rng":"PASS",
        "loaded_future_poison_and_entity_isolation":"PASS", "loaded_parameter_gradients":"PASS",
        "optimizer_updates_during_verification":0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--fit-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    require(not args.output.exists(), "Do not overwrite previous audit")
    protocol_path = args.source_root / "research/langcap105/OWNER_PROTOCOL.json"
    require(digest(protocol_path) == PROTOCOL_SHA, "Original frozen protocol changed")
    for path, expected_hash in SOURCE_HASHES.items():
        require(digest(args.source_root / path) == expected_hash, "Original implementation changed: " + path)
    protocol = json.loads(protocol_path.read_text())
    sys.path.insert(0, str(args.source_root / "research/langcap105"))
    import train_owner as owner
    import preflight
    import torch
    require(torch.__version__ == "2.15.0+cpu", "Use original CPU Torch version")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    receipt = next(args.fit_inputs.rglob("SOURCE_RECEIPT.json"))
    archive = next(args.fit_inputs.rglob("FIT_ONLY.tar"))
    data = owner.verify_source(receipt, archive, protocol)
    expected = {(s, m["arm"], seed) for s, g in protocol["models"].items()
                for m in g["configurations"] for seed in protocol["seeds"]}
    reports = []; found = set()
    for path in sorted(args.fits.rglob("FIT_RECEIPT.json")):
        v = json.loads(path.read_text()); key = (v["scale"], v["arm"], v["seed"])
        require(key in expected and key not in found, "Unexpected/duplicate fit")
        reports.append(verify_one(path, protocol, data, preflight, owner, torch)); found.add(key)
        print(json.dumps({"checkpoint":"PASS", "key":key}), flush=True)
    require(bool(found) and (args.allow_partial or found == expected), "Original cohort is incomplete")
    result = {"schema":"langcap105.readonly-trained-checkpoint-audit.v1",
        "status":"ALL_18_TRAINED_CHECKPOINTS_VERIFIED" if found == expected else "PARTIAL_COHORT_VERIFIED",
        "original_training_commit":TRAINING_COMMIT, "original_training_run":41229365510,
        "protocol_sha256":PROTOCOL_SHA, "torch_version":torch.__version__,
        "verified_original_source_sha256":SOURCE_HASHES,
        "fit_files_reverified":len(data), "verified_fit_count":len(found),
        "missing":sorted(expected - found), "models":reports, "optimizer_updates":0,
        "dev_scoring":False, "test_access":False, "independent_scientific_attestation":False,
        "architecture_superiority_eligible":False,
        "limits":["FIT-only pilot; no heldout capability estimate or matched full-FLOPs comparison",
            "Original worker overwrote LATEST.pt then deleted it on success: only final checkpoint retained",
            "RNG/load replay verified; exact full-trajectory optimizer replay not performed",
            "Bounded source-prefix causal probes do not prove absence of every information channel",
            "Original Actions checkpoint artifacts expire after 90 days"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status":result["status"], "verified":len(found), "missing":len(expected - found)}))


if __name__ == "__main__":
    main()
