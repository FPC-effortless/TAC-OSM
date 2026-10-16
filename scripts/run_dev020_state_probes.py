"""DEV-020: frozen DEV-018 state/readout/logit forensics, NOT confirmatory.

Only post-hoc probe heads train, on a disjoint synthetic public-history domain.
Original memory, encoders, reader and head are frozen; targets are probe labels
and never passed to MemoryBaseline.encode/decode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import torch  # noqa: E402
from torch import nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tac_osm.development.memory_baselines import MemoryBaseline  # noqa: E402
from tac_osm.development.schema_maintenance import make_pair  # noqa: E402
from tac_osm.development.noisy_memory import parameter_hash  # noqa: E402
from run_learned_schema_dev import public_queries, public_stream_tensors  # noqa: E402

ARMS = ("memoryless", "gru768", "schema41", "tac_joint", "tac_encoder")
PREFIXES = (32, 128, 512)
DUE = (32, 128, 512)
DEADLINES = ("at", "after")
ROLES = ("pending_cancel", "reissued_cancel")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def episodes(seed: int, domain: str, repetitions: int):
    """Balanced, consecutively paired outcomes with all target entities public."""
    result = []
    for delay in DUE:
        for deadline in DEADLINES:
            for role in ROLES:
                for rep in range(repetitions):
                    if domain == "dev018-eval":
                        episode_seed = (seed + 18000) * 10000 + rep
                    else:
                        episode_seed = 200000000 + seed * 100000 + rep
                    pair = make_pair(episode_seed, 512, delay, role, deadline, domain=domain)
                    assert pair[0].wait_targets[pair[0].target_entity] != pair[1].wait_targets[pair[1].target_entity]
                    assert pair[0].target_entity == pair[1].target_entity
                    assert pair[0].observations[-1] == pair[1].observations[-1]
                    result.extend(pair)
    assert len(result) == 24 * repetitions
    return result


def labels(cases):
    y = torch.tensor([c.wait_targets[c.target_entity] for c in cases], dtype=torch.float32)
    assert float(y.mean()) == .5
    return y


def pair_metrics(y: torch.Tensor, scores: torch.Tensor) -> dict:
    """The label in each opposite-future pair differs; report paired decisions."""
    assert len(y) == len(scores) and len(y) % 2 == 0
    assert torch.equal(y.reshape(-1, 2).sum(1), torch.ones(len(y) // 2))
    # Match PyTorch argmax on equal binary probabilities: ties select class zero.
    prediction = (scores > .5).to(y.dtype)
    grouped = prediction.reshape(-1, 2)
    return {"accuracy": float((prediction == y).float().mean()),
            "paired_class_disagreement": float((grouped[:, 0] != grouped[:, 1]).float().mean()),
            "paired_mean_probability_gap": float((scores.reshape(-1, 2)[:, 0]
                                                - scores.reshape(-1, 2)[:, 1]).abs().mean())}


def target_state_features(model, state, queries, targets):
    """An explicitly entity-conditioned *probe* interface; never model inference."""
    b = len(targets)
    q = queries[torch.arange(b), targets, 0]
    if model.arm.startswith("schema"):
        slots = state[:, :40].reshape(b, 8, 5)
        selected = slots[torch.arange(b), targets]
        return torch.cat((selected, state[:, 40:], q), -1)
    if model.arm == "memoryless":
        return torch.cat((state, q), -1)
    flat = state.reshape(b, -1)
    onehot = F.one_hot(targets, num_classes=8).to(flat.dtype)
    return torch.cat(((flat[:, None] * onehot[:, :, None]).flatten(1), q), -1)


@torch.no_grad()
def features(model, cases, prefix):
    """Trace model states, exact target-query head inputs and logits."""
    latents, state_inputs, reads, probabilities = [], [], [], []
    for start in range(0, len(cases), 8):
        # DEV-018 evaluated four opposite-outcome pairs in batches of eight.
        # Preserve that shape to avoid tiny batch-dependent changes near argmax.
        subset = cases[start:start + 8]
        original_len = subset[0].tick
        assert original_len == 512
        stream = [c.observations[:prefix] for c in subset]
        events = public_stream_tensors(stream)
        qs = public_queries([prefix] * len(subset),
                            [c.horizon + (c.tick - prefix) for c in subset])
        target = torch.tensor([c.target_entity for c in subset], dtype=torch.long)
        state = model.encode(*events)
        captured = []
        hook = model.head.register_forward_pre_hook(
            lambda _module, args: captured.append(args[0].detach().clone()))
        try:
            logits = model.decode(state, qs)
        finally:
            hook.remove()
        assert len(captured) == 1
        inp = captured[0].reshape(len(subset), 8, 3, 44)
        index = torch.arange(len(subset))
        selected = inp[index, target, 0]
        read = selected  # read32 + encoded public query12, as received by the original head
        prob = logits.softmax(-1)[index, target, 0, 1]
        latents.append(state.flatten(1))
        state_inputs.append(target_state_features(model, state, qs, target))
        reads.append(read)
        probabilities.append(prob)
    return {"latent": torch.cat(latents), "state_probe": torch.cat(state_inputs),
            "read_probe": torch.cat(reads), "p1": torch.cat(probabilities)}


def l2_pair_distance(x: torch.Tensor) -> float:
    pair = x.reshape(-1, 2, x.shape[1])
    return float(torch.linalg.vector_norm(pair[:, 0] - pair[:, 1], dim=-1).mean())


def train_linear_probe(train_x, train_y, test_x, seed, shuffled=False, steps=160):
    """Preselected fixed-capacity diagnostic, no checkpoint or test-label updates."""
    mean, scale = train_x.mean(0), train_x.std(0, unbiased=False).clamp_min(.05)
    x = ((train_x - mean) / scale).clamp(-20, 20)
    xt = ((test_x - mean) / scale).clamp(-20, 20)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        classifier = nn.Linear(train_x.shape[1], 1)
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=.015, weight_decay=.05)
        fit_y = train_y.clone()
        if shuffled:
            fit_y = fit_y[torch.randperm(len(fit_y))]
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(classifier(x).flatten(), fit_y)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            result = classifier(xt).flatten().sigmoid()
    return result.detach()


def original_primary_rows(report, arm):
    return [r for r in report["rows"][arm] if
            r["condition"]["history"] == 512 and
            r["condition"]["role"] in ROLES and
            r["condition"]["deadline"] in DEADLINES]


def run(seed, regime, output: Path, frozen: Path | None, smoke=False):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if seed not in (43, 59, 71) or regime not in ("short", "long"):
        raise ValueError("unregistered seed/regime")
    report = None
    if not smoke:
        if frozen is None:
            raise ValueError("full run needs archived DEV-018 checkpoints")
        report = json.loads((frozen / "report.json").read_text())
        assert report["study"] == "DEV-018" and report["seed"] == seed
        assert report["regime"] == regime and not report["smoke"]
        assert report["status"] == "ADAPTIVE_DEVELOPMENT_NOT_CONFIRMATORY"
    test = episodes(seed, "dev018-eval", 4 if not smoke else 1)
    train = episodes(seed, "dev020-probe-train", 12 if not smoke else 2)
    assert not {c.fingerprint for c in train} & {c.fingerprint for c in test}
    yt, yv = labels(train), labels(test)
    result = {"study": "DEV-020", "status": "POSTHOC_FROZEN_DIAGNOSTIC_NOT_CONFIRMATORY",
              "seed": seed, "regime": regime, "smoke": smoke,
              "train_episodes": len(train), "heldout_episodes": len(test),
              "training_domain": "dev020-probe-train", "test_domain": "dev018-eval",
              "probe_only": True, "arms": {}}
    for arm in ARMS:
        model = MemoryBaseline(arm, seed + 18000).eval()
        frozen_sha = None
        if report is not None:
            path = frozen / (arm + ".pt")
            frozen_sha = sha256(path)
            assert frozen_sha == report["models"][arm]["sha256"]
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            assert parameter_hash(model) == report["training"][arm]["final_hash"]
        for p in model.parameters():
            p.requires_grad_(False)
        result["arms"][arm] = {"checkpoint_sha256": frozen_sha, "prefixes": {}}
        for h in PREFIXES:
            test_features = features(model, test, h)
            scores = test_features["p1"]
            if h == 512 and report is not None:
                original = original_primary_rows(report, arm)
                assert len(original) == len(test)
                archived = []
                for i, row in enumerate(original):
                    assert row["fingerprint"] == test[i].fingerprint
                    assert row["wait_targets"][row["target_entity"]] == int(yv[i])
                    logits = torch.tensor(row["logits"][row["target_entity"]][0])
                    expected = float(logits.softmax(-1)[1])
                    assert abs(expected - float(scores[i])) < 2e-5, (
                        f"frozen probability drift: {arm} {seed} {regime} {i}")
                    archived.append(expected)
                # Registered archived predictions are the frozen authority:
                # near-.5 float drift in batched recomputation cannot rewrite
                # DEV-018's endpoint. Features still come from the frozen model.
                scores = torch.tensor(archived)
                frozen_predictions = torch.tensor([
                    r["predictions"][r["target_entity"]][0] for r in original])
                recomputed = (scores > .5).long()
                assert torch.equal(recomputed, frozen_predictions), (
                    f"original logits/decisions disagree for {arm}")
            y = pair_metrics(yv, scores)
            y.update(latent_pair_l2=l2_pair_distance(test_features["latent"]),
                     target_read_pair_l2=l2_pair_distance(test_features["read_probe"]))
            result["arms"][arm]["prefixes"][str(h)] = y
            if h != 512:
                continue
            if report is not None:
                assert abs(y["accuracy"] - report["summaries"][arm]["primary"]) < 1e-9
            training_features = features(model, train, 512)
            for level in ("state_probe", "read_probe"):
                predicted = train_linear_probe(training_features[level], yt,
                                               test_features[level], 20200 + seed, steps=5 if smoke else 160)
                control = train_linear_probe(training_features[level], yt,
                                             test_features[level], 40200 + seed,
                                             shuffled=True, steps=5 if smoke else 160)
                result["arms"][arm][level] = {
                    "heldout": pair_metrics(yv, predicted),
                    "shuffled_label_control": pair_metrics(yv, control),
                    "linear_features": int(training_features[level].shape[1])}
        print("DEV020_ARM " + json.dumps({"seed": seed, "regime": regime, "arm": arm,
                                         **result["arms"][arm]}), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print("DEV020_COMPLETE " + json.dumps({"seed": seed, "regime": regime,
                                             "smoke": smoke, "cases": len(test),
                                             "arms": list(result["arms"])}), flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--regime", choices=("short", "long"), required=True)
    p.add_argument("--frozen", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    run(args.seed, args.regime, args.output, args.frozen, args.smoke)
