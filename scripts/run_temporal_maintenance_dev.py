"""DEV-016 CPU training, evidence and saved-checkpoint evaluation replay."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tac_osm.contract import load_contract  # noqa: E402
from tac_osm.development.casm_readout_alignment import CASMAlignedModel  # noqa: E402
from tac_osm.development.noisy_memory import parameter_hash  # noqa: E402
from tac_osm.development.delayed_maintenance import make_pair  # noqa: E402
from tac_osm.development.temporal_maintenance_model import (  # noqa: E402
    ARMS, TemporalMaintenanceModel,
)

PARENT_SHA = "880d56e177b78d44961a8fc75e6e0a4151ebe6eeb7a55ce81afaa6c358172d65"
ROLES = ("retention", "acquisition", "cancelled_service", "pending_cancel")
KINDS = ("inspect", "change", "scheduled", "wait")
SOURCE_FILES = (
    "src/tac_osm/development/delayed_maintenance.py",
    "src/tac_osm/development/temporal_maintenance_model.py",
    "scripts/run_temporal_maintenance_dev.py",
    "contracts/DEV-016.json",
    "src/tac_osm/development/casm_readout_alignment.py",
    "src/tac_osm/development/noisy_memory.py",
    "src/tac_osm/development/protected_memory.py",
    "src/tac_osm/development/rehearsal_writer.py",
    "src/tac_osm/development/conditional_compute.py",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parent_model(archive, seed):
    if sha(archive) != PARENT_SHA:
        raise AssertionError("wrong parent archive")
    with zipfile.ZipFile(archive) as z:
        report = json.loads(z.read("report.json"))
        if report["study"] != "DEV-013" or report["seeds"] != [43, 59, 71]:
            raise AssertionError("wrong parent provenance")
        record = next(x for x in report["records"] if x["seed"] == seed)
        data = z.read(f"aligned_aux-seed{seed}.pt")
        if hashlib.sha256(data).hexdigest() != record["models"]["aligned_aux"]["checkpoint_sha256"]:
            raise AssertionError("parent checkpoint modified")
    model = CASMAlignedModel()
    model.load_state_dict(torch.load(io.BytesIO(data), map_location="cpu", weights_only=True), strict=True)
    if parameter_hash(model) != record["training"]["trained_hashes"]["aligned_aux"]:
        raise AssertionError("parent tensor digest mismatch")
    return model.eval()


def public_tensors(observations, entities, delays):
    """No Case/target objects accepted by the policy input encoder."""
    batch = len(observations)
    length = max(len(x) for x in observations)
    events = torch.zeros(batch, length, 14)
    values = torch.zeros(batch, length, dtype=torch.long)
    mask = torch.zeros(batch, length, dtype=torch.bool)
    queries = torch.zeros(batch, 2, 14)
    for b, (stream, entity, delay) in enumerate(zip(observations, entities, delays)):
        for t, obs in enumerate(stream):
            events[b, t, obs.entity] = 1
            events[b, t, 8 + KINDS.index(obs.kind)] = 1
            events[b, t, 12] = obs.tick / 2048
            events[b, t, 13] = max(0, obs.due) / 2048
            values[b, t] = max(0, obs.value)
            mask[b, t] = True
        for candidate, kind in enumerate(("wait", "scheduled")):
            queries[b, candidate, entity] = 1
            queries[b, candidate, 8 + KINDS.index(kind)] = 1
            queries[b, candidate, 12] = stream[-1].tick / 2048
            queries[b, candidate, 13] = (stream[-1].tick + delay) / 2048
    return events, values, mask, queries


def tensors(cases):
    return public_tensors([x.observations for x in cases],
                          [x.query_entity for x in cases],
                          [x.candidate.delay for x in cases])


def matching_models(parent, seed):
    models = {}
    for arm in ARMS:
        torch.manual_seed(seed + 16000)
        models[arm] = TemporalMaintenanceModel(parent, arm)
    hashes = {a: parameter_hash(m) for a, m in models.items()}
    if len(set(hashes.values())) != 1:
        raise AssertionError("unequal initial tensors")
    return models, hashes


def train(parent, seed, smoke, spec):
    models, initial = matching_models(parent, seed)
    optimizers = {a: torch.optim.AdamW((p for p in m.parameters() if p.requires_grad),
                    lr=spec["learning_rate"], weight_decay=.0001) for a, m in models.items()}
    fingerprints = []
    gradients = {a: {"writer": False, "casm": False} for a in ARMS}
    losses = {}
    seconds = {a: 0.0 for a in ARMS}
    steps = 2 if smoke else spec["steps"]
    for step in range(steps):
        history = spec["training_histories"][step % 2]
        delay = spec["training_delays"][(step // 2) % 2]
        cases = []
        # Four pairs per batch; all four roles are represented at each update.
        for j, role in enumerate(ROLES):
            cases.extend(make_pair(seed * 100000 + step * 8 + j, history, delay,
                                   domain="train", role=role))
        fingerprints.extend(x.fingerprint for x in cases)
        inputs = tensors(cases)
        target = torch.tensor([x.expected for x in cases])
        future_target = torch.tensor([[x.expected, x.service_expected] for x in cases])
        g = torch.Generator().manual_seed(seed * 10000 + step)
        shuffled = torch.randint(0, 2, future_target.shape, generator=g)
        for arm, model in models.items():
            start = time.monotonic()
            model.train()
            optimizers[arm].zero_grad(set_to_none=True)
            action, future, predictive_aux = model(*inputs)
            task_loss = F.cross_entropy(action, target)
            readout_loss = F.cross_entropy(future.reshape(-1, 2), future_target.reshape(-1))
            auxiliary = task_loss.new_zeros(())
            if arm in ("joint_predictive", "joint_shuffled"):
                labels = shuffled if arm == "joint_shuffled" else future_target
                auxiliary = F.cross_entropy(predictive_aux.reshape(-1, 2), labels.reshape(-1))
            loss = task_loss + readout_loss + spec["auxiliary_weight"] * auxiliary
            loss.backward()
            for key, p in (("writer", model.reader.memory.write.weight),
                           ("casm", model.reader.compute.selector[-1].weight)):
                gradients[arm][key] |= bool(p.grad is not None and torch.isfinite(p.grad).all()
                                            and p.grad.abs().sum() > 0)
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            optimizers[arm].step()
            losses[arm] = {"task": float(task_loss.detach()), "common_future_readout": float(readout_loss.detach()),
                           "predictive_aux": float(auxiliary.detach())}
            seconds[arm] += time.monotonic() - start
        if step % 32 == 0:
            print(f"seed={seed} step={step}/{steps} losses={losses}", flush=True)
    if len(set(fingerprints)) != steps * spec["batch_size"]:
        raise AssertionError("duplicate or missing training episodes")
    for a in ARMS:
        if not gradients[a]["casm"] or gradients[a]["writer"] != (a != "frozen_writer"):
            raise AssertionError("writer/CASM gradient contract failed")
    return models, {"steps": steps, "initial_hashes": initial,
                    "final_hashes": {a: parameter_hash(m) for a, m in models.items()},
                    "writer_hash_initial": parameter_hash(parent.memory),
                    "writer_hash_final": {a: parameter_hash(m.reader.memory) for a, m in models.items()},
                    "train_fingerprints": fingerprints, "gradients": gradients,
                    "final_losses": losses, "training_seconds": seconds}


def evaluation_cases(seed, smoke, contract):
    histories = [32] if smoke else contract["histories"]
    delays = [4] if smoke else contract["delays"]
    repetitions = 1 if smoke else contract["pairs_per_cell"]
    for h in histories:
        for d in delays:
            for role in ROLES:
                for rep in range(repetitions):
                    cases = make_pair(seed * 10000 + rep, h, d, domain="learned-eval", role=role)
                    yield {"history": h, "delay": d, "role": role, "repetition": rep}, cases


@torch.no_grad()
def evaluate(model, seed, smoke, contract):
    model.eval()
    rows = []
    for condition, cases in evaluation_cases(seed, smoke, contract):
        inputs = tensors(cases)
        state = model.encode(*inputs[:3])
        action, future, _ = model.decode(state, inputs[3])
        calls = model.last_operator_calls
        reset_action, reset_future, _ = model.decode(torch.zeros_like(state), inputs[3])
        swapped_action, swapped_future, _ = model.decode(state.flip(0), inputs[3])
        for b, case in enumerate(cases):
            prediction = int(action.argmax(-1)[b])
            row = {"condition": condition, "partner": b, "fingerprint": case.fingerprint,
                   "target": case.expected, "service_target": case.service_expected,
                   "action": prediction, "future": future.argmax(-1)[b].tolist(),
                   "action_logits": action[b].tolist(), "future_logits": future[b].tolist(),
                   "reset_action": int(reset_action.argmax(-1)[b]),
                   "reset_future": reset_future.argmax(-1)[b].tolist(),
                   "swapped_action": int(swapped_action.argmax(-1)[b]),
                   "swapped_future": swapped_future.argmax(-1)[b].tolist(),
                   "post_action_failure": case.service_expected if prediction else case.expected,
                   "verification_accepted": (int(future.argmax(-1)[b, prediction]) ==
                                              (case.service_expected if prediction else case.expected)),
                   "memory_scalars": state[b].numel(),
                   "observation_count": len(case.observations),
                   "pair_operator_sample_calls": calls}
            rows.append(row)
    return rows


def summarize(rows):
    n = len(rows)
    result = {"episodes": n,
              "action_accuracy": sum(r["action"] == r["target"] for r in rows) / n,
              "wait_future_accuracy": sum(r["future"][0] == r["target"] for r in rows) / n,
              "service_future_accuracy": sum(r["future"][1] == r["service_target"] for r in rows) / n,
              "reset_action_accuracy": sum(r["reset_action"] == r["target"] for r in rows) / n,
              "reset_wait_future_accuracy": sum(r["reset_future"][0] == r["target"] for r in rows) / n,
              "swap_action_accuracy": sum(r["swapped_action"] == r["target"] for r in rows) / n,
              "post_action_failure_rate": sum(r["post_action_failure"] for r in rows) / n,
              "verification_rate": sum(r["verification_accepted"] for r in rows) / n}
    for role in ROLES:
        subset = [r for r in rows if r["condition"]["role"] == role]
        result[role + "_future_accuracy"] = sum(r["future"][0] == r["target"] for r in subset) / len(subset)
        result[role + "_action_accuracy"] = sum(r["action"] == r["target"] for r in subset) / len(subset)
    long = [r for r in rows if r["condition"]["history"] == 512]
    result["H512_future_accuracy"] = (sum(r["future"][0] == r["target"] for r in long) / len(long)
                                      if long else None)
    result["H512_action_accuracy"] = (sum(r["action"] == r["target"] for r in long) / len(long)
                                      if long else None)
    if result["reset_action_accuracy"] != .5 or result["reset_wait_future_accuracy"] != .5:
        raise AssertionError("reset learned model leaks counterfactual identity")
    if any(r["memory_scalars"] != 768 for r in rows):
        raise AssertionError("persistent budget changed")
    return result


def run(archive, output, seed, smoke=False, replay=False):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    contract = json.loads((ROOT / "contracts/DEV-016.json").read_text())
    canonical = load_contract("DEV-016")
    canonical.require_levels(contract["histories"])
    canonical.require_seeds(contract["learned_phase"]["train_seeds"])
    canonical.require_steps(contract["learned_phase"]["steps"])
    canonical.require_arms(ARMS)
    canonical.require_eval_steps(contract["pairs_per_cell"])
    if seed not in canonical.seeds:
        raise AssertionError("unregistered model seed")
    reference = parent_model(archive, seed)
    parent_hash = parameter_hash(reference)
    hashes = {p: sha(ROOT / p) for p in SOURCE_FILES}
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    if replay:
        saved = json.loads(report_path.read_text())
        if (saved["source_sha256"] != hashes or saved["seed"] != seed or saved["smoke"] != smoke
                or saved["parent_tensor_hash"] != parent_hash or saved["parent_archive_sha256"] != PARENT_SHA):
            raise AssertionError("source, seed, mode or parent changed")
        for arm in ARMS:
            info = saved["models"][arm]
            path = output / f"{arm}-seed{seed}.pt"
            if info["checkpoint_sha256"] != sha(path):
                raise AssertionError("checkpoint changed")
            model = TemporalMaintenanceModel(reference, arm)
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
            if parameter_hash(model) != saved["training"]["final_hashes"][arm]:
                raise AssertionError("trained parameter hash mismatch")
            rows = evaluate(model, seed + 16000, smoke, contract)
            if rows != saved["rows"][arm] or summarize(rows) != saved["summaries"][arm]:
                raise AssertionError("evaluation replay changed")
        print(f"REPLAY VERIFIED seed={seed} checkpoints=4 episodes={4 * len(rows)}", flush=True)
        return saved
    models, training = train(reference, seed, smoke, contract["learned_phase"])
    if parameter_hash(reference) != parent_hash:
        raise AssertionError("parent changed")
    if training["writer_hash_final"]["frozen_writer"] != training["writer_hash_initial"]:
        raise AssertionError("frozen writer changed")
    report = {"study": "DEV-016-LEARNED", "status": "DEVELOPMENT_ONLY_NOT_CONFIRMATORY",
              "seed": seed, "smoke": smoke, "parent_tensor_hash": parent_hash,
              "parent_archive_sha256": PARENT_SHA, "source_sha256": hashes,
              "runtime": {"torch": torch.__version__, "threads": 2},
              "training": training, "models": {}, "rows": {}, "summaries": {}}
    for arm, model in models.items():
        path = output / f"{arm}-seed{seed}.pt"
        torch.save(model.state_dict(), path)
        report["models"][arm] = {"checkpoint_sha256": sha(path),
                                 "parameters": sum(p.numel() for p in model.parameters()),
                                 "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
        rows = evaluate(model, seed + 16000, smoke, contract)
        if set(training["train_fingerprints"]) & {r["fingerprint"] for r in rows}:
            raise AssertionError("train/evaluation overlap")
        report["rows"][arm] = rows
        report["summaries"][arm] = summarize(rows)
        print(f"seed={seed} {arm}: {report['summaries'][arm]}", flush=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps({"seed": seed, "smoke": smoke,
        "summaries": report["summaries"], "training_seconds": training["training_seconds"]}, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    run(args.parent, args.output, args.seed, args.smoke, args.replay)
