"""DEV-018 fixed event-budget CPU experiment and exact checkpoint replay."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tac_osm.contract import load_contract  # noqa: E402
from tac_osm.development.delayed_maintenance import Action, PublicObservation  # noqa: E402
from tac_osm.development.memory_baselines import ARMS, MemoryBaseline  # noqa: E402
from tac_osm.development.noisy_memory import parameter_hash  # noqa: E402
from tac_osm.development.schema_maintenance import (  # noqa: E402
    ROLES, DEADLINES, make_pair, execute, exact_public_targets,
)
from run_learned_schema_dev import public_stream_tensors, public_queries, labels  # noqa: E402

SOURCES = ("scripts/run_memory_controls_dev.py", "scripts/run_learned_schema_dev.py",
           "scripts/audit_memory_controls_dev.py", "contracts/DEV-018.json",
           "src/tac_osm/development/memory_baselines.py", "src/tac_osm/development/noisy_memory.py",
           "src/tac_osm/development/schema_maintenance.py", "src/tac_osm/development/delayed_maintenance.py")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def component_hash(model, prefix):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.startswith(prefix):
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train(model, seed, regime, spec, smoke):
    cfg = spec["regimes"][regime]
    steps = 2 if smoke else spec["steps"]
    history = 32 if smoke else cfg["history"]
    pairs = 1 if smoke else cfg["batch_size"] // 2
    before = {p: component_hash(model, p) for p in ("event_encoder.", "memory.")}
    initial_hash = parameter_hash(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=spec["optimizer"]["learning_rate"], weight_decay=.0001)
    gradients = {p: False for p in before}
    training_fps = []
    loss = None
    start = time.monotonic()
    for step in range(steps):
        d = spec["train_due_delays"][(step // 4) % 3]
        deadline = DEADLINES[(step // 12) % 3]
        cases = [c for j in range(pairs) for c in make_pair(
            seed * 1000000 + step * 4 + j, history, d,
            ROLES[(step + j) % 4], deadline, domain="dev018-train")]
        training_fps.extend(c.fingerprint for c in cases)
        inputs = public_stream_tensors([c.observations for c in cases])
        queries = public_queries([c.tick for c in cases], [c.horizon for c in cases])
        target = labels(cases)
        receipts = [(c.world.act(Action("inspect", random.Random(
            f"dev018-train:{seed}:{step}:{b}").randrange(8))),) for b, c in enumerate(cases)]
        obs = public_stream_tensors(receipts)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        state = model.encode(*inputs)
        pre = model.decode(state, queries)
        post = model.decode(model.update(state, *(x[:, 0] for x in obs)), queries)
        loss = (F.cross_entropy(pre.reshape(-1, 2), target.reshape(-1)) +
                F.cross_entropy(post.reshape(-1, 2), target.reshape(-1))) / 2
        loss.backward()
        for prefix in gradients:
            for name, p in model.named_parameters():
                if name.startswith(prefix) and p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        raise AssertionError("nonfinite gradient")
                    gradients[prefix] |= bool(p.grad.abs().sum() > 0)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        optimizer.step()
        if step % 36 == 0:
            print(f"TRAIN seed={seed} regime={regime} arm={model.arm} step={step}/{steps} loss={float(loss.detach()):.6f}", flush=True)
    after = {p: component_hash(model, p) for p in before}
    for prefix in before:
        active = any(p.requires_grad for n, p in model.named_parameters() if n.startswith(prefix))
        if (before[prefix] != after[prefix]) != active or gradients[prefix] != active:
            raise AssertionError("component intervention/gradient mismatch")
    if len(set(training_fps)) != len(training_fps):
        raise AssertionError("duplicate training stream")
    return {"steps": steps, "history": history, "batch_size": pairs * 2,
            "observed_events": steps * history * pairs * 2,
            "training_receipts": steps * pairs * 2, "fingerprints": training_fps,
            "initial_hash": initial_hash, "final_hash": parameter_hash(model),
            "before_components": before, "after_components": after, "gradients": gradients,
            "training_seconds": time.monotonic() - start, "final_loss": float(loss.detach())}


def evaluation_batches(seed, spec, smoke):
    for h in ([32] if smoke else spec["h_levels"]):
        for d in ([32] if smoke else spec["eval_due_delays"]):
            for deadline in (["at"] if smoke else DEADLINES):
                for role in ROLES:
                    cases, conditions = [], []
                    for rep in range(1 if smoke else spec["eval_steps"]):
                        cases.extend(make_pair(seed * 10000 + rep, h, d, role, deadline, domain="dev018-eval"))
                        conditions.extend([dict(history=h, due_delay=d, deadline=deadline,
                                                role=role, repetition=rep, partner=p) for p in (0, 1)])
                    yield conditions, cases


def physical(probabilities, threshold=.25):
    return tuple(e for e, p in enumerate(probabilities) if p[0] - p[1] > threshold)


def predicted_cost(probabilities):
    return sum(min(4 * p[0], 1 + 4 * p[1]) for p in probabilities)


@torch.no_grad()
def sensed(model, state, queries, cases, logits):
    """One chosen inspection: expected reduction in predicted service cost.

    Enumerates both possible PUBLIC receipts; model predictions, not hidden
    evaluator state, weight hypothetical observations and determine selection.
    """
    b = len(cases)
    pre = logits.softmax(-1)[..., 1].tolist()
    streams = [(PublicObservation(c.tick, e, "inspect", bit),)
               for c in cases for e in range(8) for bit in (0, 1)]
    inputs = public_stream_tensors(streams)
    expanded = state.repeat_interleave(16, dim=0)
    proposed = model.update(expanded, *(x[:, 0] for x in inputs))
    hypothetical = model.decode(proposed, queries.repeat_interleave(16, 0)).softmax(-1)[..., 1]
    hypothetical = hypothetical.reshape(b, 8, 2, 8, 3).tolist()
    selected, randoms = [], []
    for i, c in enumerate(cases):
        costs = [[predicted_cost(p) for p in pair] for pair in hypothetical[i]]
        baseline = predicted_cost(pre[i])
        gains = [baseline - ((1 - pre[i][e][2]) * costs[e][0] + pre[i][e][2] * costs[e][1]) - .1
                 for e in range(8)]
        chosen = max(range(8), key=lambda e: (gains[e], -e))
        chosen = chosen if gains[chosen] > 0 else None
        random_entity = random.Random("dev018-sensing:" + c.fingerprint).randrange(8) if chosen is not None else None
        for entity, result in ((chosen, selected), (random_entity, randoms)):
            if entity is None:
                probs = pre[i]
                receipt = None
            else:
                # Actual hidden world consulted ONLY after selecting the action.
                receipt = c.world.act(Action("inspect", entity))
                probs = hypothetical[i][entity][receipt.value]
            services = physical(probs)
            row = execute(c.world, c.horizon, services, int(entity is not None))
            row.update(entity=entity, receipt=asdict(receipt) if receipt else None,
                       decision_probabilities=probs)
            result.append(row)
        selected[-1].update(estimated_gains=gains, hypothetical_costs=costs)
    return selected, randoms


@torch.no_grad()
def evaluate(model, seed, spec, smoke):
    model.eval()
    rows = []
    encode_seconds = decode_seconds = sensing_seconds = 0.
    for conditions, cases in evaluation_batches(seed, spec, smoke):
        inputs = public_stream_tensors([c.observations for c in cases])
        queries = public_queries([c.tick for c in cases], [c.horizon for c in cases])
        start = time.monotonic()
        state = model.encode(*inputs)
        encode_seconds += time.monotonic() - start
        start = time.monotonic()
        logits = model.decode(state, queries)
        reset = model.decode(torch.zeros_like(state), queries).argmax(-1).tolist()
        swapped = model.decode(state.reshape(-1, 2, *state.shape[1:]).flip(1).reshape_as(state), queries).argmax(-1).tolist()
        decode_seconds += time.monotonic() - start
        probabilities = logits.softmax(-1)[..., 1].tolist()
        predictions = logits.argmax(-1).tolist()
        # Physical control and sensing are evaluated on all H512 conditions;
        # smoke uses its H32 conditions. Primary predictions never see receipts.
        control = smoke or conditions[0]["history"] == 512
        selected, randoms = (None, None)
        if control:
            start = time.monotonic()
            selected, randoms = sensed(model, state, queries, cases, logits)
            sensing_seconds += time.monotonic() - start
        for b, c in enumerate(cases):
            public = exact_public_targets(c.observations, c.tick, c.horizon)
            if public != c.wait_targets or len(c.observations) != conditions[b]["history"]:
                raise AssertionError("public solvability/exact length violation")
            row = {"condition": conditions[b], "fingerprint": c.fingerprint, "target_entity": c.target_entity,
                   "wait_targets": list(c.wait_targets), "service_targets": list(c.service_targets),
                   "current_targets": list(c.current_targets), "logits": logits[b].tolist(),
                   "predictions": predictions[b], "reset_predictions": reset[b], "swapped_predictions": swapped[b],
                   "state_scalars": model.state_scalars, "stored_feature_scalars": state[b].numel()}
            if control:
                no = execute(c.world, c.horizon, physical(probabilities[b]), 0)
                no["decision_probabilities"] = probabilities[b]
                row["controls"] = {"none": no, "selected": selected[b], "matched_random": randoms[b],
                    "do_nothing": execute(c.world, c.horizon, (), 0),
                    "always_service": execute(c.world, c.horizon, tuple(range(8)), 0),
                    "public_controller": execute(c.world, c.horizon, tuple(e for e in range(8) if public[e]), 0)}
                row["frontier"] = {str(t): execute(c.world, c.horizon, physical(probabilities[b], t), 0)
                                   for t in spec["frontier_thresholds"]}
            rows.append(row)
    return rows, {"encode_seconds": encode_seconds, "three_decode_seconds": decode_seconds,
                  "sensing_seconds": sensing_seconds}


def run(seed, regime, out, smoke=False, replay=False):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    spec = json.loads((ROOT / "contracts/DEV-018.json").read_text())
    contract = load_contract("DEV-018")
    contract.require_arms(ARMS)
    contract.require_steps(spec["steps"])
    contract.require_levels(spec["h_levels"])
    contract.require_eval_steps(spec["eval_steps"])
    if seed not in spec["seeds"] or regime not in spec["regimes"]:
        raise ValueError("unregistered seed/regime")
    sources = {p: digest(ROOT / p) for p in SOURCES}
    from audit_memory_controls_dev import metrics  # stdlib-only metric implementation
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "report.json"
    if replay:
        report = json.loads(report_path.read_text())
        if (report["source_sha256"] != sources or report["seed"] != seed or
                report["regime"] != regime or report["smoke"] != smoke):
            raise AssertionError("replay provenance changed")
    else:
        report = {"study": "DEV-018", "status": "ADAPTIVE_DEVELOPMENT_NOT_CONFIRMATORY", "seed": seed,
                  "regime": regime, "smoke": smoke, "torch": torch.__version__, "source_sha256": sources,
                  "training": {}, "models": {}, "rows": {}, "summaries": {}}
    for arm in ARMS:
        model = MemoryBaseline(arm, seed + 18000)
        path = out / f"{arm}.pt"
        if replay:
            if digest(path) != report["models"][arm]["sha256"]:
                raise AssertionError("checkpoint digest changed")
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            if parameter_hash(model) != report["training"][arm]["final_hash"]:
                raise AssertionError("tensor hash changed")
        else:
            report["training"][arm] = train(model, seed, regime, spec, smoke)
            torch.save(model.state_dict(), path)
            report["models"][arm] = {"sha256": digest(path), "state_scalars": model.state_scalars,
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
        raw, timing = evaluate(model, seed + 18000, spec, smoke)
        if set(r["fingerprint"] for r in raw) & set(report["training"][arm]["fingerprints"]):
            raise AssertionError("train/eval overlap")
        summary = metrics(raw)
        if replay:
            if raw != report["rows"][arm] or summary != report["summaries"][arm]:
                raise AssertionError("checkpoint prediction/action/cost replay mismatch")
        else:
            report["rows"][arm] = raw
            report["summaries"][arm] = summary
            report["models"][arm]["evaluation_timing"] = timing
            report_path.write_text(json.dumps(report) + "\n")  # preserve completed arms on interruption
        print("SUMMARY " + json.dumps(dict(seed=seed, regime=regime, arm=arm, **summary)), flush=True)
    if not replay:
        compact = {k: report[k] for k in ("study", "status", "seed", "regime", "smoke", "models", "summaries")}
        (out / "summary.json").write_text(json.dumps(compact, indent=2) + "\n")
    print(f"{'REPLAY VERIFIED' if replay else 'TRAINING COMPLETE'} seed={seed} regime={regime} checkpoints={len(ARMS)}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--regime", required=True, choices=("short", "long"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    run(args.seed, args.regime, args.output, args.smoke, args.replay)
