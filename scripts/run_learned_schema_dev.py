"""DEV-017 matched fresh CPU training, active sensing, physical-cost replay."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tac_osm.contract import load_contract  # noqa: E402
from tac_osm.development.delayed_maintenance import Action  # noqa: E402
from tac_osm.development.learned_schema_state import ARMS, MaintenancePredictor  # noqa: E402
from tac_osm.development.noisy_memory import parameter_hash  # noqa: E402
from tac_osm.development.schema_maintenance import (  # noqa: E402
    ROLES, DEADLINES, FAILURE_COST, SERVICE_COST, INSPECTION_COST,
    make_pair, execute, exact_public_targets,
)

KINDS = ("inspect", "change", "scheduled", "wait")
SOURCES = ("src/tac_osm/development/learned_schema_state.py",
           "src/tac_osm/development/schema_maintenance.py", "scripts/run_learned_schema_dev.py",
           "contracts/DEV-017.json", "src/tac_osm/development/delayed_maintenance.py",
           "src/tac_osm/development/casm_readout_alignment.py",
           "src/tac_osm/development/rehearsal_writer.py", "src/tac_osm/development/protected_memory.py",
           "src/tac_osm/development/noisy_memory.py", "src/tac_osm/development/conditional_compute.py")


def observation_features(observation):
    vector = [0.] * 16
    vector[observation.entity] = 1.
    vector[8 + KINDS.index(observation.kind)] = 1.
    vector[12] = observation.tick / 2048
    vector[13] = max(0, observation.due) / 2048
    value = max(0, observation.value)
    vector[14 + value] = 1.
    return vector, observation.entity, value


def public_stream_tensors(streams):
    """Accept only public observation sequences, no episodes/evaluator targets."""
    if len({len(s) for s in streams}) != 1:
        raise ValueError("batch must have matching exact public lengths")
    encoded = [[observation_features(o) for o in s] for s in streams]
    return (torch.tensor([[x[0] for x in s] for s in encoded], dtype=torch.float32),
            torch.tensor([[x[1] for x in s] for s in encoded]),
            torch.tensor([[x[2] for x in s] for s in encoded]))


def public_queries(ticks, horizons):
    query = torch.zeros(len(ticks), 8, 3, 14)
    for b, (tick, horizon) in enumerate(zip(ticks, horizons)):
        for e in range(8):
            for candidate, kind in enumerate(("wait", "scheduled", "inspect")):
                query[b, e, candidate, e] = 1.
                query[b, e, candidate, 8 + KINDS.index(kind)] = 1.
                query[b, e, candidate, 12] = tick / 2048
                query[b, e, candidate, 13] = (tick + (horizon if candidate < 2 else 0)) / 2048
    return query


def labels(cases):
    return torch.tensor([[[c.wait_targets[e], c.service_targets[e], c.current_targets[e]]
                           for e in range(8)] for c in cases])


def component_hash(model, component):
    prefixes = ("event_encoder.",) if component == "encoder" else (
        ("slot_gru.", "global_gru.") if model.schema else ("reader.memory.",))
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.startswith(prefixes):
            digest.update(name.encode())
            digest.update(bytes(value.detach().cpu().contiguous().view(torch.uint8).flatten().tolist()))
    return digest.hexdigest()


def matching_models(seed):
    models = {}
    for arm in ARMS:
        torch.manual_seed(seed + 17000)
        models[arm] = MaintenancePredictor(arm)
    hashes = {a: parameter_hash(m) for a, m in models.items()}
    for group in (ARMS[:2], ARMS[2:]):
        if len({hashes[a] for a in group}) != 1:
            raise AssertionError("unequal initialization within causal-control group")
    return models, hashes


def train(seed, spec, smoke):
    models, initial = matching_models(seed)
    optimizers = {a: torch.optim.AdamW((p for p in m.parameters() if p.requires_grad),
                      lr=spec["optimizer"]["learning_rate"], weight_decay=spec["optimizer"]["weight_decay"])
                  for a, m in models.items()}
    initial_components = {a: {p: component_hash(m, p) for p in ("memory", "encoder")}
                          for a, m in models.items()}
    gradients = {a: {"memory": False, "encoder": False} for a in ARMS}
    seconds = {a: 0. for a in ARMS}
    fingerprints = []
    sensing_receipt_hashes = []
    steps = 2 if smoke else spec["steps"]
    for step in range(steps):
        h = spec["train_histories"][step % 2]
        d = spec["train_due_delays"][(step // 2) % 3]
        deadline = DEADLINES[(step // 6) % 3]
        cases = [case for j, role in enumerate(ROLES) for case in
                 make_pair(seed * 100000 + step * 8 + j, h, d, role, deadline, domain="train")]
        inputs = public_stream_tensors([c.observations for c in cases])
        queries = public_queries([c.tick for c in cases], [c.horizon for c in cases])
        target = labels(cases)
        fingerprints.extend(c.fingerprint for c in cases)
        # Shared random training inspections teach receipt transitions without
        # giving different arms different training evidence or hidden targets.
        inspection_batches = [[], []]
        for b, c in enumerate(cases):
            entities = random.Random(f"dev017-train-inspect:{seed}:{step}:{b}").sample(range(8), 2)
            observations = [c.world.act(Action("inspect", e)) for e in entities]
            for t, observation in enumerate(observations):
                inspection_batches[t].append((observation,))
            sensing_receipt_hashes.append(hashlib.sha256(json.dumps(
                [asdict(o) for o in observations], sort_keys=True).encode()).hexdigest())
        inspection_inputs = [public_stream_tensors(s) for s in inspection_batches]
        losses = {}
        for arm, model in models.items():
            start = time.monotonic()
            model.train()
            optimizers[arm].zero_grad(set_to_none=True)
            state = model.encode(*inputs)
            logits = model.decode(state, queries)
            pre_loss = F.cross_entropy(logits.reshape(-1, 2), target.reshape(-1))
            for features, entities, values in inspection_inputs:
                state = model.update(state, features[:, 0], entities[:, 0], values[:, 0])
            post_logits = model.decode(state, queries)
            post_loss = F.cross_entropy(post_logits.reshape(-1, 2), target.reshape(-1))
            loss = (pre_loss + post_loss) / 2
            loss.backward()
            for part, prefixes in (("encoder", ("event_encoder.",)),
                                   ("memory", ("slot_gru.", "global_gru.") if model.schema else ("reader.memory.",))):
                for name, p in model.named_parameters():
                    if name.startswith(prefixes) and p.grad is not None:
                        if not torch.isfinite(p.grad).all():
                            raise AssertionError("nonfinite training gradient")
                        gradients[arm][part] |= bool(p.grad.abs().sum() > 0)
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
                                          spec["optimizer"]["clip_norm"])
            optimizers[arm].step()
            seconds[arm] += time.monotonic() - start
            losses[arm] = float(loss.detach())
        if step % 32 == 0:
            print(f"seed={seed} step={step}/{steps} losses={losses}", flush=True)
    final_components = {a: {p: component_hash(m, p) for p in ("memory", "encoder")}
                        for a, m in models.items()}
    expected_memory = {"schema41", "tac_memory", "tac_joint"}
    expected_encoder = {"schema41", "schema41_encoder", "tac_encoder", "tac_joint"}
    for a in ARMS:
        for part, expected in (("memory", expected_memory), ("encoder", expected_encoder)):
            if gradients[a][part] != (a in expected):
                raise AssertionError("causal-control gradient mismatch")
            changed = initial_components[a][part] != final_components[a][part]
            if changed != (a in expected):
                raise AssertionError("causal-control component changed incorrectly")
    if len(set(fingerprints)) != steps * spec["batch_size"]:
        raise AssertionError("training observations duplicated or incomplete")
    return models, {"steps": steps, "initial_hashes": initial,
                    "final_hashes": {a: parameter_hash(m) for a, m in models.items()},
                    "initial_components": initial_components, "final_components": final_components,
                    "gradients": gradients, "training_seconds": seconds,
                    "train_fingerprints": fingerprints, "training_sensing_receipt_sha256": sensing_receipt_hashes,
                    "final_losses": losses}


def evaluation_batches(seed, spec, smoke):
    for h in ([32] if smoke else spec["h_levels"]):
        for d in ([16] if smoke else spec["eval_due_delays"]):
            for deadline in (["at"] if smoke else DEADLINES):
                for role in ROLES:
                    cases = []
                    conditions = []
                    for rep in range(1 if smoke else spec["eval_steps"]):
                        cases.extend(make_pair(seed * 10000 + rep, h, d, role, deadline))
                        conditions.extend([{"history": h, "due_delay": d, "deadline": deadline,
                                            "role": role, "repetition": rep, "partner": partner}
                                           for partner in (0, 1)])
                    yield conditions, cases


def services_from_probabilities(probabilities):
    return tuple(e for e in range(8) if FAILURE_COST * probabilities[e][0] >
                 SERVICE_COST + FAILURE_COST * probabilities[e][1])


def select_entity(probabilities, inspected):
    candidates = [(FAILURE_COST * p[0] * (1 - p[0]), e)
                  for e, p in enumerate(probabilities) if e not in inspected]
    if not candidates:
        return None
    score, entity = max(candidates, key=lambda x: (x[0], -x[1]))
    return entity if score > INSPECTION_COST else None


@torch.no_grad()
def closed_loop(model, state, queries, cases, mode, selected_counts=None):
    state = state.clone()
    worlds = [copy.deepcopy(c.world) for c in cases]
    inspected = [[] for _ in cases]
    receipts = [[] for _ in cases]
    traces = [[] for _ in cases]
    random_orders = []
    for c in cases:
        order = list(range(8))
        random.Random("dev017-random-sensing:" + c.fingerprint).shuffle(order)
        random_orders.append(order)
    for sensing_step in range(2 if mode != "none" else 0):
        probs = model.decode(state, queries).softmax(-1)[..., 1].tolist() if mode == "selected" else None
        observations, active = [], []
        for b, world in enumerate(worlds):
            if mode == "selected":
                entity = select_entity(probs[b], inspected[b])
                traces[b].append({"wait_probabilities": [p[0] for p in probs[b]],
                                  "already_inspected": inspected[b].copy(), "chosen_entity": entity})
            elif mode == "matched_random":
                entity = random_orders[b][sensing_step] if sensing_step < selected_counts[b] else None
            else:
                raise ValueError("unknown sensing mode")
            active.append(entity is not None)
            if entity is None:
                # Padding is a real public wait token but its state update is masked.
                obs = world.act(Action("wait", 0))
            else:
                inspected[b].append(entity)
                obs = world.act(Action("inspect", entity))
                receipts[b].append(asdict(obs))
            observations.append((obs,))
        features, entities, values = public_stream_tensors(observations)
        proposal = model.update(state, features[:, 0], entities[:, 0], values[:, 0])
        mask = torch.tensor(active).reshape(-1, *([1] * (state.ndim - 1)))
        state = torch.where(mask, proposal, state)
    logits = model.decode(state, queries)
    probs = logits.softmax(-1)[..., 1].tolist()
    results = []
    for b, c in enumerate(cases):
        services = services_from_probabilities(probs[b])
        outcome = execute(worlds[b], c.horizon, services, len(inspected[b]))
        # Verification only after actions; never used for forecast/action repair.
        predicted = [int(logits[b, e, 1 if e in services else 0].argmax()) for e in range(8)]
        outcome.update(inspected_entities=inspected[b], inspection_receipts=receipts[b],
                       sensing_trace=traces[b], decision_probabilities=probs[b],
                       decision_logits=logits[b].tolist(), committed_forecast=predicted,
                       verification=[x == y for x, y in zip(predicted, outcome["outcomes"])])
        results.append(outcome)
    return results


@torch.no_grad()
def evaluate(model, seed, spec, smoke):
    model.eval()
    rows = []
    for conditions, cases in evaluation_batches(seed, spec, smoke):
        inputs = public_stream_tensors([c.observations for c in cases])
        queries = public_queries([c.tick for c in cases], [c.horizon for c in cases])
        state = model.encode(*inputs)
        logits = model.decode(state, queries)
        calls = model.last_calls_per_episode.copy()
        reset = model.decode(torch.zeros_like(state), queries).argmax(-1).tolist()
        swapped_state = state.reshape(-1, 2, *state.shape[1:]).flip(1).reshape_as(state)
        swapped = model.decode(swapped_state, queries).argmax(-1).tolist()
        none = closed_loop(model, state, queries, cases, "none")
        selected = closed_loop(model, state, queries, cases, "selected")
        matched = closed_loop(model, state, queries, cases, "matched_random",
                              [r["inspections"] for r in selected])
        predicted = logits.argmax(-1).tolist()
        for b, c in enumerate(cases):
            if len(c.observations) != conditions[b]["history"]:
                raise AssertionError("observation length differs from declared H")
            if exact_public_targets(c.observations, c.tick, c.horizon) != c.wait_targets:
                raise AssertionError("public reference and hidden dynamics disagree")
            rows.append({"condition": conditions[b], "fingerprint": c.fingerprint,
                         "target_entity": c.target_entity, "wait_targets": list(c.wait_targets),
                         "service_targets": list(c.service_targets), "current_targets": list(c.current_targets),
                         "logits": logits[b].tolist(), "predictions": predicted[b],
                         "reset_predictions": reset[b], "swapped_predictions": swapped[b],
                         "pre_sensing_observations": len(c.observations), "state_scalars": state[b].numel(),
                         "pre_sensing_operator_sample_calls": calls[b],
                         "controls": {"none": none[b], "selected": selected[b], "matched_random": matched[b]}})
    return rows


def summarize(rows):
    mature = [r for r in rows if r["condition"]["deadline"] != "before"]
    correct = lambda r, field: r[field][r["target_entity"]][0] == r["wait_targets"][r["target_entity"]]
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    primary = mean([correct(r, "predictions") for r in mature])
    reset = mean([correct(r, "reset_predictions") for r in mature])
    if reset != .5:
        raise AssertionError("balanced mature-pair reset leakage gate failed")
    long = [r for r in mature if r["condition"]["history"] == 512 and
            r["condition"]["due_delay"] >= 64 and "cancel" in r["condition"]["role"]]
    h512 = [r for r in rows if r["condition"]["history"] == 512]
    out = {"episodes": len(rows), "primary_future_accuracy": primary,
           "reset_primary_accuracy": reset,
           "swap_primary_accuracy": mean([correct(r, "swapped_predictions") for r in mature]),
           "H512_long_pending_accuracy": mean([correct(r, "predictions") for r in long]),
           "before_due_accuracy": mean([correct(r, "predictions") for r in rows
                                        if r["condition"]["deadline"] == "before"])}
    for role in ROLES:
        out[role + "_accuracy"] = mean([correct(r, "predictions") for r in mature
                                        if r["condition"]["role"] == role])
    for mode in ("none", "selected", "matched_random"):
        for field, divisor in (("total_cost", 8), ("failures", 8), ("unnecessary_services", 8),
                               ("inspections", 1), ("services", 8)):
            numeric = lambda r: len(r["controls"][mode][field]) if field == "services" else r["controls"][mode][field]
            out[mode + "_" + field] = mean([numeric(r) / divisor for r in rows])
            out["H512_" + mode + "_" + field] = mean([numeric(r) / divisor for r in h512])
    return out


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(seed, out, smoke=False, replay=False):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    spec = json.loads((ROOT / "contracts/DEV-017.json").read_text())
    contract = load_contract("DEV-017")
    contract.require_arms(ARMS)
    contract.require_steps(spec["steps"])
    contract.require_levels(spec["h_levels"])
    contract.require_eval_steps(spec["eval_steps"])
    if seed not in contract.seeds or spec["physical_costs"] != {
            "failure": FAILURE_COST, "service": SERVICE_COST, "inspection": INSPECTION_COST}:
        raise AssertionError("seed/cost contract mismatch")
    hashes = {p: sha(ROOT / p) for p in SOURCES}
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "report.json"
    if replay:
        report = json.loads(report_path.read_text())
        if (report["source_sha256"] != hashes or report["seed"] != seed or report["smoke"] != smoke or
                report["study"] != "DEV-017" or report["status"] != "DEVELOPMENT_ONLY_NOT_CONFIRMATORY"):
            raise AssertionError("source/mode/provenance changed")
        for arm in ARMS:
            path = out / f"{arm}-seed{seed}.pt"
            if sha(path) != report["models"][arm]["sha256"]:
                raise AssertionError("checkpoint changed")
            model = MaintenancePredictor(arm)
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
            if parameter_hash(model) != report["training"]["final_hashes"][arm]:
                raise AssertionError("model tensor hash changed")
            raw = evaluate(model, seed + 17000, spec, smoke)
            if raw != report["rows"][arm] or summarize(raw) != report["summaries"][arm]:
                raise AssertionError("complete forecast/cost/sensing replay failed")
        print(f"REPLAY VERIFIED seed={seed} checkpoints=6 episodes={len(raw) * 6}", flush=True)
        return report
    models, training = train(seed, spec, smoke)
    report = {"study": "DEV-017", "status": "DEVELOPMENT_ONLY_NOT_CONFIRMATORY", "seed": seed,
              "smoke": smoke, "source_sha256": hashes, "torch": torch.__version__,
              "training": training, "models": {}, "rows": {}, "summaries": {}}
    public = None
    for arm, model in models.items():
        path = out / f"{arm}-seed{seed}.pt"
        torch.save(model.state_dict(), path)
        report["models"][arm] = {"sha256": sha(path), "state_scalars": model.state_scalars,
                                 "parameters": sum(p.numel() for p in model.parameters()),
                                 "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
        raw = evaluate(model, seed + 17000, spec, smoke)
        fps = [r["fingerprint"] for r in raw]
        if public is not None and public != fps:
            raise AssertionError("unmatched evaluation observations")
        public = fps
        if set(fps) & set(training["train_fingerprints"]):
            raise AssertionError("train/evaluation overlap")
        report["rows"][arm] = raw
        report["summaries"][arm] = summarize(raw)
        print(f"seed={seed} {arm}: {report['summaries'][arm]}", flush=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps({"seed": seed, "smoke": smoke,
        "summaries": report["summaries"], "models": report["models"]}, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    run(args.seed, args.output, args.smoke, args.replay)
