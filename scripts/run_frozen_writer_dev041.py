"""DEV-041 matched frozen-writer development study. No confirmatory or test access."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import torch
import torch.nn.functional as F
from tac_osm.development.bounded_mtsk import build_bounded
from tac_osm.development.memory_baselines import MemoryBaseline
from tac_osm.development.finite_clipping import clip_finite_cpu_gradients_
from tac_osm.development.schema_maintenance import make_pair, ROLES, DEADLINES, exact_public_targets
from tac_osm.development.delayed_maintenance import Action
from run_learned_schema_dev import public_stream_tensors, public_queries, labels
from diagnose_head_cost_dev040a import last_keys

ARMS = ("bounded_trainable", "bounded_random_frozen", "memoryless")
THRESHOLDS = (0., .05, .1, .15, .2, .25, .3, .4, .5, .6, .7, .75, .8, .9, 1.)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(arm, seed):
    if arm in ARMS[:2]:
        model = build_bounded(seed + 40000)
        if arm == "bounded_random_frozen":
            for p in model.memory.parameters():
                p.requires_grad_(False)
    elif arm == "memoryless":
        model = MemoryBaseline("memoryless", seed + 40000)
    else:
        raise ValueError("unknown registered arm")
    return model


def compare_initialization(trainable, frozen):
    a, b = trainable.state_dict(), frozen.state_dict()
    if a.keys() != b.keys() or any(not torch.equal(a[k], b[k]) for k in a):
        raise AssertionError("trainable and frozen-writer arms must begin identically")
    assert all(not p.requires_grad for p in frozen.memory.parameters())
    assert any(p.requires_grad for p in trainable.memory.parameters())


def cases_for(split, repetitions, history=512):
    for delay in (32, 128, 512):
        for role in ROLES:
            for deadline in DEADLINES:
                for rep in range(repetitions):
                    worlds = make_pair(410000 + rep, history, delay, role, deadline,
                                       domain="dev041-" + split + "-v1")
                    yield worlds, [dict(delay=delay, role=role, deadline=deadline,
                        repetition=rep, partner=b, fingerprint=c.fingerprint,
                        target_entity=c.target_entity) for b, c in enumerate(worlds)]


def train(model, arm, seed, count, output):
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=.002, weight_decay=.0001)
    fps, curve = [], []
    start = time.monotonic()
    try:
        for step in range(count):
            worlds = make_pair(seed * 100000 + step, 32 if count == 2 else 512,
                               (32, 128, 512)[(step // 4) % 3], ROLES[step % 4],
                               DEADLINES[(step // 12) % 3], domain="dev041-backbone-train-v1")
            fps.extend(c.fingerprint for c in worlds)
            ev = public_stream_tensors([c.observations for c in worlds])
            queries = public_queries([c.tick for c in worlds], [c.horizon for c in worlds])
            targets = labels(worlds)
            receipts = [(c.world.act(Action("inspect", random.Random(
                "dev041:%s:%s:%s" % (seed, step, b)).randrange(8))),)
                for b, c in enumerate(worlds)]
            receipt = public_stream_tensors(receipts)
            opt.zero_grad(set_to_none=True)
            state = model.encode(*ev)
            before = model.decode(state, queries)
            after = model.decode(model.update(state, *(v[:, 0] for v in receipt)), queries)
            loss = (F.cross_entropy(before.reshape(-1, 2), targets.reshape(-1))
                    + F.cross_entropy(after.reshape(-1, 2), targets.reshape(-1))) / 2
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite backbone loss")
            loss.backward()
            norm = clip_finite_cpu_gradients_(model.parameters(), 1.)
            opt.step()
            if step in (0, count - 1) or (step + 1) % 36 == 0:
                curve.append(dict(update=step + 1, loss=float(loss.detach()), gradnorm=norm))
                print(json.dumps(dict(arm=arm, seed=seed, **curve[-1])), flush=True)
    except Exception as exc:
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(),
                        rng=torch.get_rng_state(), attempted=step + 1),
                   output / (arm + "-failure.pt"))
        (output / (arm + "-failure.json")).write_text(json.dumps(
            dict(arm=arm, seed=seed, error=str(exc), attempted=step + 1)) + "\n")
        raise
    return dict(updates=count, fingerprints=fps, curve=curve,
                training_seconds=time.monotonic() - start)


@torch.no_grad()
def predict(model, split, repetitions, history=512):
    model.eval()
    rows = []
    for worlds, conditions in cases_for(split, repetitions, history):
        ev = public_stream_tensors([c.observations for c in worlds])
        query = public_queries([c.tick for c in worlds], [c.horizon for c in worlds])
        logits = model.decode(model.encode(*ev), query)
        probabilities = logits.softmax(-1)[..., 1].tolist()
        for i, (case, condition) in enumerate(zip(worlds, conditions)):
            if exact_public_targets(case.observations, case.tick, case.horizon) != case.wait_targets:
                raise AssertionError("public-history target mismatch")
            rows.append(dict(**condition, probabilities=probabilities[i],
                             labels=labels([case])[0].tolist()))
    return rows


def cost(row, threshold):
    p = row["probabilities"]
    y = row["labels"]
    chosen = tuple(e for e in range(8) if p[e][0] - p[e][1] > threshold)
    failures = sum(y[e][int(e in chosen)] for e in range(8))
    unnecessary = sum(e in chosen and y[e][0] == 0 for e in range(8))
    return dict(failures=int(failures), unnecessary_services=int(unnecessary),
                services=len(chosen), inspections=0, total_cost=4 * failures + len(chosen))


def choose_threshold(rows):
    totals = {str(t): sum(cost(row, t)["total_cost"] for row in rows) for t in THRESHOLDS}
    best = min(THRESHOLDS, key=lambda t: (totals[str(t)], abs(t - .25), t))
    return best, totals


def last_event_fit(rows, split, reps):
    counts = {}
    for worlds, conditions in cases_for(split, reps):
        for case, condition in zip(worlds, conditions):
            matching = next(r for r in rows if r["fingerprint"] == condition["fingerprint"])
            for key, target in zip(last_keys(case), matching["labels"]):
                entry = counts.setdefault(key, [torch.zeros(3), 0])
                entry[0] += torch.tensor(target, dtype=torch.float32)
                entry[1] += 1
    return {k: (v / n).tolist() for k, (v, n) in counts.items()}


def last_event_rows(fit, original, split, reps):
    rows = []
    lookup = {r["fingerprint"]: r for r in original}
    for worlds, _ in cases_for(split, reps):
        for case in worlds:
            row = lookup[case.fingerprint]
            probs = [fit.get(key, [0., 0., 0.]) for key in last_keys(case)]
            rows.append({**row, "probabilities": probs})
    return rows


def summarize(rows, threshold):
    totals = dict(failures=0, unnecessary_services=0, services=0, inspections=0, total_cost=0)
    for r in rows:
        for k, v in cost(r, threshold).items():
            totals[k] += v
    mature = [r for r in rows if r["deadline"] != "before"]
    cancel = [r for r in mature if r["role"] in ("pending_cancel", "reissued_cancel")]
    reissued = [r for r in mature if r["role"] == "reissued_cancel"]
    pending = [r for r in rows if r["deadline"] == "before"]
    accuracy = lambda subset, q: (sum(
        int((r["probabilities"][r["target_entity"]][q] > .5) == bool(r["labels"][r["target_entity"]][q]))
        for r in subset) / len(subset)) if subset else None
    return dict(worlds=len(rows), entities=len(rows) * 8,
                cost_per_entity=totals["total_cost"] / (len(rows) * 8),
                physical=totals, mature_wait=accuracy(mature, 0),
                mature_cancellation_wait=accuracy(cancel, 0),
                mature_reissued_wait=accuracy(reissued, 0),
                pending_wait=accuracy(pending, 0), pending_service=accuracy(pending, 1),
                service_all=accuracy(rows, 1), wait_all=accuracy(rows, 0))


def run(seed, output, smoke=False, replay=False):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    protocol = json.loads((ROOT / "research/dev041/protocol.json").read_text())
    assert seed in protocol["seeds"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rootsha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip():
        raise ValueError("source must be clean and frozen")
    spec_sha = sha(ROOT / "research/dev041/protocol.json")
    record = output / "report.json"
    if replay:
        report = json.loads(record.read_text())
        assert report["source"] == rootsha and report["protocol_sha256"] == spec_sha
        assert report["seed"] == seed and report["smoke"] == smoke
        models = {arm: build(arm, seed) for arm in ARMS}
        for arm in ARMS:
            assert sha(output / (arm + ".pt")) == report["arms"][arm]["checkpoint_sha256"]
            state = torch.load(output / (arm + ".pt"), map_location="cpu", weights_only=True)
            models[arm].load_state_dict(state["model"], strict=True)
    else:
        if record.exists():
            raise ValueError("never overwrite a completed report")
        models = {arm: build(arm, seed) for arm in ARMS}
        compare_initialization(models["bounded_trainable"], models["bounded_random_frozen"])
        init_writer = {arm: copy.deepcopy(models[arm].memory.state_dict())
                       for arm in ARMS[:2]}
        count = 2 if smoke else protocol["train"]["steps"]
        report = dict(study="DEV-041", status="DEVELOPMENT_NOT_CONFIRMATORY",
                      seed=seed, source=rootsha, protocol_sha256=spec_sha, smoke=smoke, arms={})
        for arm in ARMS:
            info = train(models[arm], arm, seed, count, output)
            saved = output / (arm + ".pt")
            torch.save(dict(model=models[arm].state_dict(),
                            initial_writer=init_writer.get(arm),
                            rng=torch.get_rng_state(),
                            steps=count), saved)
            if arm == "bounded_random_frozen" and any(
                not torch.equal(v, init_writer[arm][k])
                for k, v in models[arm].memory.state_dict().items()):
                raise AssertionError("frozen random writer changed")
            report["arms"][arm] = dict(training=info, checkpoint_sha256=sha(saved),
                                      state_scalars=models[arm].state_scalars,
                                      trainable_parameters=sum(p.numel() for p in models[arm].parameters() if p.requires_grad))
    repetitions = 1 if smoke else protocol["calibration"]["repetitions"]
    ereps = 1 if smoke else protocol["evaluation"]["repetitions"]
    for arm in ARMS:
        calibration = predict(models[arm], "calibration", repetitions, 32 if smoke else 512)
        evaluation = predict(models[arm], "eval", ereps, 32 if smoke else 512)
        thr, costs = choose_threshold(calibration)
        info = dict(threshold=thr, calibration_costs=costs,
                    evaluation=summarize(evaluation, thr),
                    fixed=summarize(evaluation, .25),
                    calibration_fingerprints=[r["fingerprint"] for r in calibration],
                    rows=evaluation)
        train_fps = set(report["arms"][arm]["training"]["fingerprints"])
        if train_fps & set(info["calibration_fingerprints"]) or train_fps & {r["fingerprint"] for r in evaluation}:
            raise AssertionError("training/test fingerprint collision")
        if set(info["calibration_fingerprints"]) & {r["fingerprint"] for r in evaluation}:
            raise AssertionError("calibration/evaluation overlap")
        if replay:
            if info != report["arms"][arm]["results"]:
                raise AssertionError("checkpoint exact prediction and decision replay failed")
        else:
            report["arms"][arm]["results"] = info
    anchor = report["arms"]["bounded_trainable"]["results"]
    # Fit baseline only on disjoint calibration cases; never fit on evaluation.
    calib = predict(models["bounded_trainable"], "calibration", repetitions, 32 if smoke else 512)
    ref_fit = last_event_fit(calib, "calibration", repetitions)
    last_dev = last_event_rows(ref_fit, calib, "calibration", repetitions)
    last_eval = last_event_rows(ref_fit, anchor["rows"], "eval", ereps)
    ref_threshold, _ = choose_threshold(last_dev)
    public = sum(sum(int(r["labels"][e][0]) for e in range(8)) for r in anchor["rows"])
    never = sum(4 * sum(int(e[0]) for e in r["labels"]) for r in anchor["rows"])
    always = sum(8 + 4 * sum(int(e[1]) for e in r["labels"]) for r in anchor["rows"])
    refs = dict(last_event=summarize(last_eval, ref_threshold),
                last_event_threshold=ref_threshold,
                public_cost_per_entity=public / (8 * len(anchor["rows"])),
                never_cost_per_entity=never / (8 * len(anchor["rows"])),
                always_cost_per_entity=always / (8 * len(anchor["rows"])))
    if replay:
        if refs != report["references"]:
            raise AssertionError("reference cost replay failed")
    else:
        report["references"] = refs
        record.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not replay:
        m = report["arms"]
        print(json.dumps(dict(seed=seed, models={arm:m[arm]["results"]["evaluation"] for arm in ARMS},
                              last_event=refs["last_event"]), allow_nan=False), flush=True)
    else:
        print("DEV-041 STRICT REPLAY PASSED", seed, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--replay", action="store_true")
    a = ap.parse_args()
    run(a.seed, a.output, a.smoke, a.replay)
