"""DEV-021: frozen trained decay/erase transport of matched public suffixes.

No new world-model training, altered-state predictions, or capability claims.
Only original DEV-018 checkpoint weights are used, with float64 diagnostics
to verify the exact linear difference equation on the identical suffix.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tac_osm.development.memory_baselines import MemoryBaseline  # noqa: E402
from tac_osm.development.schema_maintenance import make_pair  # noqa: E402
from tac_osm.development.noisy_memory import parameter_hash  # noqa: E402
from run_learned_schema_dev import public_queries, public_stream_tensors  # noqa: E402

SEEDS = (43, 59, 71)
REGIMES = ("short", "long")
NATIVE = ("tac_joint", "tac_encoder")
CONTROLS = ("schema41", "gru768", "memoryless")
PREFIXES = (32, 128, 512)
DUE = (32, 128, 512)
DEADLINES = ("at", "after")
ROLES = ("pending_cancel", "reissued_cancel")
DOMAIN = "dev021-transport-v1"
ATOL = 2e-9
RTOL = 2e-5


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fresh_groups(seed, repetitions):
    """Separate synthetic domain; evaluator-only future labels never enter model."""
    for delay in DUE:
        for deadline in DEADLINES:
            for role in ROLES:
                pairs = [
                    make_pair(seed * 1_000_000 + rep, 512, delay, role, deadline, domain=DOMAIN)
                    for rep in range(repetitions)
                ]
                pivots = []
                fingerprints = set()
                for a, b in pairs:
                    differing = [i for i, (x, y) in enumerate(zip(a.observations, b.observations))
                                 if x != y]
                    if not differing:
                        raise AssertionError("no historical difference")
                    last = differing[-1]
                    if last >= 31 or not all(x == y for x, y in
                                               zip(a.observations[last+1:], b.observations[last+1:])):
                        raise AssertionError("missing common public suffix")
                    if a.target_entity != b.target_entity or a.tick != 512 or b.tick != 512:
                        raise AssertionError("pair target/tick mismatch")
                    if a.wait_targets[a.target_entity] == b.wait_targets[b.target_entity]:
                        raise AssertionError("pair has no opposite outcomes")
                    if a.fingerprint == b.fingerprint:
                        raise AssertionError("pair fingerprints equal")
                    for e in (a, b):
                        if e.fingerprint in fingerprints:
                            raise AssertionError("duplicate test history")
                        fingerprints.add(e.fingerprint)
                    pivots.append(last)
                if len(set(pivots)) != 1:
                    raise AssertionError("a batch must have a single divergence boundary")
                yield delay, deadline, role, pairs, pivots[0]


def frozen_model(report, folder, arm, seed):
    checkpoint = folder / (arm + ".pt")
    expected = report["models"][arm]["sha256"]
    if digest(checkpoint) != expected:
        raise AssertionError("frozen checkpoint SHA256 mismatch")
    model = MemoryBaseline(arm, seed + 18000).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    if parameter_hash(model) != report["training"][arm]["final_hash"]:
        raise AssertionError("frozen parameter hash mismatch")
    for p in model.parameters():
        p.requires_grad_(False)
    return model, expected


def suffix_difference_step(memory, delta, encoded, public_value, decay_on=True,
                           erase_on=True):
    """One precisely defined factor of the original trained pair difference.

    The additive value-side write is absent because public suffixes are equal.
    """
    joint = torch.cat((encoded, F.one_hot(public_value.long(), 2).to(encoded.dtype)), -1)
    keys = memory.keys(encoded)
    assignment = memory.assignment(encoded).softmax(-1)
    erase = memory.erase(joint).reshape(-1, 4, 24).sigmoid()
    decay = memory.decay(joint).sigmoid()
    outer = torch.einsum("bevk,bek->bev", delta, erase * keys)[..., None] * keys[:, :, None, :]
    contraction = assignment[:, :, None, None] * outer
    retained = delta * (decay[:, :, None, None] if decay_on else 1.)
    return retained - (contraction if erase_on else 0.), decay


def l2_pairs(t):
    p = t.reshape(-1, 2, *t.shape[1:])
    return torch.linalg.vector_norm((p[:, 0] - p[:, 1]).flatten(1), dim=-1)


def run(seed, regime, frozen, output, smoke=False):
    if seed not in SEEDS or regime not in REGIMES:
        raise ValueError("unknown frozen regime/seed")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    report = None
    if not smoke:
        if frozen is None:
            raise ValueError("full diagnostic requires the archived DEV-018 data")
        report = json.loads((frozen / "report.json").read_text())
        if (report["study"] != "DEV-018" or report["seed"] != seed or
                report["regime"] != regime or report["smoke"] or
                report["status"] != "ADAPTIVE_DEVELOPMENT_NOT_CONFIRMATORY"):
            raise AssertionError("frozen report provenance mismatch")
    repetitions = 1 if smoke else 16
    source = {}
    models = {}
    for arm in NATIVE + CONTROLS:
        if report is not None:
            models[arm], source[arm] = frozen_model(report, frozen, arm, seed)
        else:
            models[arm] = MemoryBaseline(arm, seed + 18000).eval()
            source[arm] = None
    for arm in NATIVE:
        models[arm] = models[arm].double()  # Float64 ONLY for algebraic diagnostic
    result = {"study": "DEV-021", "status": "FROZEN_ADAPTIVE_DIAGNOSTIC_NOT_CONFIRMATORY",
              "seed": seed, "regime": regime, "smoke": smoke,
              "dtype": "torch.float64", "atol": ATOL, "rtol": RTOL,
              "domain": DOMAIN, "pairs_per_arm": 12 * repetitions,
              "source_run": 38802569971, "checkpoint_sha256": source,
              "rows": []}
    with torch.no_grad():
        for due, deadline, role, pairs, pivot in fresh_groups(seed, repetitions):
            cases = [episode for pair in pairs for episode in pair]
            observations, entity, value = public_stream_tensors([c.observations for c in cases])
            if observations.shape != (2 * repetitions, 512, 16):
                raise AssertionError("incorrect observation budget")
            for t in range(pivot + 1, 512):
                if not (torch.equal(observations[::2, t], observations[1::2, t]) and
                        torch.equal(entity[::2, t], entity[1::2, t]) and
                        torch.equal(value[::2, t], value[1::2, t])):
                    raise AssertionError("nonidentical model-visible suffix")
            control_norms = {}
            for ctrl in CONTROLS:
                control_norms[ctrl] = {}
                for h in PREFIXES:
                    state = models[ctrl].encode(observations[:, :h], entity[:, :h], value[:, :h])
                    control_norms[ctrl][h] = l2_pairs(state).tolist()
            for arm in NATIVE:
                model = models[arm]
                x = model.event_encoder(observations.double())
                state = torch.zeros((len(cases), 4, 8, 24), dtype=torch.float64)
                for t in range(pivot + 1):
                    state = model.memory.update(state, x[:, t], value[:, t])
                d = state[::2] - state[1::2]
                init = torch.linalg.vector_norm(d.flatten(1), dim=-1)
                transported = {k: d.clone() for k in ("actual", "no_decay", "no_erase", "neither")}
                log_decay = torch.zeros((repetitions, 4), dtype=torch.float64)
                traces = {}
                max_error = torch.zeros(repetitions, dtype=torch.float64)
                for t in range(pivot + 1, 512):
                    suffix_x, suffix_v = x[::2, t], value[::2, t]
                    transported["actual"], decay = suffix_difference_step(
                        model.memory, transported["actual"], suffix_x, suffix_v)
                    transported["no_decay"], _ = suffix_difference_step(
                        model.memory, transported["no_decay"], suffix_x, suffix_v, False, True)
                    transported["no_erase"], _ = suffix_difference_step(
                        model.memory, transported["no_erase"], suffix_x, suffix_v, True, False)
                    transported["neither"], _ = suffix_difference_step(
                        model.memory, transported["neither"], suffix_x, suffix_v, False, False)
                    log_decay += decay.log()
                    state = model.memory.update(state, x[:, t], value[:, t])
                    real = state[::2] - state[1::2]
                    err = (transported["actual"] - real).abs().flatten(1).amax(1)
                    max_error = torch.maximum(max_error, err)
                    limit = ATOL + RTOL * real.abs().flatten(1).amax(1)
                    if not bool(torch.all(err <= limit)):
                        raise AssertionError(f"frozen transport identity failed {arm} {t} {float(err.max())}")
                    if t + 1 in PREFIXES:
                        traces[t + 1] = {
                            "actual": torch.linalg.vector_norm(real.flatten(1), dim=-1).tolist(),
                            **{key: torch.linalg.vector_norm(arr.flatten(1), dim=-1).tolist()
                               for key, arr in transported.items()}}
                if set(traces) != set(PREFIXES):
                    raise AssertionError("missing checkpoint distances")
                queries = public_queries([512] * len(cases), [c.horizon for c in cases]).double()
                hidden_read = []
                hook = model.head.register_forward_pre_hook(
                    lambda _module, args: hidden_read.append(args[0].detach().clone()))
                try:
                    logits = model.decode(state, queries)
                finally:
                    hook.remove()
                if len(hidden_read) != 1:
                    raise AssertionError("missing target readout")
                head_inputs = hidden_read[0].reshape(len(cases), 8, 3, 44)
                target = torch.tensor([c.target_entity for c in cases])
                inds = torch.arange(len(cases))
                read = head_inputs[inds, target, 0]
                prob = logits.softmax(-1)[inds, target, 0, 1]
                prediction = logits.argmax(-1)[inds, target, 0]
                for i, (a, b) in enumerate(pairs):
                    per_prefix = {}
                    for h in PREFIXES:
                        measures = {key + "_l2": vals[i] for key, vals in traces[h].items()}
                        measures.update({ctrl + "_l2": control_norms[ctrl][h][i]
                                         for ctrl in CONTROLS})
                        per_prefix[str(h)] = measures
                    result["rows"].append({
                        "arm": arm, "role": role, "deadline": deadline, "due_delay": due,
                        "repetition": i, "divergence_tick": pivot + 1,
                        "suffix_events": 511 - pivot,
                        "fingerprints": [a.fingerprint, b.fingerprint],
                        "initial_l2": float(init[i]),
                        "zero_initial": bool(init[i] < 1e-12),
                        "prefixes": per_prefix,
                        "max_transport_error": float(max_error[i]),
                        "trained_decay_product": log_decay[i].exp().tolist(),
                        "read_input_l2": float(torch.linalg.vector_norm(read[2*i] - read[2*i+1])),
                        "original_probability_gap": float((prob[2*i] - prob[2*i+1]).abs()),
                        "original_class_difference": int(prediction[2*i] != prediction[2*i+1])
                    })
        expected = (len(NATIVE) * 12 * repetitions)
        if len(result["rows"]) != expected:
            raise AssertionError(f"incomplete grid: {len(result['rows'])} vs {expected}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    for arm in NATIVE:
        rows = [r for r in result["rows"] if r["arm"] == arm]
        print("DEV021_ARM " + json.dumps({
            "arm": arm, "seed": seed, "regime": regime,
            "pairs": len(rows),
            "mean_initial_l2": sum(r["initial_l2"] for r in rows) / len(rows),
            "mean_H512_l2": sum(r["prefixes"]["512"]["actual_l2"] for r in rows)/len(rows),
            "mean_no_decay_l2": sum(r["prefixes"]["512"]["no_decay_l2"] for r in rows)/len(rows),
            "mean_no_erase_l2": sum(r["prefixes"]["512"]["no_erase_l2"] for r in rows)/len(rows),
            "mean_no_both_l2": sum(r["prefixes"]["512"]["neither_l2"] for r in rows)/len(rows),
            "max_transport_error": max(r["max_transport_error"] for r in rows)
        }), flush=True)
    print("DEV021_DONE " + json.dumps({"seed": seed, "regime": regime,
                                       "smoke": smoke, "rows": len(result["rows"])}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--regime", choices=REGIMES, required=True)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.seed, args.regime, args.frozen, args.output, args.smoke)
