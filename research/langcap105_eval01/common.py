"""Frozen evaluation contract and deterministic corpus coordinates. No model fit."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REGISTRATION_COMMIT = "48c41e27a59659bac03ba59547646fc26aa9cc3d"
EVAL_PROTOCOL_SHA256 = "79d82775eeb346068bb8e38b2209f4976d2ed36920a643baa0f407581917447c"
sys.path.insert(0, str(ROOT / "research/langcap105_verification"))
from checkpoints import SOURCE_HASHES, PROTOCOL_SHA, digest, require


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def load_plan():
    require(digest(HERE / "EVAL_PROTOCOL.json") == EVAL_PROTOCOL_SHA256, "Frozen evaluation protocol changed")
    p = json.loads((HERE / "EVAL_PROTOCOL.json").read_text())
    require(p["optimizer_updates"] == 0 and p["owner_authorized"] is True and
            p["architecture_superiority_eligible"] is False and p["protected_test_access"] is False,
            "Exploratory authorization/qualification boundary changed")
    keys = [(r["scale"], r["arm"], r["seed"]) for r in p["checkpoint_locks"]]
    require(len(keys) == len(set(keys)) == 18, "Incomplete or duplicate checkpoint registration")
    require(digest(ROOT / "research/langcap105/OWNER_PROTOCOL.json") == PROTOCOL_SHA, "Original training recipe changed")
    for path, expected in SOURCE_HASHES.items():
        require(digest(ROOT / path) == expected, "Original model/source implementation changed: " + path)
    return p


def coordinates(records, split, length=128):
    return [(r["path"], offset) for r in sorted(records, key=lambda r: r["path"]) if r["split"] == split
            for offset in range(0, r["bytes"] - length, length)]


def cohorts(receipt, plan, seed):
    records = receipt["identity"]["files"]
    dev = coordinates(records, "DEV")
    shuffled = coordinates(records, "FIT")
    random.Random(seed).shuffle(shuffled)
    require(len(shuffled) >= 2560 * 4, "Original exposure would require a different shuffle epoch")
    seen = shuffled[:2560 * 4]
    fit = seen[:len(dev)]
    e = plan["evaluation"]
    require(len(dev) == e["dev_windows"] == 4646 and len(dev) * 128 == e["dev_targets"], "DEV count changed")
    require(identity(dev) == e["dev_window_ids_sha256"], "DEV coordinates changed")
    require(identity(fit) == e["fit_references_by_seed"][str(seed)]["window_ids_sha256"], "Actually-seen FIT coordinates changed")
    require({n for n, _ in fit}.isdisjoint({n for n, _ in dev}), "FIT/DEV file leakage")
    return {"DEV":dev, "FIT_SEEN_REFERENCE":fit}, seen


def verify_corpus(receipt_path, archive_path, plan):
    sys.path.insert(0, str(ROOT / "research/langcap103"))
    from source import verify_package
    receipt, data = verify_package(receipt_path, archive_path)
    require(receipt["source_sha256"] == plan["source"]["source_sha256"] and
            receipt["archive_sha256"] == plan["source"]["archive_sha256"], "Publisher corpus differs")
    rows = receipt["identity"]["files"]
    fit = [r for r in rows if r["split"] == "FIT"]
    dev = [r for r in rows if r["split"] == "DEV"]
    require(len(fit) == 560 and len(dev) == 18 and len({r["group"] for r in dev}) == 17,
            "Frozen source partition differs")
    return receipt, data


def batch_bytes(data, window_ids):
    pieces = [data[name][offset:offset+129] for name, offset in window_ids]
    require(all(len(raw) == 129 for raw in pieces), "Sample crosses original file boundary")
    return [list(raw[:-1]) for raw in pieces], [list(raw[1:]) for raw in pieces]


class Totals:
    def __init__(self, records):
        self.records = {r["path"]:r for r in records}
        self.parts = {}

    def add(self, name, nll, correct, targets):
        require(math.isfinite(nll) and nll >= 0 and 0 <= correct <= targets and targets > 0, "Invalid metric contribution")
        row = self.parts.setdefault(name, {"losses":[], "correct":0, "targets":0})
        row["losses"].append(float(nll)); row["correct"] += int(correct); row["targets"] += int(targets)

    def finish(self):
        files = {}; groups = {}
        for name, r in sorted(self.parts.items()):
            group = self.records[name]["group"]
            files[name] = {"group":group, "nll_sum":math.fsum(r["losses"]), "correct":r["correct"], "targets":r["targets"]}
            g = groups.setdefault(group, {"losses":[], "correct":0, "targets":0})
            g["losses"].extend(r["losses"]); g["correct"] += r["correct"]; g["targets"] += r["targets"]
        groups = {k:{"nll_sum":math.fsum(v["losses"]), "correct":v["correct"], "targets":v["targets"]} for k, v in sorted(groups.items())}
        n = sum(v["targets"] for v in files.values()); loss = math.fsum(v["nll_sum"] for v in files.values())
        require(n > 0, "Empty scoring denominator")
        return {"targets":n, "nll_sum":loss, "nll":loss/n, "bits_per_byte":loss/n/math.log(2),
                "correct":sum(v["correct"] for v in files.values()),
                "accuracy":sum(v["correct"] for v in files.values())/n, "by_file":files, "by_group":groups}


def unigram_control(data, seen, dev, records):
    # Exactly the original seed's target exposures; no DEV fit, no neural training.
    counts = Counter()
    for name, offset in seen:
        counts.update(data[name][offset+1:offset+129])
    require(sum(counts.values()) == 1310720, "Diagnostic FIT exposure differs")
    denominator = sum(counts.values()) + 256
    losses = [-math.log((counts[b]+1)/denominator) for b in range(256)]
    prediction = min(range(256), key=lambda b:(-counts[b], b))
    total = Totals(records)
    for name, offset in dev:
        target = data[name][offset+1:offset+129]
        total.add(name, math.fsum(losses[b] for b in target), sum(b == prediction for b in target), 128)
    return {"classification":"UNMATCHED_FIT_EXPOSURE_DIAGNOSTIC_ONLY", "fit_target_exposures":1310720,
            "exposure_window_ids_sha256":identity(seen), "metrics":total.finish()}
