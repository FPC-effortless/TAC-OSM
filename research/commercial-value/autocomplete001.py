#!/usr/bin/env python3
"""AUTOCOMPLETE001: no-fit, authentic-source, offline code prediction value screen.

Reuses the 20 immutable LANGHEAD094 trained checkpoints. No model fitting, TEST,
model promotion, payment, external account, or mobile performance claim.

A 6-byte completion suggestion is emitted at predetermined nonoverlapping
positions, without looking at future source bytes. "Oracle saved keystrokes"
assumes perfect acceptance ONLY of exact suggestions with one Tab keystroke.
It is an optimistic simulation, not measured user adoption or monetary value.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import statistics
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
REG = ROOT / "research" / "commercial-value"
LANG = ROOT / "research" / "mathematical-engineering" / "langhead094"
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(LANG)]
from model094 import ARMS, SEEDS, PARAMS, make_model  # noqa: E402
from scripts.replay_langmem086087 import fit_sources  # noqa: E402

SOURCE_COMMIT = "2c1b30d0503cfb064f1cb252e6614a06915a362a"
SOURCE_BLOBS = {
    "src/flask/app.py": ("1232b03dd40bb45aa0f7448ee0ba780d21aa3778", 61744),
    "src/flask/blueprints.py": ("b6d4e43339f0d88fc551cb5211c62be8988d3666", 4541),
    "src/flask/helpers.py": ("5d412c90f2bde5ab32adcf6126908330bb2880d6", 23517),
}
START, STOP = 128, 640
OFFER_EVERY, OFFER_LEN = 16, 6
FIT_SHA = "911981cc63fd5436e0495cb067784e53475971131c811b7026e07d97a886d2d4"
ARCHIVE = ROOT / "research/replay-bundles/LANGHEAD094"

def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()

def git_blob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()

def validate_archive() -> dict:
    """Fail closed before opening any source outcomes."""
    archives = {}
    for arm in ARMS:
        for seed in SEEDS:
            folder = ARCHIVE / f"{arm}-{seed}"
            cp, receipt = folder / "MODEL.pt", folder / "RESULTS.json"
            if not cp.is_file() or not receipt.is_file():
                raise FileNotFoundError(f"original checkpoint or receipt absent: {arm}-{seed}")
            r = json.loads(receipt.read_text())
            if r.get("study") != "LANGHEAD094" or r.get("arm") != arm or r.get("seed") != seed:
                raise ValueError("wrong original checkpoint identity")
            if sha256(cp.read_bytes()) != r["checkpoint_sha256"]:
                raise ValueError("checkpoint SHA256 mismatch")
            if r["counts"] != {"trainable": PARAMS[arm], "deployed": PARAMS[arm]}:
                raise ValueError("incorrect trained parameter count")
            archives[(arm, seed)] = {"path": cp, "sha256": r["checkpoint_sha256"]}
    return archives

def fetch_flask_sources() -> dict[str, bytes]:
    docs = {}
    for path, (expected_git_blob, size) in sorted(SOURCE_BLOBS.items()):
        url = f"https://raw.githubusercontent.com/pallets/flask/{SOURCE_COMMIT}/{path}"
        with urllib.request.urlopen(url, timeout=60) as response:
            raw = response.read()
        if len(raw) != size or git_blob(raw) != expected_git_blob:
            raise ValueError(f"immutable Flask publisher Git blob mismatch: {path}")
        if len(raw) <= STOP + OFFER_LEN:
            raise ValueError("too little source for fixed original positions")
        docs[path] = raw
    return docs

def state_bytes(state) -> int:
    if hasattr(state, "tensors"):
        return sum(x.numel() * x.element_size() for x in state.tensors.values())
    if hasattr(state, "value"):
        return state.value.numel() * state.value.element_size()
    raise ValueError("unsupported original state")

def load_model(arm: str, seed: int, entry: dict):
    cp = torch.load(entry["path"], map_location="cpu", weights_only=True)
    if cp["arm"] != arm or cp["seed"] != seed or cp["source_sha256"] != FIT_SHA:
        raise ValueError("model FIT/source identity failed")
    model = make_model(arm).eval()
    model.load_state_dict(cp["state"], strict=True)
    return model

def quantile(xs: list[float], fraction: float) -> float:
    return sorted(xs)[int((len(xs) - 1) * fraction)]

def greedy_suggestion(step, logits, state, length=OFFER_LEN) -> bytes:
    """Decode only the model's own generated observations, NEVER real future."""
    outputs = []
    trial_state = state
    for i in range(length):
        b = int(logits[0, -1].argmax())
        outputs.append(b)
        if i + 1 != length:
            logits, trial_state = step(b, trial_state)
    return bytes(outputs)

@torch.inference_mode()
def score_neural_doc(model, owner: str, raw: bytes) -> dict:
    s = model.start((owner,))
    if state_bytes(s) <= 0:
        raise ValueError("no state")
    timing = []
    nll = []
    top1 = top5 = exact = offers = 0
    for t in range(STOP):
        start = time.perf_counter()
        logits, s = model(torch.tensor([[raw[t]]], dtype=torch.long),
                          state=s, owners=(owner,))
        if t < START:
            continue
        timing.append((time.perf_counter() - start) * 1000)
        target = raw[t + 1]
        lp = logits[0, -1].log_softmax(-1)
        nll.append(float(-lp[target]))
        rank = torch.argsort(lp, descending=True)[:5]
        top1 += int(int(rank[0]) == target)
        top5 += int(bool((rank == target).any()))
        if (t - START) % OFFER_EVERY == 0:
            offers += 1
            def step(b, state):
                return model(torch.tensor([[b]], dtype=torch.long),
                             state=state, owners=(owner,))
            proposed = greedy_suggestion(step, logits, s)
            exact += int(proposed == raw[t + 1:t + 1 + OFFER_LEN])
    n = STOP - START
    return {
        "owner": owner, "n": n,
        "nll_sum": sum(nll), "top1_count": top1, "top5_count": top5,
        "offers": offers, "exact6_accepts_oracle": exact,
        "oracle_net_saved_keys": exact * (OFFER_LEN - 1),
        "inference_one_byte_p50_ms": quantile(timing, .50),
        "inference_one_byte_p95_ms": quantile(timing, .95),
        "persistent_state_bytes": state_bytes(s),
    }

class FitOnlyP3:
    """No-trained-weights hierarchical FIT-only byte count reference.

    p(y|abc)=(c_abc(y)+tau*p_unigram(y))/(N_abc+tau).
    Source lookups are from the original verified FIT corpus, never Flask.
    """
    def __init__(self, docs: dict[str, bytes], tau: float=4.):
        self.counts = defaultdict(Counter)
        self.base = Counter()
        self.tau = tau
        for raw in docs.values():
            self.base.update(raw)
            for i in range(3, len(raw)):
                self.counts[raw[i - 3:i]][raw[i]] += 1
        denom = sum(self.base.values()) + 256
        self.prior = [(self.base[b] + 1) / denom for b in range(256)]
        self.state_approx_bytes = (sum(3 + 2 * len(v) + 4 * len(v) for v in self.counts.values())
                                   + 256 * 8)  # logical packed estimate, not Python RSS
    def probabilities(self, prior: bytes) -> list[float]:
        key = prior[-3:]
        c = self.counts.get(key, {})
        total = sum(c.values())
        den = total + self.tau
        return [(c.get(y, 0) + self.tau * self.prior[y]) / den for y in range(256)]
    def predict(self, prior: bytes) -> int:
        p = self.probabilities(prior)
        return max(range(256), key=lambda i: (p[i], -i))

def score_count_doc(p3: FitOnlyP3, owner: str, raw: bytes) -> dict:
    nll = []
    top1 = top5 = offers = exact = 0
    timing = []
    for t in range(START, STOP):
        context = raw[t - 2:t + 1]
        start = time.perf_counter()
        p = p3.probabilities(context)
        timing.append((time.perf_counter() - start) * 1000)
        target = raw[t + 1]
        nll.append(-math.log(p[target]))
        ordering = sorted(range(256), key=lambda b: (-p[b], b))[:5]
        top1 += int(ordering[0] == target)
        top5 += int(target in ordering)
        if (t - START) % OFFER_EVERY == 0:
            offers += 1
            predicted = []
            trial = context
            for _ in range(OFFER_LEN):
                nxt = p3.predict(trial)
                predicted.append(nxt)
                trial = (trial + bytes([nxt]))[-3:]
            exact += int(bytes(predicted) == raw[t + 1:t + 1 + OFFER_LEN])
    return {
        "owner": owner, "n": STOP - START,
        "nll_sum": sum(nll), "top1_count": top1, "top5_count": top5,
        "offers": offers, "exact6_accepts_oracle": exact,
        "oracle_net_saved_keys": exact * (OFFER_LEN - 1),
        "inference_one_byte_p50_ms": quantile(timing, .50),
        "inference_one_byte_p95_ms": quantile(timing, .95),
        "persistent_state_bytes": 3,
    }

def aggregate(doc_rows: list[dict]) -> dict:
    n = sum(r["n"] for r in doc_rows)
    offers = sum(r["offers"] for r in doc_rows)
    if n != 3 * (STOP - START) or offers != 3 * ((STOP - START) // OFFER_EVERY):
        raise ValueError("missing fixed label or offer denominator")
    return {
        "targets": n, "offers": offers,
        "mean_nll": sum(r["nll_sum"] for r in doc_rows) / n,
        "top1_accuracy": sum(r["top1_count"] for r in doc_rows) / n,
        "top5_accuracy": sum(r["top5_count"] for r in doc_rows) / n,
        "oracle_accepted_suggestions": sum(r["exact6_accepts_oracle"] for r in doc_rows),
        "oracle_net_saved_keys": sum(r["oracle_net_saved_keys"] for r in doc_rows),
        "oracle_net_saved_vs_typed_target_ratio": sum(r["oracle_net_saved_keys"] for r in doc_rows) / n,
        "persistent_state_bytes": max(r["persistent_state_bytes"] for r in doc_rows),
        "per_owner": doc_rows,
    }

def main(output: Path):
    if output.exists():
        raise FileExistsError("refuse to overwrite original report")
    torch.set_num_threads(2)
    archive = validate_archive()
    docs = fetch_flask_sources()  # after checkpoint existence and SHA verification
    original_fit, fit_hash = fit_sources(ROOT)
    if fit_hash != FIT_SHA or len(original_fit) != 14:
        raise ValueError("original authenticated FIT corpus failed")
    p3 = FitOnlyP3(original_fit)
    baselines = aggregate([score_count_doc(p3, owner, data) for owner, data in sorted(docs.items())])
    measures = {}
    for arm in ARMS:
        runs = []
        for seed in SEEDS:
            model = load_model(arm, seed, archive[(arm, seed)])
            rows = [score_neural_doc(model, owner, data) for owner, data in sorted(docs.items())]
            entry = aggregate(rows)
            entry.update({"seed": seed, "checkpoint_sha256": archive[(arm, seed)]["sha256"],
                          "weight_buffer_bytes": sum(v.numel() * v.element_size() for v in model.state_dict().values())})
            runs.append(entry)
        measures[arm] = {
            "seeds": runs,
            "mean_nll": statistics.mean(v["mean_nll"] for v in runs),
            "mean_top1_accuracy": statistics.mean(v["top1_accuracy"] for v in runs),
            "mean_oracle_net_saved_keys": statistics.mean(v["oracle_net_saved_keys"] for v in runs),
            "min_oracle_net_saved_keys": min(v["oracle_net_saved_keys"] for v in runs),
            "max_oracle_net_saved_keys": max(v["oracle_net_saved_keys"] for v in runs),
        }
    result = {
        "study": "AUTOCOMPLETE001", "classification": "INTERNAL_EXPLORATORY_NOT_COMMERCIAL_PROOF",
        "original_checkpoints": "LANGHEAD094", "no_new_training": True,
        "no_model_promotion": True, "no_protected_test": True, "gpu_minutes": 0,
        "flask_commit": SOURCE_COMMIT,
        "source_git_blob": {k: {"git_blob_sha1": v[0], "size": v[1], "sha256": sha256(docs[k])}
                            for k, v in SOURCE_BLOBS.items()},
        "fit_source_sha256": fit_hash, "position_range": [START, STOP],
        "offer_every": OFFER_EVERY, "suggestion_length": OFFER_LEN,
        "utility_is_oracle_simulation_not_users": True,
        "limitations": [
            "No real developer accepted suggestions; keystroke saving is an optimistic oracle upper bound.",
            "CPU latency only, hardware and training FLOPs unmatched; no physical Android energy or RAM.",
            "Public Flask code is new to this study, not provably unseen by all researchers or free of train overlap.",
            "Only three Flask files; no measured time saving, willingness-to-pay, revenue or profit.",
            "P3 FIT-only count-store memory is larger and not compute/quality-matched to neural predictors.",
            "Model predicts bytes, not semantic multi-line completions.",
        ],
        "baseline_fit_only_p3": baselines, "baseline_p3_logical_packed_estimate_bytes": p3.state_approx_bytes,
        "models": measures,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "status": "INTERNAL_EXPLORATORY", "output": str(output),
        "p3_nll": baselines["mean_nll"],
        "original_tac_mean_nll": measures["original"]["mean_nll"],
        "p3_oracle_saved": baselines["oracle_net_saved_keys"],
        "tac_mean_oracle_saved": measures["original"]["mean_oracle_net_saved_keys"],
    }, sort_keys=True))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    main(args.output)
