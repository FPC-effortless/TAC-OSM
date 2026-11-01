"""Training utilities with a separate pixels-only inference boundary.

No imports in this module initiate downloads, allocate GPUs or fit a model.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from research.tac_ocr001.tokenizer import BOS, EOS, PAD, decode


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "research/mathematical-engineering/tac_ocr003/FAIR_COMPARISON.json"
PROTOCOL = Path(__file__).with_name("PROTOCOL.json")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def require_prefit():
    """Use the canonical signed gate. Never manufacture an audit/signing key."""
    import os
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from scripts.validate_research_parity import check_manifest
    raw = os.environ.get("TAC_OSM_PREFIT_ATTESTATION_JSON")
    evidence = json.loads(raw) if raw else None
    return check_manifest(json.loads(MANIFEST.read_text()), "preflight",
                          reward_attestation=evidence,
                          trusted_reward_public_key_hex=os.environ.get(
                              "TAC_OSM_REWARD_VERIFIER_PUBLIC_KEY_HEX"))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def load_rows(data, split, expected_sha):
    name = {"train": "train_filtered.jsonl", "validation": "validation_original.jsonl"}[split]
    path = Path(data) / name
    if sha256(path) != expected_sha:
        raise ValueError("prepared split differs from the frozen registration")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if any(row["split"] != split or not row["text"] for row in rows):
        raise ValueError("invalid source split or blank target")
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("sample identities must be unique")
    return rows


def epoch_batches(rows, seed, epoch, batch_size, pool_size=1024):
    """Length buckets reduce padded work; every original row occurs once."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    order = rng.permutation(len(rows)).tolist()
    batches = []
    for offset in range(0, len(order), pool_size):
        pool = sorted(order[offset:offset + pool_size],
                      key=lambda i: len(rows[i]["text"].encode("utf-8")))
        batches.extend(pool[i:i + batch_size] for i in range(0, len(pool), batch_size))
    rng.shuffle(batches)
    return batches


def targets(rows, device):
    encoded = [[BOS, *r["text"].encode("utf-8"), EOS] for r in rows]
    length = max(map(len, encoded)) - 1
    inputs = torch.full((len(rows), length), PAD, dtype=torch.long, device=device)
    labels = torch.full_like(inputs, -100)
    for i, values in enumerate(encoded):
        inputs[i, :len(values) - 1] = torch.tensor(values[:-1], device=device)
        labels[i, :len(values) - 1] = torch.tensor(values[1:], device=device)
    return inputs, labels


def learning_rate(step, total_steps, warmup_steps, peak):
    if step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    return peak * (0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2)


@torch.inference_mode()
def predict(model, features, boxes, owners, max_bytes=64):
    """No text targets, categories or target lengths accepted by inference."""
    model.eval()
    refs = model.prepare_visual(features, boxes, owners=owners,
                                page_ids=tuple("region" for _ in owners))
    state = model.start(owners)
    token = torch.full((len(owners), 1), BOS, dtype=torch.long, device=features.device)
    finished = torch.zeros(len(owners), dtype=torch.bool, device=features.device)
    sequences = [[] for _ in owners]
    terminated = [False for _ in owners]
    for _ in range(max_bytes + 1):
        logits, state = model.step(token, refs, owners=owners, state=state)
        logits = logits[:, -1].clone()
        logits[:, BOS] = logits[:, PAD] = -torch.inf
        token = logits.argmax(-1, keepdim=True)
        values, previously_finished = token[:, 0].tolist(), finished.tolist()
        for i, value in enumerate(values):
            if not previously_finished[i]:
                if value == EOS:
                    terminated[i] = True
                elif len(sequences[i]) < max_bytes:
                    sequences[i].append(value)
        finished |= token[:, 0] == EOS
        if bool(finished.all()):
            break
    return [decode(values, errors="replace") for values in sequences], terminated


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def transcription_metrics(gold, predicted):
    if len(gold) != len(predicted) or not gold:
        raise ValueError("paired nonempty transcription lists required")
    char_denominator = sum(len(x) for x in gold)
    word_denominator = sum(len(x.split()) for x in gold)
    return {"regions": len(gold),
            "character_error_rate": sum(edit_distance(a, b) for a, b in zip(gold, predicted)) / max(1, char_denominator),
            "word_error_rate": sum(edit_distance(a.split(), b.split()) for a, b in zip(gold, predicted)) / max(1, word_denominator),
            "exact_region_accuracy": sum(a == b for a, b in zip(gold, predicted)) / len(gold),
            "gold_characters": char_denominator, "gold_words": word_denominator,
            "normalization": "none; Unicode characters; whitespace-split words"}


def atomic_checkpoint(path, model, optimizer, *, cursor, identity, counters):
    value = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "cursor": cursor, "identity": identity, "counters": counters,
             "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
             "python_rng": random.getstate()}
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def restore_checkpoint(path, model, optimizer, identity):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value["identity"] != identity:
        raise ValueError("checkpoint source/protocol/seed identity changed")
    model.load_state_dict(value["model"], strict=True)
    optimizer.load_state_dict(value["optimizer"])
    torch.set_rng_state(value["torch_rng"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(value["cuda_rng"])
    random.setstate(value["python_rng"])
    return value["cursor"], value["counters"]
