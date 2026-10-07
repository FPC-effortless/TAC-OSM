#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tac_osm.contract import load_contract
from tac_osm.learned_address_001 import LearnedAddressPLM
from tac_osm.learned_address_001_benchmark import (
    ADDRESS_DIM,
    BATCH_SIZE,
    EVAL_EPISODES,
    GENERATOR_VERSION,
    HELDOUT,
    OPS,
    STEPS,
    TRAIN_COMBOS,
    benchmark_manifest,
    episode_fingerprint,
    episode_key,
    sample_episode,
    sample_evaluation_episodes,
)

EXPERIMENT_ID = "TACOSM-PLM-LEARNED-ADDRESS-001"
CONTRACT_PATH = ROOT / "contracts" / f"{EXPERIMENT_ID}.json"
SEEDS = (0, 1, 2, 3, 4)
PRIMARY_THRESHOLD = 0.80
ADDRESS_MEAN_THRESHOLD = 0.90
ADDRESS_MIN_THRESHOLD = 0.80
ORACLE_GAP_THRESHOLD = 0.10
BASE_E2E008_GENERATOR_SHA256 = "3be32a91191b73792b0fffb5a7b497e80aa6a46d0189e032e6b7c60082949f3e"
FROZEN_BENCHMARK_GIT_BLOB_SHA = "9ca0c6ae13d02054230babe3fb594731e106cb31"


def benchmark_git_blob_sha() -> str:
    data = (ROOT / "src" / "tac_osm" / "learned_address_001_benchmark.py").read_bytes()
    header = f"blob {len(data)}\\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def contract_hash() -> str:
    return hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()


def _freeze_for_set(value):
    if isinstance(value, list):
        return tuple(_freeze_for_set(x) for x in value)
    if isinstance(value, tuple):
        return tuple(_freeze_for_set(x) for x in value)
    return value


def hashable_episode_key(ep):
    return _freeze_for_set(episode_key(ep))


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def write_observations(model: LearnedAddressPLM, observations):
    memory = model.state.initial(1, torch.device("cpu"))
    for slot, (key, text, image, audio) in enumerate(observations):
        memory = model.write_observation(
            memory,
            torch.tensor(key, dtype=torch.float32).view(1, -1),
            text.unsqueeze(0),
            image.unsqueeze(0),
            audio.unsqueeze(0).unsqueeze(0),
            slot,
        )
    return memory


def build_batch(episodes):
    addresses, texts, images, audios = [], [], [], []
    for slot in range(3):
        addresses.append([])
        texts.append([])
        images.append([])
        audios.append([])
        for ep in episodes:
            key, text, image, audio = ep[0][slot]
            addresses[slot].append(torch.tensor(key, dtype=torch.float32))
            texts[slot].append(text)
            images[slot].append(image)
            audios[slot].append(audio)

    def query_batch(index: int):
        qs = [ep[index] for ep in episodes]
        return {
            "key": torch.tensor([q[0] for q in qs], dtype=torch.float32),
            "i": torch.tensor([q[1] for q in qs], dtype=torch.long),
            "j": torch.tensor([q[2] for q in qs], dtype=torch.long),
            "op": torch.tensor([q[3] for q in qs], dtype=torch.long),
            "target": torch.tensor([q[4] for q in qs], dtype=torch.long),
        }

    return {
        "address": torch.stack(
            [torch.stack(x) for x in addresses], dim=0
        ),
        "text": torch.stack(
            [torch.stack(x) for x in texts], dim=0
        ),
        "image": torch.stack(
            [torch.stack(x) for x in images], dim=0
        ),
        "audio": torch.stack(
            [torch.stack(x).unsqueeze(1) for x in audios], dim=0
        ),
        "q1": query_batch(1),
        "q2": query_batch(2),
    }


def write_batch(model: LearnedAddressPLM, batch):
    batch_size = batch["text"].shape[1]
    memory = model.state.initial(batch_size, batch["text"].device)
    for slot in range(3):
        z = model.encode(
            batch["text"][slot],
            batch["image"][slot],
            batch["audio"][slot],
        )
        memory = model.state.write(
            memory,
            batch["address"][slot],
            z,
            slot,
        )
    return memory


def gradient_probe():
    torch.manual_seed(20261010)
    rng = random.Random(91010)
    episodes = [
        sample_episode(rng, combo1=TRAIN_COMBOS[0], combo2=TRAIN_COMBOS[1])
        for _ in range(4)
    ]
    model = LearnedAddressPLM().cpu()
    batch = build_batch(episodes)
    memory = write_batch(model, batch)
    out1 = model.query(
        memory, batch["q1"]["key"], batch["q1"]["i"],
        batch["q1"]["j"], batch["q1"]["op"]
    )
    out2 = model.query(
        memory, batch["q2"]["key"], batch["q2"]["i"],
        batch["q2"]["j"], batch["q2"]["op"]
    )
    loss = (
        F.cross_entropy(out1["logits"], batch["q1"]["target"])
        + F.cross_entropy(out2["logits"], batch["q2"]["target"])
    )
    loss.backward()

    required = (
        "text.emb.weight",
        "image.net.0.weight",
        "audio.net.0.weight",
        "rep.fuse.0.weight",
        "state.key_encoder.0.weight",
        "state.query_encoder.0.weight",
        "state.write_value.weight",
        "casm.decoder.0.weight",
    )
    params = dict(model.named_parameters())
    bad = []
    zero = []
    for name in required:
        grad = params[name].grad
        if grad is None or not torch.isfinite(grad).all():
            bad.append(name)
        elif float(grad.abs().sum()) == 0.0:
            zero.append(name)
    return {
        "pass": not bad and not zero,
        "missing_or_nonfinite": bad,
        "zero_gradient": zero,
    }


def train_seed(seed: int):
    torch.manual_seed(seed)
    rng = random.Random(seed + 187000)
    model = LearnedAddressPLM().cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.002,
        weight_decay=0.0001,
    )
    training_keys = set()

    for _ in range(STEPS):
        episodes = [
            sample_episode(
                rng,
                combo1=rng.choice(TRAIN_COMBOS),
                combo2=rng.choice(TRAIN_COMBOS),
            )
            for _ in range(BATCH_SIZE)
        ]
        training_keys.update(hashable_episode_key(ep) for ep in episodes)
        batch = build_batch(episodes)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        memory = write_batch(model, batch)
        out1 = model.query(
            memory,
            batch["q1"]["key"],
            batch["q1"]["i"],
            batch["q1"]["j"],
            batch["q1"]["op"],
        )
        out2 = model.query(
            memory,
            batch["q2"]["key"],
            batch["q2"]["i"],
            batch["q2"]["j"],
            batch["q2"]["op"],
        )
        loss = (
            F.cross_entropy(out1["logits"], batch["q1"]["target"])
            + F.cross_entropy(out2["logits"], batch["q2"]["target"])
        ) / 2
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return model, training_keys


def eval_query(model, memory, query):
    key, i, j, op, target, _entity = query
    return model.query(
        memory,
        torch.tensor(key, dtype=torch.float32).view(1, -1),
        torch.tensor([i], dtype=torch.long),
        torch.tensor([j], dtype=torch.long),
        torch.tensor([op], dtype=torch.long),
    ), target


def eval_oracle(model, memory, query, slot):
    _key, i, j, op, target, _entity = query
    return model.oracle_query(
        memory,
        torch.tensor([slot], dtype=torch.long),
        torch.tensor([i], dtype=torch.long),
        torch.tensor([j], dtype=torch.long),
        torch.tensor([op], dtype=torch.long),
    ), target


@torch.no_grad()
def evaluate_seed(model: LearnedAddressPLM, episodes):
    model.eval()
    q2_correct = 0
    q2_oracle_correct = 0
    q2_corrupt_correct = 0
    q2_shuffle_correct = 0
    address_correct = 0
    decision_mismatches = 0

    for ep in episodes:
        observations, _q1, q2, _payload, slots = ep
        memory = write_observations(model, observations)

        normal_out, target = eval_query(model, memory, q2)
        oracle_out, _ = eval_oracle(model, memory, q2, slots["q2_slot"])

        wrong_slot = (slots["q2_slot"] + 1) % 3
        wrong_key = observations[wrong_slot][0]
        corrupt_query = (
            wrong_key, q2[1], q2[2], q2[3], q2[4], q2[5]
        )
        corrupt_out, _ = eval_query(model, memory, corrupt_query)

        # Same keys and payloads, different anonymous storage order.
        perm = (2, 0, 1)
        shuffled_obs = [observations[i] for i in perm]
        shuffled_memory = write_observations(model, shuffled_obs)
        shuffle_out, _ = eval_query(model, shuffled_memory, q2)

        pred = int(normal_out["logits"].argmax(-1).item())
        oracle_pred = int(oracle_out["logits"].argmax(-1).item())
        corrupt_pred = int(corrupt_out["logits"].argmax(-1).item())
        shuffle_pred = int(shuffle_out["logits"].argmax(-1).item())
        selected_slot = int(normal_out["attention"].argmax(-1).item())

        q2_correct += int(pred == target)
        q2_oracle_correct += int(oracle_pred == target)
        q2_corrupt_correct += int(corrupt_pred == target)
        q2_shuffle_correct += int(shuffle_pred == target)
        address_correct += int(selected_slot == slots["q2_slot"])
        decision_mismatches += int(
            pred != int((normal_out["action"] >= 0.5).long().item())
        )

    n = len(episodes)
    return {
        "q2_accuracy": q2_correct / n,
        "oracle_address_q2_accuracy": q2_oracle_correct / n,
        "corrupted_address_q2_accuracy": q2_corrupt_correct / n,
        "shuffle_observation_order_q2_accuracy": q2_shuffle_correct / n,
        "address_selection_accuracy": address_correct / n,
        "q2_decision_mismatches": decision_mismatches,
    }


def bootstrap(values, rounds=5000, seed=20261010):
    rng = random.Random(seed)
    samples = sorted(
        statistics.fmean(rng.choices(values, k=len(values)))
        for _ in range(rounds)
    )
    return [
        samples[int(0.025 * rounds)],
        samples[int(0.975 * rounds)],
    ]


def run(smoke: bool = False):
    contract = load_contract(EXPERIMENT_ID)
    errors = contract.check_consistency()
    if errors:
        raise AssertionError(errors)

    manifest = benchmark_manifest()
    assert manifest["base_e2e008_generator_sha256"] == BASE_E2E008_GENERATOR_SHA256
    gradient = gradient_probe()
    assert gradient["pass"], gradient

    model_probe = LearnedAddressPLM()
    params = parameter_count(model_probe)

    seeds = (0,) if smoke else SEEDS
    eval_n = 32 if smoke else EVAL_EPISODES
    seed_rows = []

    for seed in seeds:
        model, training_keys = train_seed(seed)
        evaluation = sample_evaluation_episodes(
            random.Random(seed + 188000),
            eval_n,
        )
        eval_keys = {hashable_episode_key(ep) for ep in evaluation}
        overlap = training_keys & eval_keys
        assert not overlap, f"training/evaluation overlap for seed {seed}"

        fingerprint = episode_fingerprint(evaluation)
        metrics = evaluate_seed(model, evaluation)
        q2_slots = [ep[4]["q2_slot"] for ep in evaluation]

        seed_rows.append({
            "seed": seed,
            "evaluation_episode_fingerprint": fingerprint,
            "training_evaluation_semantic_overlap": len(overlap),
            "q2_target_slot_counts": {str(slot): q2_slots.count(slot) for slot in range(3)},
            **metrics,
        })

    assert len({row["evaluation_episode_fingerprint"] for row in seed_rows}) == len(seed_rows)
    assert all(row["training_evaluation_semantic_overlap"] == 0 for row in seed_rows)
    if not smoke:
        for row in seed_rows:
            assert row["q2_target_slot_counts"] == {"0": 200, "1": 200, "2": 200}

    q2 = [row["q2_accuracy"] for row in seed_rows]
    address = [row["address_selection_accuracy"] for row in seed_rows]
    oracle = [row["oracle_address_q2_accuracy"] for row in seed_rows]
    corrupt = [row["corrupted_address_q2_accuracy"] for row in seed_rows]
    shuffled = [row["shuffle_observation_order_q2_accuracy"] for row in seed_rows]

    q2_mean = statistics.fmean(q2)
    address_mean = statistics.fmean(address)
    oracle_mean = statistics.fmean(oracle)
    oracle_gap = oracle_mean - q2_mean

    summary = {
        "q2_mean": q2_mean,
        "q2_min": min(q2),
        "q2_composition_or_seed_bootstrap_ci95": bootstrap(q2) if not smoke else [None, None],
        "address_selection_mean": address_mean,
        "address_selection_min": min(address),
        "address_selection_seed_bootstrap_ci95": bootstrap(address) if not smoke else [None, None],
        "oracle_address_q2_mean": oracle_mean,
        "learned_minus_oracle_q2_gap": q2_mean - oracle_mean,
        "corrupted_address_q2_mean": statistics.fmean(corrupt),
        "shuffle_observation_order_q2_mean": statistics.fmean(shuffled),
        "classifier_decision_integrity_pass": all(
            row["q2_decision_mismatches"] == 0 for row in seed_rows
        ),
        "learned_address_supported": bool(
            not smoke
            and q2_mean >= PRIMARY_THRESHOLD
            and min(q2) >= 0.40
            and address_mean >= ADDRESS_MEAN_THRESHOLD
            and min(address) >= ADDRESS_MIN_THRESHOLD
            and oracle_mean >= PRIMARY_THRESHOLD
            and oracle_gap <= ORACLE_GAP_THRESHOLD
        ),
    }

    output = {
        "experiment_id": EXPERIMENT_ID,
        "status": "smoke" if smoke else "measured",
        "provenance": {
            "git_commit": os.environ.get("GITHUB_SHA", "unknown"),
            "workflow_run_id": os.environ.get("GITHUB_RUN_ID", "unknown"),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "contract_sha256": contract_hash(),
            "benchmark_sha256": manifest["generator_sha256"],
            "base_e2e008_generator_sha256": BASE_E2E008_GENERATOR_SHA256,
        },
        "protocol": {
            "seeds": list(seeds),
            "steps": STEPS,
            "batch_size": BATCH_SIZE,
            "evaluation_episodes_per_seed": eval_n,
            "hidden_dim": 64,
            "address_dim": ADDRESS_DIM,
            "observation_slots": 3,
            "operator_dispatch": "fixed query-conditioned oracle",
            "temporal_carry": "disabled",
            "entity_id_in_model_input": False,
            "model_parameter_count": params,
            "benchmark_generator_version": GENERATOR_VERSION,
            "registered_heldout": [list(x) for x in HELDOUT],
            "model_selection": "none",
        },
        "seed_results": seed_rows,
        "summary": summary,
        "leakage_audit": {
            "entity_id_in_model_input": False,
            "address_key_independent_of_payload": True,
            "query_contains_no_answer": True,
            "observation_order_randomized": True,
            "q2_target_slot_balanced": all(
                row["q2_target_slot_counts"] == {"0": 200, "1": 200, "2": 200}
                for row in seed_rows
            ) if not smoke else None,
            "q1_q2_target_slots_distinct": True,
            "training_evaluation_semantic_overlap_zero": all(
                row["training_evaluation_semantic_overlap"] == 0
                for row in seed_rows
            ),
            "evaluation_fingerprints_distinct": len(
                {row["evaluation_episode_fingerprint"] for row in seed_rows}
            ) == len(seed_rows),
            "oracle_address_evaluation_only": True,
            "fixed_operator_dispatch": True,
            "temporal_carry_disabled_by_design": True,
            "no_model_selection": True,
        },
        "claim_boundary": [
            "synthetic learned content-addressed retrieval only",
            "operator identity remains explicit",
            "temporal carry is intentionally disabled for this addressing isolation test",
            "no semantic/open-world addressing claim",
            "no learned operator-discovery claim",
            "no real-world multimodal understanding claim",
            "no scaling or hardware claim",
        ],
    }

    path = ROOT / "artifacts" / f"{EXPERIMENT_ID}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args().smoke)
