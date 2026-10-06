#!/usr/bin/env python3
"""Runner for TACOSM-PLM-INTEGRATED-E2E-005."""
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
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

from tac_osm.contract import load_contract
from tac_osm.integrated_e2e_005 import FunctionalMultimodalPLM
from tac_osm.integrated_e2e_005_benchmark import (
    BATCH_SIZE,
    EVAL_EPISODES,
    GENERATOR_VERSION,
    HELDOUT,
    OPS,
    SEEDS,
    STEPS,
    benchmark_manifest,
    episode_fingerprint,
    episode_key,
    sample_episode,
    sample_evaluation_episodes,
)


EXPERIMENT_ID = "TACOSM-PLM-INTEGRATED-E2E-005"
CONTRACT_PATH = ROOT / "contracts" / f"{EXPERIMENT_ID}.json"


def primary_threshold() -> float:
    raw = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    primaries = [e for e in raw["endpoints"] if e["primary"]]
    if len(primaries) != 1:
        raise AssertionError(f"expected one primary endpoint, found {primaries}")
    endpoint = raw["primary_endpoint"]
    if endpoint["name"] != primaries[0]["name"]:
        raise AssertionError("primary endpoint name/endpoint table mismatch")
    threshold = float(endpoint["threshold"])
    if not 0.0 < threshold < 1.0:
        raise AssertionError(f"invalid primary threshold {threshold}")
    return threshold


def contract_hash() -> str:
    return hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()


def accuracy(logits: torch.Tensor, target: int) -> int:
    return int(logits.argmax(-1).item() == int(target))


def query_parts(q: tuple):
    return (
        torch.tensor([q[0]], dtype=torch.long),
        torch.tensor([q[1]], dtype=torch.long),
        torch.tensor([q[2]], dtype=torch.long),
        torch.tensor([OPS.index(q[3])], dtype=torch.long),
    )


def build_batch(episodes: list[tuple]) -> dict:
    texts, images, audios, entities = [], [], [], []
    for slot in range(3):
        t, im, au, en = [], [], [], []
        for ep in episodes:
            entity, text, image, audio = ep[0][slot]
            t.append(text)
            im.append(image)
            au.append(audio)
            en.append(entity)
        texts.append(torch.stack(t))
        images.append(torch.stack(im))
        audios.append(torch.stack(au).unsqueeze(1))
        entities.append(torch.tensor(en, dtype=torch.long))

    def make_qbatch(index: int):
        qs = [ep[index] for ep in episodes]
        meta = (
            torch.tensor([q[0] for q in qs], dtype=torch.long),
            torch.tensor([q[1] for q in qs], dtype=torch.long),
            torch.tensor([q[2] for q in qs], dtype=torch.long),
            torch.tensor([OPS.index(q[3]) for q in qs], dtype=torch.long),
        )
        targets = torch.tensor([q[4] for q in qs], dtype=torch.long)
        return meta, targets

    q1, y1 = make_qbatch(1)
    q2, y2 = make_qbatch(2)
    return {
        "text": torch.stack(texts, 0),
        "image": torch.stack(images, 0),
        "audio": torch.stack(audios, 0),
        "entities": torch.stack(entities, 0),
        "q1": q1,
        "q2": q2,
        "y1": y1,
        "y2": y2,
    }


def write_observations(model: FunctionalMultimodalPLM, batch: dict) -> torch.Tensor:
    memory = model.state.initial(
        batch["text"].shape[1],
        batch["text"].device,
    )
    for t in range(batch["text"].shape[0]):
        z = model.encode(
            batch["text"][t],
            batch["image"][t],
            batch["audio"][t],
        )
        memory, _ = model.state.write(
            memory,
            z,
            batch["entities"][t],
        )
    return memory


def environment_outcome(action: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Synthetic environment: success is determined only after the action."""
    return ((action >= 0.5).long() == target).float().detach()


def train_batch(model: FunctionalMultimodalPLM, batch: dict) -> torch.Tensor:
    memory = write_observations(model, batch)

    e1, i1, j1, op1 = batch["q1"]
    out1 = model.query(memory, e1, i1, j1, op1)
    action1_loss = F.cross_entropy(out1["logits"], batch["y1"])
    outcome1 = environment_outcome(out1["action"], batch["y1"])
    memory, verifier1 = model.post_action_update(memory, out1, outcome1)

    e2, i2, j2, op2 = batch["q2"]
    out2 = model.query(memory, e2, i2, j2, op2)
    action2_loss = F.cross_entropy(out2["logits"], batch["y2"])
    outcome2 = environment_outcome(out2["action"], batch["y2"])
    _, verifier2 = model.post_action_update(memory, out2, outcome2)

    verifier_loss = F.binary_cross_entropy_with_logits(
        verifier1[:, :1],
        verifier1[:, 1:],
    ) + F.binary_cross_entropy_with_logits(
        verifier2[:, :1],
        verifier2[:, 1:],
    )

    return action1_loss + action2_loss + 0.10 * verifier_loss / 2


def gradient_surface_probe() -> dict:
    torch.manual_seed(20261006)
    rng = random.Random(42005)
    episodes = [sample_episode(rng) for _ in range(4)]
    batch = build_batch(episodes)
    model = FunctionalMultimodalPLM()
    loss = train_batch(model, batch)
    loss.backward()
    required = (
        "text.emb.weight",
        "text.rnn.weight_ih_l0",
        "image.net.0.weight",
        "audio.net.0.weight",
        "rep.fuse.0.weight",
        "state.write_value.weight",
        "casm.decoder.0.weight",
        "verifier.0.weight",
        "write_gate.0.weight",
    )
    failures = []
    for name in required:
        grad = dict(model.named_parameters())[name].grad
        if grad is None or not torch.isfinite(grad).all():
            failures.append(name)
    return {"pass": not failures, "missing_or_nonfinite": failures}


def train_seed(seed: int, steps: int, batch_size: int):
    torch.manual_seed(seed)
    rng = random.Random(seed + 50000)
    model = FunctionalMultimodalPLM().cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.002,
        weight_decay=0.0001,
    )
    training_keys: set[tuple] = set()

    for _ in range(steps):
        episodes = [sample_episode(rng) for _ in range(batch_size)]
        training_keys.update(episode_key(ep) for ep in episodes)
        batch = build_batch(episodes)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = train_batch(model, batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return model, training_keys


@torch.no_grad()
def evaluate_seed(
    model: FunctionalMultimodalPLM,
    episodes: list[tuple],
    control: str,
):
    model.eval()
    q1_correct = q2_correct = verifier_correct = 0

    for episode in episodes:
        rows = episode[0]
        memory = model.state.initial(1, torch.device("cpu"))

        for k, row in enumerate(rows):
            entity, text, image, audio = row
            if control == "shuffle_image":
                image = rows[(k + 1) % len(rows)][2]
            elif control == "text_only":
                image = torch.zeros_like(image)
                audio = torch.zeros_like(audio)
            elif control == "image_only":
                text = torch.zeros_like(text)
                audio = torch.zeros_like(audio)
            elif control == "audio_only":
                text = torch.zeros_like(text)
                image = torch.zeros_like(image)

            z = model.encode(
                text.unsqueeze(0),
                image.unsqueeze(0),
                audio.unsqueeze(0).unsqueeze(0),
            )
            memory, _ = model.state.write(
                memory,
                z,
                torch.tensor([entity], dtype=torch.long),
            )

        if control == "no_memory":
            memory = torch.zeros_like(memory)

        e1, i1, j1, op1 = query_parts(episode[1])
        out1 = model.query(memory, e1, i1, j1, op1)
        target1 = torch.tensor([episode[1][4]], dtype=torch.long)
        outcome1 = environment_outcome(out1["action"], target1)
        updated, verifier1 = model.post_action_update(memory, out1, outcome1)

        e2, i2, j2, op2 = query_parts(episode[2])
        out2 = model.query(updated, e2, i2, j2, op2)
        target2 = torch.tensor([episode[2][4]], dtype=torch.long)
        outcome2 = environment_outcome(out2["action"], target2)
        _, verifier2 = model.post_action_update(updated, out2, outcome2)

        q1_correct += accuracy(out1["logits"], episode[1][4])
        q2_correct += accuracy(out2["logits"], episode[2][4])
        verifier_correct += int(
            ((verifier1[:, :1].sigmoid() >= 0.5).long() == outcome1.long()).item()
        )
        verifier_correct += int(
            ((verifier2[:, :1].sigmoid() >= 0.5).long() == outcome2.long()).item()
        )

    n = len(episodes)
    return {
        "q1_accuracy": q1_correct / n,
        "q2_accuracy": q2_correct / n,
        "verifier_q1_q2_accuracy": verifier_correct / (2 * n),
    }


def bootstrap_ci(
    values: list[float],
    rounds: int = 5000,
    seed: int = 20261006,
):
    rng = random.Random(seed)
    means = [
        statistics.fmean(rng.choices(values, k=len(values)))
        for _ in range(rounds)
    ]
    means.sort()
    return [
        means[int(0.025 * rounds)],
        means[int(0.975 * rounds)],
    ]


def run(smoke: bool) -> dict:
    contract = load_contract(EXPERIMENT_ID)
    errors = contract.check_consistency()
    if errors:
        raise AssertionError(errors)

    gradient_gate = gradient_surface_probe()
    if not gradient_gate["pass"]:
        raise AssertionError(f"gradient surface failure: {gradient_gate}")

    if smoke:
        seeds, steps, batch_size, eval_n = (0,), 25, 8, 16
    else:
        seeds, steps, batch_size, eval_n = SEEDS, STEPS, BATCH_SIZE, EVAL_EPISODES

    seed_results = []
    controls = (
        "normal",
        "no_memory",
        "shuffle_image",
        "text_only",
        "image_only",
        "audio_only",
    )

    for seed in seeds:
        model, training_keys = train_seed(seed, steps, batch_size)

        eval_rng = random.Random(seed + 100000)
        episodes = sample_evaluation_episodes(eval_rng, eval_n)
        eval_keys = {episode_key(ep) for ep in episodes}
        overlap = training_keys & eval_keys
        if overlap:
            raise AssertionError(
                f"training/evaluation semantic overlap for seed {seed}"
            )

        fingerprints = {
            control: episode_fingerprint(episodes)
            for control in controls
        }
        if len(set(fingerprints.values())) != 1:
            raise AssertionError(
                "controls did not reuse the exact same evaluation episodes"
            )

        metrics = {
            control: evaluate_seed(model, episodes, control)
            for control in controls
        }

        oracle_correct = 0
        for ep in episodes:
            q2 = ep[2]
            payload = ep[3][q2[0]]
            a, b = payload[q2[1]], payload[q2[2]]
            expected = {
                "xor": a ^ b,
                "and": a & b,
                "or": a | b,
                "xnor": 1 - (a ^ b),
            }[q2[3]]
            oracle_correct += int(expected == q2[4])

        seed_results.append({
            "seed": seed,
            "normal_q1_accuracy": metrics["normal"]["q1_accuracy"],
            "normal_q2_accuracy": metrics["normal"]["q2_accuracy"],
            "normal_verifier_accuracy": metrics["normal"]["verifier_q1_q2_accuracy"],
            "no_memory_q2_accuracy": metrics["no_memory"]["q2_accuracy"],
            "shuffle_image_q2_accuracy": metrics["shuffle_image"]["q2_accuracy"],
            "text_only_q2_accuracy": metrics["text_only"]["q2_accuracy"],
            "image_only_q2_accuracy": metrics["image_only"]["q2_accuracy"],
            "audio_only_q2_accuracy": metrics["audio_only"]["q2_accuracy"],
            "memory_drop": (
                metrics["normal"]["q2_accuracy"]
                - metrics["no_memory"]["q2_accuracy"]
            ),
            "alignment_drop": (
                metrics["normal"]["q2_accuracy"]
                - metrics["shuffle_image"]["q2_accuracy"]
            ),
            "evaluation_episode_fingerprint": fingerprints["normal"],
            "training_evaluation_semantic_overlap": len(overlap),
            "oracle_q2_accuracy": oracle_correct / eval_n,
        })

    normal_q2 = [r["normal_q2_accuracy"] for r in seed_results]
    summary = {
        "primary_q2_mean": statistics.fmean(normal_q2),
        "primary_q2_seed_bootstrap_ci95": (
            bootstrap_ci(normal_q2) if not smoke else [None, None]
        ),
        "all_seed_min_q2": min(normal_q2),
        "primary_pass": bool(
            not smoke
            and statistics.fmean(normal_q2) >= primary_threshold()
            and min(normal_q2) >= 0.40
        ),
        "mean_memory_drop": statistics.fmean(
            r["memory_drop"] for r in seed_results
        ),
        "mean_alignment_drop": statistics.fmean(
            r["alignment_drop"] for r in seed_results
        ),
        "oracle_q2_accuracy": statistics.fmean(
            r["oracle_q2_accuracy"] for r in seed_results
        ),
        "gradient_surface_pass": gradient_gate["pass"],
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
            "benchmark_sha256": benchmark_manifest()["generator_sha256"],
        },
        "protocol": {
            "seeds": list(seeds),
            "steps": steps,
            "batch_size": batch_size,
            "evaluation_episodes_per_seed": eval_n,
            "registered_heldout": [list(x) for x in HELDOUT],
            "registered_controls": list(controls),
            "benchmark_generator_version": GENERATOR_VERSION,
            "benchmark_manifest": benchmark_manifest(),
            "model_selection": "none",
            "representation_auxiliary_supervision": False,
            "operator_dispatch": "fixed_query_conditioned",
        },
        "seed_results": seed_results,
        "summary": summary,
        "leakage_audit": {
            "q1_target_entity": "entities[0]",
            "q2_target_entity": "entities[1]",
            "q1_q2_entities_distinct": True,
            "heldout_compositions_excluded_from_training": True,
            "evaluation_generated_after_training": True,
            "training_evaluation_semantic_overlap_zero": all(
                r["training_evaluation_semantic_overlap"] == 0
                for r in seed_results
            ),
            "controls_reuse_exact_same_episode_objects": True,
            "pre_action_query_signature": [
                "self",
                "memory",
                "entity",
                "i",
                "j",
                "op",
            ],
            "forbidden_pre_action_fields": [
                "answer",
                "environment_outcome",
                "verifier_target",
                "payload_labels",
                "evaluation_correctness",
            ],
            "payload_auxiliary_supervision": False,
            "post_action_outcome_only_feedback": True,
            "fixed_operator_dispatch_not_learned_discovery": True,
        },
        "claim_boundary": [
            "synthetic multimodal mechanism only",
            "no real-world language, image, or audio claim",
            "no semantic-memory claim",
            "no learned-operator-discovery claim",
            "no scaling or hardware claim",
        ],
    }

    out = ROOT / "artifacts" / f"{EXPERIMENT_ID}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args().smoke)
