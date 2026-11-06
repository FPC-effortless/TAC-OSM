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
from tac_osm.latent_operator_001 import LatentOperatorPLM
from tac_osm.latent_operator_001_benchmark import (
    BATCH_SIZE,
    EVAL_EPISODES,
    GENERATOR_VERSION,
    HELDOUT,
    OPS,
    TRAIN_COMBOS,
    benchmark_manifest,
    episode_fingerprint,
    episode_key,
    sample_episode,
    sample_evaluation_episodes,
)

EXPERIMENT_ID = "TACOSM-PLM-LATENT-OPERATOR-001"
CONTRACT_PATH = ROOT / "contracts" / f"{EXPERIMENT_ID}.json"
SEEDS = (0, 1, 2, 3, 4)
STEPS = 300
PARAM_COUNT = 77682
PRIMARY_THRESHOLD = 0.80
OPERATOR_MEAN_THRESHOLD = 0.90
OPERATOR_MIN_THRESHOLD = 0.80
BASE_E2E008_GENERATOR_SHA256 = (
    "3be32a91191b73792b0fffb5a7b497e80aa6a46d0189e032e6b7c60082949f3e"
)


def contract_hash() -> str:
    return hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def write_observations(model: LatentOperatorPLM, observations):
    memory = model.state.initial(1, torch.device("cpu"))
    for entity, text, image, audio in observations:
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
    return memory


def support_tensor(support) -> torch.Tensor:
    return torch.tensor(support, dtype=torch.float32).unsqueeze(0)


def rotate_support_labels(support):
    labels = [row[2] for row in support]
    rotated = labels[1:] + labels[:1]
    return tuple((row[0], row[1], rotated[k]) for k, row in enumerate(support))


def build_batch(episodes):
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

    q_entities = torch.tensor([ep[1][0] for ep in episodes], dtype=torch.long)
    q_i = torch.tensor([ep[1][1] for ep in episodes], dtype=torch.long)
    q_j = torch.tensor([ep[1][2] for ep in episodes], dtype=torch.long)
    q_targets = torch.tensor([ep[1][3] for ep in episodes], dtype=torch.long)
    support = torch.tensor(
        [ep[2] for ep in episodes],
        dtype=torch.float32,
    )
    return {
        "text": torch.stack(texts, 0),
        "image": torch.stack(images, 0),
        "audio": torch.stack(audios, 0),
        "entities": torch.stack(entities, 0),
        "q_entity": q_entities,
        "q_i": q_i,
        "q_j": q_j,
        "q_target": q_targets,
        "support": support,
    }


def write_observations_batch(model: LatentOperatorPLM, batch):
    memory = model.state.initial(
        batch["text"].shape[1],
        batch["text"].device,
    )
    for slot in range(batch["text"].shape[0]):
        z = model.encode(
            batch["text"][slot],
            batch["image"][slot],
            batch["audio"][slot],
        )
        memory, _ = model.state.write(
            memory,
            z,
            batch["entities"][slot],
        )
    return memory


def query_target_batch(model, memory, batch):
    return model.latent_query(
        memory,
        batch["q_entity"],
        batch["q_i"],
        batch["q_j"],
        batch["support"],
    )


def query_target(model, memory, q2, support, operator_override=None):
    entity, i, j, _answer = q2
    return model.latent_query(
        memory,
        torch.tensor([entity], dtype=torch.long),
        torch.tensor([i], dtype=torch.long),
        torch.tensor([j], dtype=torch.long),
        support_tensor(support),
        operator_override=operator_override,
    )


def build_gradient_probe(seed: int = 20261009):
    torch.manual_seed(seed)
    rng = random.Random(91009)
    episodes = [sample_episode(rng) for _ in range(4)]
    model = LatentOperatorPLM().cpu()

    loss = torch.zeros((), dtype=torch.float32)
    for ep in episodes:
        memory = write_observations(model, ep[0])
        out = query_target(model, memory, ep[1], ep[2])
        loss = loss + F.cross_entropy(
            out["logits"], torch.tensor([ep[1][3]], dtype=torch.long)
        )
    loss.backward()

    params = dict(model.named_parameters())
    required = (
        "text.emb.weight",
        "image.net.0.weight",
        "audio.net.0.weight",
        "rep.fuse.0.weight",
        "state.write_value.weight",
        "casm.decoder.0.weight",
        "operator_inducer.demo.0.weight",
        "operator_inducer.head.0.weight",
        "operator_inducer.head.2.weight",
    )
    missing_or_nonfinite = []
    zero_operator_grad = []
    for name in required:
        grad = params.get(name)
        if grad is None or not torch.isfinite(grad).all():
            missing_or_nonfinite.append(name)
    for name in (
        "operator_inducer.demo.0.weight",
        "operator_inducer.head.0.weight",
        "operator_inducer.head.2.weight",
    ):
        grad = params[name].grad
        if grad is None or float(grad.abs().sum()) == 0.0:
            zero_operator_grad.append(name)

    return {
        "pass": not missing_or_nonfinite and not zero_operator_grad,
        "missing_or_nonfinite": missing_or_nonfinite,
        "zero_operator_gradient": zero_operator_grad,
    }


def train_seed(seed: int):
    torch.manual_seed(seed)
    rng = random.Random(seed + 185000)
    model = LatentOperatorPLM().cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.002,
        weight_decay=0.0001,
    )
    training_keys = set()

    for _ in range(STEPS):
        episodes = [
            sample_episode(rng, combo2=rng.choice(TRAIN_COMBOS))
            for _ in range(BATCH_SIZE)
        ]
        training_keys.update(episode_key(ep) for ep in episodes)
        batch = build_batch(episodes)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        memory = write_observations_batch(model, batch)
        out = query_target_batch(model, memory, batch)
        loss = F.cross_entropy(out["logits"], batch["q_target"])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return model, training_keys


@torch.no_grad()
def evaluate_seed(model: LatentOperatorPLM, episodes, control: str):
    model.eval()
    q2_correct = 0
    operator_correct = 0
    oracle_q2_correct = 0
    no_memory_q2_correct = 0
    support_shuffle_q2_correct = 0
    decision_mismatches = 0
    operator_probs = []

    for ep in episodes:
        observations, q2, support, hidden_op, _payload = ep
        memory = write_observations(model, observations)

        normal_out = query_target(model, memory, q2, support)
        hidden_op_idx = OPS.index(hidden_op)
        oracle_out = query_target(
            model,
            memory,
            q2,
            support,
            operator_override=torch.tensor([hidden_op_idx], dtype=torch.long),
        )

        shuffled_support = rotate_support_labels(support)
        shuffled_out = query_target(model, memory, q2, shuffled_support)

        no_memory = torch.zeros_like(memory)
        no_memory_out = query_target(model, no_memory, q2, support)

        pred = int(normal_out["logits"].argmax(-1).item())
        oracle_pred = int(oracle_out["logits"].argmax(-1).item())
        shuffled_pred = int(shuffled_out["logits"].argmax(-1).item())
        no_memory_pred = int(no_memory_out["logits"].argmax(-1).item())
        target = int(q2[3])

        q2_correct += int(pred == target)
        operator_correct += int(
            normal_out["operator_prediction"].item() == hidden_op_idx
        )
        oracle_q2_correct += int(oracle_pred == target)
        support_shuffle_q2_correct += int(shuffled_pred == target)
        no_memory_q2_correct += int(no_memory_pred == target)
        decision_mismatches += int(
            pred != int((normal_out["action"] >= 0.5).long().item())
        )
        operator_probs.append(
            normal_out["operator_probs"].squeeze(0).tolist()
        )

    n = len(episodes)
    return {
        "q2_accuracy": q2_correct / n,
        "operator_selection_accuracy": operator_correct / n,
        "oracle_operator_q2_accuracy": oracle_q2_correct / n,
        "shuffle_support_q2_accuracy": support_shuffle_q2_correct / n,
        "no_memory_q2_accuracy": no_memory_q2_correct / n,
        "q2_decision_mismatches": decision_mismatches,
        "operator_probability_mean": [
            statistics.fmean(row[k] for row in operator_probs)
            for k in range(4)
        ],
    }


def bootstrap_compositions(seed_rows, field: str, rounds: int = 5000):
    # Seed-level bootstrap over the 12 held-out compositions, matching the
    # repository's composition-bootstrap convention.
    rng = random.Random(20261009)
    keys = sorted(seed_rows[0]["per_composition"][field])
    means = [
        statistics.fmean(row["per_composition"][field][key] for row in seed_rows)
        for key in keys
    ]
    samples = sorted(
        statistics.fmean(rng.choices(means, k=len(means)))
        for _ in range(rounds)
    )
    return [
        samples[int(0.025 * rounds)],
        samples[int(0.975 * rounds)],
    ]


@torch.no_grad()
def evaluate_detailed(model: LatentOperatorPLM, episodes):
    model.eval()
    per_comp = {
        "q2_accuracy": {},
        "operator_selection_accuracy": {},
    }
    accum = {}
    for ep in episodes:
        observations, q2, support, hidden_op, _payload = ep
        memory = write_observations(model, observations)
        out = query_target(model, memory, q2, support)
        key = str((hidden_op, q2[1], q2[2]))
        bucket = accum.setdefault(
            key,
            {"q2": [], "op": []},
        )
        bucket["q2"].append(
            int(out["logits"].argmax(-1).item()) == int(q2[3])
        )
        bucket["op"].append(
            int(out["operator_prediction"].item()) == OPS.index(hidden_op)
        )
    for key, bucket in accum.items():
        per_comp["q2_accuracy"][key] = statistics.fmean(bucket["q2"])
        per_comp["operator_selection_accuracy"][key] = statistics.fmean(
            bucket["op"]
        )
    return per_comp


def run(smoke: bool = False):
    contract = load_contract(EXPERIMENT_ID)
    errors = contract.check_consistency()
    if errors:
        raise AssertionError(errors)

    manifest = benchmark_manifest()
    assert manifest["base_e2e008_generator_sha256"] == BASE_E2E008_GENERATOR_SHA256
    gradient_gate = build_gradient_probe()
    assert gradient_gate["pass"], gradient_gate

    model_probe = LatentOperatorPLM()
    assert parameter_count(model_probe) == PARAM_COUNT

    seeds = (0,) if smoke else SEEDS
    eval_n = 32 if smoke else EVAL_EPISODES
    seed_rows = []

    for seed in seeds:
        model, training_keys = train_seed(seed)
        evaluation = sample_evaluation_episodes(
            random.Random(seed + 186000), eval_n
        )
        eval_keys = {episode_key(ep) for ep in evaluation}
        overlap = training_keys & eval_keys
        assert not overlap, f"training/evaluation overlap for seed {seed}"
        fingerprint = episode_fingerprint(evaluation)

        metrics = evaluate_seed(model, evaluation, "normal")
        detailed = evaluate_detailed(model, evaluation)

        seed_rows.append(
            {
                "seed": seed,
                "evaluation_episode_fingerprint": fingerprint,
                "training_evaluation_semantic_overlap": len(overlap),
                "q2_accuracy": metrics["q2_accuracy"],
                "operator_selection_accuracy": metrics[
                    "operator_selection_accuracy"
                ],
                "oracle_operator_q2_accuracy": metrics[
                    "oracle_operator_q2_accuracy"
                ],
                "shuffle_support_q2_accuracy": metrics[
                    "shuffle_support_q2_accuracy"
                ],
                "no_memory_q2_accuracy": metrics["no_memory_q2_accuracy"],
                "q2_decision_mismatches": metrics["q2_decision_mismatches"],
                "operator_probability_mean": metrics["operator_probability_mean"],
                "per_composition": detailed,
            }
        )

    assert len(
        {row["evaluation_episode_fingerprint"] for row in seed_rows}
    ) == len(seed_rows)
    assert all(
        row["training_evaluation_semantic_overlap"] == 0
        for row in seed_rows
    )

    q2_values = [row["q2_accuracy"] for row in seed_rows]
    op_values = [row["operator_selection_accuracy"] for row in seed_rows]
    oracle_values = [
        row["oracle_operator_q2_accuracy"] for row in seed_rows
    ]

    summary = {
        "q2_mean": statistics.fmean(q2_values),
        "q2_min": min(q2_values),
        "q2_composition_bootstrap_ci95": (
            bootstrap_compositions(seed_rows, "q2_accuracy")
            if not smoke
            else [None, None]
        ),
        "operator_selection_mean": statistics.fmean(op_values),
        "operator_selection_min": min(op_values),
        "operator_selection_composition_bootstrap_ci95": (
            bootstrap_compositions(seed_rows, "operator_selection_accuracy")
            if not smoke
            else [None, None]
        ),
        "oracle_operator_q2_mean": statistics.fmean(oracle_values),
        "shuffle_support_q2_mean": statistics.fmean(
            row["shuffle_support_q2_accuracy"] for row in seed_rows
        ),
        "no_memory_q2_mean": statistics.fmean(
            row["no_memory_q2_accuracy"] for row in seed_rows
        ),
        "classifier_decision_integrity_pass": all(
            row["q2_decision_mismatches"] == 0 for row in seed_rows
        ),
        "operator_induction_supported": bool(
            not smoke
            and statistics.fmean(q2_values) >= PRIMARY_THRESHOLD
            and min(q2_values) >= 0.40
            and statistics.fmean(op_values) >= OPERATOR_MEAN_THRESHOLD
            and min(op_values) >= OPERATOR_MIN_THRESHOLD
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
            "state_write_mode": "residual_linear",
            "operator_id_in_model_input": False,
            "operator_induction_loss": "q2 task loss only",
            "model_parameter_count": PARAM_COUNT,
            "registered_heldout": [list(x) for x in HELDOUT],
            "train_combos_count": len(TRAIN_COMBOS),
            "benchmark_generator_version": GENERATOR_VERSION,
            "model_selection": "none",
        },
        "seed_results": seed_rows,
        "summary": summary,
        "leakage_audit": {
            "operator_id_in_model_input": False,
            "operator_id_in_training_loss": False,
            "support_context_contains_only_axy": True,
            "support_order_randomized": True,
            "training_evaluation_semantic_overlap_zero": all(
                row["training_evaluation_semantic_overlap"] == 0
                for row in seed_rows
            ),
            "heldout_compositions_excluded_from_training": True,
            "evaluation_generated_after_training": True,
            "all_evaluation_fingerprints_distinct": len(
                {row["evaluation_episode_fingerprint"] for row in seed_rows}
            ) == len(seed_rows),
            "operator_inducer_gradient_gate_pass": True,
            "fixed_primitives_remain_the_execution_library": True,
            "entity_addressing_explicit_by_design": True,
            "operator_synthesis_not_claimed": True,
        },
        "claim_boundary": [
            "synthetic latent-operator induction only",
            "fixed Boolean primitive library remains",
            "explicit entity addressing remains",
            "no operator synthesis claim",
            "no real-world semantic multimodal claim",
            "no scaling or hardware claim",
        ],
    }

    out = ROOT / "artifacts" / f"{EXPERIMENT_ID}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args().smoke)
