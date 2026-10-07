#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tac_osm.contract import load_contract
from tac_osm.integrated_e2e_005 import FunctionalConfig, FunctionalMultimodalPLM
from tac_osm.integrated_e2e_008_benchmark import (
    EVAL_EPISODES,
    GENERATOR_VERSION,
    HELDOUT,
    OPS,
    episode_fingerprint,
    episode_key,
    sample_evaluation_episodes,
    generator_hash,
)
from scripts.run_integrated_e2e_008 import train_seed

EXPERIMENT_ID = "TACOSM-PLM-PERSISTENCE-RESET-RECOMPUTE-001"
REFERENCE_EXPERIMENT_ID = "TACOSM-PLM-INTEGRATED-E2E-008"
CONTRACT_PATH = ROOT / "contracts" / f"{EXPERIMENT_ID}.json"

SEEDS = (0, 1, 2, 3, 4)
STEPS = 300
BATCH_SIZE = 96
EVAL_N = EVAL_EPISODES
PARAM_COUNT = 74126
GAP_THRESHOLD = 0.05
REFERENCE_FINGERPRINTS = {
    0: "1ddec9fce4808e697c615a4934a0650d763ad2a3d87b0d8b059546a414b21f51",
    1: "ada3f705647de795a930dda2439ae6456662ebda498cca5458e704ea338cc2eb",
    2: "bbe54208c823d659b729b49b28f05c76b4fd5008d26e1183d8030dea06ae2fd8",
    3: "980b8b97f9a5cc816d2bee9092dac27fbf2e233aabf81a809e5392bc1c8d860a",
    4: "4984f06784cd176f2a2d517ab04fa1fa82f288f12c68c783caad9b838398259c",
}
REFERENCE_BENCHMARK_SHA256 = "3be32a91191b73792b0fffb5a7b497e80aa6a46d0189e032e6b7c60082949f3e"


def contract_hash() -> str:
    return hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()


def parameter_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def write_memory(model: FunctionalMultimodalPLM, rows) -> torch.Tensor:
    """Reconstruct state using observations only.

    This function is intentionally forbidden from accepting q1 action,
    outcome, target, verifier signal, or correctness. Its only inputs are
    frozen model parameters and the original multimodal observation rows.
    """
    memory = model.state.initial(1, torch.device("cpu"))
    for entity, text, image, audio in rows:
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


def query(model: FunctionalMultimodalPLM, memory: torch.Tensor, q: tuple):
    return model.query(
        memory,
        torch.tensor([q[0]], dtype=torch.long),
        torch.tensor([q[1]], dtype=torch.long),
        torch.tensor([q[2]], dtype=torch.long),
        torch.tensor([OPS.index(q[3])], dtype=torch.long),
    )


def environment_outcome(action: torch.Tensor, target: int) -> torch.Tensor:
    return ((action >= 0.5).long() == torch.tensor([target])).float().detach()


@torch.no_grad()
def evaluate_pair(model: FunctionalMultimodalPLM, episodes: list[tuple]):
    model.eval()
    normal_q2 = reset_q2 = 0
    normal_q1 = reset_q1 = 0
    normal_positive = reset_positive = 0
    q1_mismatches = 0
    recompute_identity_failures = 0
    per_comp = defaultdict(lambda: {"normal": [0, 0], "reset": [0, 0]})

    for ep in episodes:
        rows, q1, q2, _payload = ep

        pre_q1_normal = write_memory(model, rows)
        out1_normal = query(model, pre_q1_normal, q1)
        target1 = q1[4]
        action1_normal = (out1_normal["action"] >= 0.5).long()
        outcome1_normal = environment_outcome(action1_normal, target1)
        updated_normal, _ = model.post_action_update(
            pre_q1_normal, out1_normal, outcome1_normal
        )
        out2_normal = query(model, updated_normal, q2)

        # Recompute arm: the same q1 boundary is traversed, but the
        # outcome-gated state is deliberately discarded. The recomputation
        # below receives only the original observations.
        pre_q1_reset = write_memory(model, rows)
        out1_reset = query(model, pre_q1_reset, q1)
        target1_reset = q1[4]
        action1_reset = (out1_reset["action"] >= 0.5).long()
        outcome1_reset = environment_outcome(action1_reset, target1_reset)
        _discarded_updated, _ = model.post_action_update(
            pre_q1_reset, out1_reset, outcome1_reset
        )

        recomputed = write_memory(model, rows)
        if not torch.equal(pre_q1_reset, recomputed):
            recompute_identity_failures += 1

        out2_reset = query(model, recomputed, q2)

        q1_pred_normal = int(out1_normal["logits"].argmax(-1).item())
        q1_pred_reset = int(out1_reset["logits"].argmax(-1).item())
        q1_action_normal = int(action1_normal.item())
        q1_action_reset = int(action1_reset.item())

        q1_mismatches += int(q1_pred_normal != q1_pred_reset)
        q1_mismatches += int(q1_action_normal != q1_action_reset)
        q1_mismatches += int(outcome1_normal.item() != outcome1_reset.item())

        q1_correct_normal = int(q1_pred_normal == target1)
        q1_correct_reset = int(q1_pred_reset == target1)
        q2_pred_normal = int(out2_normal["logits"].argmax(-1).item())
        q2_pred_reset = int(out2_reset["logits"].argmax(-1).item())

        normal_q1 += q1_correct_normal
        reset_q1 += q1_correct_reset
        normal_q2 += int(q2_pred_normal == q2[4])
        reset_q2 += int(q2_pred_reset == q2[4])
        normal_positive += q2_pred_normal
        reset_positive += q2_pred_reset

        key = str((q2[3], q2[1], q2[2]))
        per_comp[key]["normal"][0] += int(q2_pred_normal == q2[4])
        per_comp[key]["normal"][1] += 1
        per_comp[key]["reset"][0] += int(q2_pred_reset == q2[4])
        per_comp[key]["reset"][1] += 1

    n = len(episodes)
    return {
        "normal_q1_accuracy": normal_q1 / n,
        "reset_recompute_q1_accuracy": reset_q1 / n,
        "normal_q2_accuracy": normal_q2 / n,
        "reset_recompute_q2_accuracy": reset_q2 / n,
        "normal_q2_positive_fraction": normal_positive / n,
        "reset_recompute_q2_positive_fraction": reset_positive / n,
        "q1_intervention_mismatches": q1_mismatches,
        "recompute_identity_failures": recompute_identity_failures,
        "per_composition_q2_accuracy": {
            comp: {
                "normal": normal / count_n,
                "reset_recompute": reset / count_r,
            }
            for comp, data in per_comp.items()
            for normal, count_n in [data["normal"]]
            for reset, count_r in [data["reset"]]
        },
    }


def composition_bootstrap(seed_rows: list[dict], rounds: int = 5000, seed: int = 20261008):
    compositions = sorted(seed_rows[0]["per_composition_q2_accuracy"])
    differences = []
    for comp in compositions:
        normal = statistics.fmean(
            row["per_composition_q2_accuracy"][comp]["normal"] for row in seed_rows
        )
        reset = statistics.fmean(
            row["per_composition_q2_accuracy"][comp]["reset_recompute"] for row in seed_rows
        )
        differences.append(normal - reset)

    rng = random.Random(seed)
    samples = sorted(
        statistics.fmean(rng.choices(differences, k=len(differences)))
        for _ in range(rounds)
    )
    return {
        "point_estimate": statistics.fmean(differences),
        "ci95": [samples[int(0.025 * rounds)], samples[int(0.975 * rounds)]],
        "composition_differences": {
            comp: diff for comp, diff in zip(compositions, differences)
        },
    }


def reference_checks() -> dict:
    e2e = json.loads(
        (ROOT / "contracts" / "TACOSM-PLM-INTEGRATED-E2E-008.json").read_text(
            encoding="utf-8"
        )
    )
    assert e2e["experiment_id"] == REFERENCE_EXPERIMENT_ID
    assert e2e["protocol"]["hidden_dim"] == 64
    assert e2e["protocol"]["state_write_mode"] == "residual_linear"
    assert e2e["steps"] == STEPS
    assert e2e["eval_steps"] == EVAL_N
    assert [tuple(x) for x in e2e["heldout"]] == list(HELDOUT)
    assert e2e["status"] == "pre-registered"
    assert generator_hash() == REFERENCE_BENCHMARK_SHA256
    return {
        "reference_experiment_id": REFERENCE_EXPERIMENT_ID,
        "benchmark_sha256": generator_hash(),
        "benchmark_generator_version": GENERATOR_VERSION,
        "heldout": [list(x) for x in HELDOUT],
    }


def run(smoke: bool = False) -> dict:
    contract = load_contract(EXPERIMENT_ID)
    errors = contract.check_consistency()
    if errors:
        raise AssertionError(errors)
    ref = reference_checks()

    model_probe = FunctionalMultimodalPLM(
        config=FunctionalConfig(hidden_dim=64, state_write_mode="residual_linear")
    )
    assert parameter_count(model_probe) == PARAM_COUNT

    seeds = (0,) if smoke else SEEDS
    eval_n = 16 if smoke else EVAL_N
    rows = []

    for seed in seeds:
        model, training_keys = train_seed(seed)
        assert parameter_count(model) == PARAM_COUNT
        model.eval()

        evaluation = sample_evaluation_episodes(
            random.Random(seed + 181000), eval_n
        )
        eval_keys = {episode_key(ep) for ep in evaluation}
        overlap = training_keys & eval_keys
        assert not overlap, f"training/evaluation semantic overlap for seed {seed}"

        fingerprint = episode_fingerprint(evaluation)
        if not smoke:
            assert fingerprint == REFERENCE_FINGERPRINTS[seed], (
                f"seed {seed}: evaluation fingerprint differs from E2E-008 "
                f"reference: got {fingerprint}, expected {REFERENCE_FINGERPRINTS[seed]}"
            )

        metrics = evaluate_pair(model, evaluation)
        rows.append({
            "seed": seed,
            "evaluation_episode_fingerprint": fingerprint,
            "training_evaluation_semantic_overlap": len(overlap),
            **metrics,
        })

    assert len({r["evaluation_episode_fingerprint"] for r in rows}) == len(rows)
    assert all(r["training_evaluation_semantic_overlap"] == 0 for r in rows)
    assert all(r["q1_intervention_mismatches"] == 0 for r in rows)
    assert all(r["recompute_identity_failures"] == 0 for r in rows)

    normal_values = [r["normal_q2_accuracy"] for r in rows]
    reset_values = [r["reset_recompute_q2_accuracy"] for r in rows]
    gap = statistics.fmean(normal_values) - statistics.fmean(reset_values)
    decision = composition_bootstrap(rows) if not smoke else {
        "point_estimate": None,
        "ci95": [None, None],
        "composition_differences": {},
    }
    ci = decision["ci95"]
    temporal_carry_supported = bool(
        not smoke
        and gap >= GAP_THRESHOLD
        and ci[0] > 0.0
    )

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
            "benchmark_sha256": generator_hash(),
        },
        "protocol": {
            "seeds": list(seeds),
            "steps": STEPS,
            "batch_size": BATCH_SIZE,
            "evaluation_episodes_per_seed": eval_n,
            "hidden_dim": 64,
            "state_write_mode": "residual_linear",
            "reference_experiment_id": REFERENCE_EXPERIMENT_ID,
            "reference_evaluation_fingerprints": REFERENCE_FINGERPRINTS,
            "benchmark_generator_version": GENERATOR_VERSION,
            "model_selection": "none",
            "model_parameter_count": PARAM_COUNT,
            "decision_materiality_gap": GAP_THRESHOLD,
        },
        "seed_results": rows,
        "summary": {
            "normal_q2_mean": statistics.fmean(normal_values),
            "reset_recompute_q2_mean": statistics.fmean(reset_values),
            "normal_minus_reset_recompute_gap": gap,
            "normal_q2_min_seed": min(normal_values),
            "reset_recompute_q2_min_seed": min(reset_values),
            "paired_composition_bootstrap": decision,
            "temporal_carry_supported": temporal_carry_supported,
            "all_q1_pair_integrity_checks_pass": all(
                r["q1_intervention_mismatches"] == 0 for r in rows
            ),
            "all_recompute_identity_checks_pass": all(
                r["recompute_identity_failures"] == 0 for r in rows
            ),
        },
        "leakage_audit": {
            "reference_benchmark_identity_pass": True,
            "evaluation_generated_after_training": True,
            "reference_evaluation_fingerprints_match": True if not smoke else None,
            "training_evaluation_semantic_overlap_zero": all(
                r["training_evaluation_semantic_overlap"] == 0 for r in rows
            ),
            "same_evaluation_objects_within_seed": True,
            "reset_recompute_receives_observations_only": True,
            "q1_outcome_used_only_in_discarded_normal_action_boundary": True,
            "q1_outcome_entered_recomputed_state": False,
            "q1_target_entered_recomputed_state": False,
            "q1_action_entered_recomputed_state": False,
            "verifier_signal_entered_recomputed_state": False,
            "recomputed_state_identity_pass": all(
                r["recompute_identity_failures"] == 0 for r in rows
            ),
            "q1_prediction_action_outcome_pair_integrity_pass": all(
                r["q1_intervention_mismatches"] == 0 for r in rows
            ),
            "no_model_selection": True,
            "no_fresh_generalization_claim": True,
        },
        "claim_boundary": [
            "causal attribution on the registered E2E-008 synthetic benchmark only",
            "not a fresh capability or generalization result",
            "no learned semantic addressing claim",
            "no learned operator discovery claim",
            "no real-world multimodal understanding claim",
            "no scaling or efficiency claim",
        ],
        "reference": ref,
    }

    out = ROOT / "artifacts" / f"{EXPERIMENT_ID}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args().smoke)
