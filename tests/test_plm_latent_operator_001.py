import ast
import inspect
import json
import random
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import Tensor

from tac_osm.contract import load_contract
from tac_osm.latent_operator_001 import LatentOperatorPLM
from tac_osm.latent_operator_001_benchmark import (
    OPS,
    SUPPORT_PAIRS,
    HELDOUT,
    make_support_context,
    sample_episode,
    validate_episode,
)
from scripts.run_plm_latent_operator_001 import (
    PARAM_COUNT,
    build_gradient_probe,
    rotate_support_labels,
)

ROOT = Path(__file__).resolve().parents[1]


def test_latent_operator_contract_is_registered_and_consistent():
    c = load_contract("TACOSM-PLM-LATENT-OPERATOR-001")
    assert c.check_consistency() == []
    assert c.seeds == (0, 1, 2, 3, 4)
    assert c.steps == 300
    assert c.eval_steps == 600
    assert [a.name for a in c.arms] == [
        "normal_latent_operator",
        "oracle_operator",
        "shuffle_support_labels",
        "no_memory",
    ]
    assert c.primary_endpoint() == "heldout_q2_accuracy"


def test_normal_latent_query_has_no_operator_id_argument():
    signature = inspect.signature(LatentOperatorPLM.latent_query)
    assert "op" not in signature.parameters
    assert "operator_id" not in signature.parameters
    assert "support" in signature.parameters


def test_support_context_is_a_permuted_truth_table_without_hidden_opcode():
    rng = random.Random(19001)
    for op in OPS:
        support = make_support_context(op, rng)
        assert sorted(support) == sorted(
            (a, b, {"xor": a ^ b, "and": a & b, "or": a | b, "xnor": 1 - (a ^ b)}[op])
            for a, b in SUPPORT_PAIRS
        )
    samples = {
        make_support_context("xor", random.Random(seed))
        for seed in range(20)
    }
    assert len(samples) > 1


def test_operator_inducer_is_permutation_invariant():
    torch.manual_seed(19002)
    model = LatentOperatorPLM()
    support = Tensor(
        [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]],
    ).unsqueeze(0).float()
    permuted = support[:, [2, 0, 3, 1], :]
    with torch.no_grad():
        a = model.infer_operator(support)
        b = model.infer_operator(permuted)
    assert torch.equal(a, b)


def test_operator_inducer_receives_task_gradient():
    gate = build_gradient_probe()
    assert gate["pass"], gate
    assert not gate["missing_or_nonfinite"]
    assert not gate["zero_operator_gradient"]


def test_evaluation_compositions_are_from_registered_heldout_set():
    rng = random.Random(19003)
    for _ in range(20):
        ep = sample_episode(rng, combo2=rng.choice(HELDOUT))
        validate_episode(ep)
        assert (ep[3], ep[1][1], ep[1][2]) in HELDOUT
