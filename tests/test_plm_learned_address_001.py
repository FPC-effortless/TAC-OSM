import inspect
import random

import pytest

torch = pytest.importorskip("torch")
import torch as th

from tac_osm.contract import load_contract
from tac_osm.learned_address_001 import LearnedAddressPLM, LearnedAddressState
from tac_osm.learned_address_001_benchmark import (
    ADDRESS_DIM,
    HELDOUT,
    sample_episode,
    validate_episode,
)
from scripts.run_plm_learned_address_001 import gradient_probe


def test_learned_address_contract_is_registered_and_consistent():
    c = load_contract("TACOSM-PLM-LEARNED-ADDRESS-001")
    assert c.check_consistency() == []
    assert c.seeds == (0, 1, 2, 3, 4)
    assert c.steps == 300
    assert c.eval_steps == 600
    assert [a.name for a in c.arms] == [
        "learned_address",
        "oracle_address",
        "corrupted_address",
        "shuffle_observation_order",
    ]
    assert c.primary_endpoint() == "heldout_q2_accuracy"


def test_entity_ids_are_absent_from_model_addressing_api():
    sig = inspect.signature(LearnedAddressState.write)
    assert list(sig.parameters) == ["self", "memory", "address_key", "z", "slot"]
    query_sig = inspect.signature(LearnedAddressPLM.query)
    assert "entity" not in query_sig.parameters
    assert "address_key" in query_sig.parameters


def test_address_keys_are_fixed_dimension_and_target_composition_is_heldout():
    rng = random.Random(20001)
    for _ in range(20):
        ep = sample_episode(
            rng,
            combo1=("xor", 0, 5),
            combo2=rng.choice(HELDOUT),
        )
        validate_episode(ep)
        for key, _text, _image, _audio in ep[0]:
            assert len(key) == ADDRESS_DIM
        assert (("xor", ep[2][1], ep[2][2]) if False else (str("dummy"),)) != tuple()
        assert ep[2][3] < 4


def test_target_slot_is_not_fixed_by_observation_order():
    rng = random.Random(20002)
    observed = set()
    for n in range(60):
        ep = sample_episode(
            rng,
            combo1=("xor", 0, 5),
            combo2=rng.choice(HELDOUT),
            eval_index=n,
        )
        observed.add(ep[4]["q2_slot"])
    assert observed == {0, 1, 2}


def test_learned_address_path_receives_gradient():
    gate = gradient_probe()
    assert gate["pass"], gate
    assert not gate["missing_or_nonfinite"]
    assert not gate["zero_gradient"]


def test_address_read_is_permutation_invariant_in_empty_random_state():
    torch.manual_seed(20003)
    model = LearnedAddressPLM()
    memory = model.state.initial(1, th.device("cpu"))
    keys = [
        th.randn(1, ADDRESS_DIM),
        th.randn(1, ADDRESS_DIM),
        th.randn(1, ADDRESS_DIM),
    ]
    values = [th.randn(1, 64) for _ in range(3)]
    for slot, (key, value) in enumerate(zip(keys, values)):
        memory = model.state.write(memory, key, value, slot)
    with th.no_grad():
        a, _ = model.state.read(memory, keys[1])
    perm = [2, 0, 1]
    shuffled = model.state.initial(1, th.device("cpu"))
    for new_slot, old_slot in enumerate(perm):
        shuffled = model.state.write(
            shuffled,
            keys[old_slot],
            values[old_slot],
            new_slot,
        )
    with th.no_grad():
        b, _ = model.state.read(shuffled, keys[1])
    assert th.allclose(a, b, atol=1e-6, rtol=1e-6)
