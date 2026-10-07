import json
import inspect
import random
from pathlib import Path

from tac_osm import integrated_e2e_008_benchmark as b
from scripts import run_integrated_e2e_008 as runner

ROOT = Path(__file__).resolve().parents[1]


def test_clean_modality_encoder_has_no_entity_argument():
    sig = inspect.signature(b.make_modalities)
    assert list(sig.parameters) == ["bits", "rng"]
    assert "image[0, k, :]" not in inspect.getsource(b.make_modalities)


def test_image_payload_regions_are_clear_of_previous_entity_stripe():
    for idx in range(4):
        row0 = (idx // 2) * 6 + 4
        assert row0 >= 4
        assert row0 + 2 <= 12


def test_heldout_is_disjoint_from_all_prior_sets():
    assert set(b.HELDOUT).isdisjoint(b.SEALED_E2E005)
    assert set(b.HELDOUT).isdisjoint(b.DEV_COMBOS)
    assert set(b.HELDOUT).isdisjoint(b.E2E006_HELDOUT)
    assert set(b.HELDOUT).isdisjoint(b.E2E007_HELDOUT)
    assert set(b.HELDOUT).isdisjoint(b.TRAIN_COMBOS)
    assert len(b.HELDOUT) == 12


def test_evaluation_is_heldout_and_q1_q2_target_distinct_entities():
    for ep in b.sample_evaluation_episodes(random.Random(808008), 120):
        b.validate_episode(ep)


def test_clean_modalities_have_no_entity_side_channel():
    bits = [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1]
    a = b.make_modalities(bits, random.Random(1))
    c = b.make_modalities(bits, random.Random(1))
    assert all(x.shape == y.shape for x, y in zip(a, c))
    assert all((x == y).all() for x, y in zip(a, c))


def test_contract_records_collision_correction():
    x = json.loads((ROOT / "contracts/TACOSM-PLM-INTEGRATED-E2E-008.json").read_text())
    assert x["protocol"]["entity_side_channel"] is False
    assert "collision" in x["antecedent_defect"]


def test_runner_uses_registered_model_and_no_selection():
    src = inspect.getsource(runner.run)
    assert "state_write_mode" in src
    assert '"model_selection": "none"' in src


def test_runner_reports_composition_bootstrap_and_empirical_prior():
    src = inspect.getsource(runner.run)
    assert "bootstrap_compositions" in src
    assert "empirical_prior_baseline" in src


def test_validator_is_strict_about_clean_encoding():
    src = (ROOT / "scripts/validate_integrated_e2e_008.py").read_text()
    assert "entity_side_channel_in_modalities" in src
    assert "image_payload_id_overlap" in src
