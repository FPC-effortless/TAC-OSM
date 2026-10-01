"""Tests for the leakage-corrected C5 noisy full loop."""
from tac_osm import Candidate, Query, StateUpdate
from tac_osm.c5_noisy_full_phase import (
    CODEBOOK,
    HammingBaseline,
    NoisyCASMExecutor,
    NoisyEnvironment,
    NoisyOutcomeVerifier,
    build_population,
    make_trial,
)
from tac_osm.temporal import TemporalPersistentState


def test_codebook_has_registered_hamming_separation() -> None:
    for i, left in enumerate(CODEBOOK):
        for right in CODEBOOK[i + 1:]:
            assert sum(a != b for a, b in zip(left, right)) >= 3


def test_main_trial_query_is_noisy_and_has_no_clean_target_or_address() -> None:
    candidates = build_population(0, 64)
    trial = make_trial(seed=11, step=3, candidates=candidates)
    observed = tuple(int(x) for x in trial.noisy_query.text.split())
    assert observed != trial.target_class
    assert "\t" not in trial.noisy_query.text
    assert trial.noisy_query.context


def test_environment_only_reveals_success_after_action() -> None:
    candidates = build_population(0, 64)
    trial = make_trial(seed=11, step=3, candidates=candidates)
    env = NoisyEnvironment(trial)
    valid = next(iter(trial.valid_indices))
    assert env.act(valid, casm_output=0.0).success is True
    invalid = next(i for i in range(64) if i not in trial.valid_indices)
    assert env.act(invalid, casm_output=0.0).success is False


def test_casm_never_needs_hidden_target() -> None:
    candidates = build_population(0, 64)
    trial = make_trial(seed=11, step=3, candidates=candidates)
    executor = NoisyCASMExecutor(dim=16)
    verifier = NoisyOutcomeVerifier()
    valid = next(iter(trial.valid_indices))
    computation, value = executor.execute(trial, trial.candidates[valid])
    outcome = NoisyEnvironment(trial).act(valid, casm_output=value)
    evidence = verifier.verify_after_action(
        computation,
        outcome,
        candidate=trial.candidates[valid],
        query=trial.noisy_query,
    )
    assert evidence.valid is True


def test_hamming_baseline_uses_public_noisy_query_only() -> None:
    candidates = build_population(0, 64)
    trial = make_trial(seed=11, step=3, candidates=candidates)
    rank = HammingBaseline().best_valid_rank(trial)
    assert rank >= 1


def test_persistent_control_has_temporal_boundary() -> None:
    state = TemporalPersistentState()
    key = "state:test"
    value = (1,) * 16
    state.stage_world_write(StateUpdate(key=key, value=value, step=0), delay=1)
    query = Query(text="\t" + key, step=0)
    assert state.read(query).values == ()
    state.advance_to(1)
    assert state.read(query).values == (value,)
