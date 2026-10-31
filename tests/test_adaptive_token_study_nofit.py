"""Constructed no-fit falsifiers. No optimizer steps or real-data claims."""
import copy
import math

import pytest
import torch
from torch.nn import functional as F

from tac_osm.adaptive_codebook_token import AdaptiveCodebookByteTokenGRU
from tac_osm.adaptive_token_controls import ARM_NAMES, TokenComparisonArm
from tac_osm.adaptive_token_evaluation import bits_per_byte, codebook_interventions, decision_regret


def example(raw=b'abc123xyz456!?abc123xyz456!?'):
    targets = torch.tensor([list(raw)], dtype=torch.long)
    ids = torch.cat((torch.tensor([[256]]), targets[:, :-1]), 1)
    return ids, targets


@pytest.mark.parametrize('name', ARM_NAMES)
def test_comparison_candidates_have_live_matched_counts_and_task_gradients(name):
    arm = TokenComparisonArm(name, seed=7,
                             random_byte_probability=.5 if name == 'random_gate' else None)
    expected = {'dense_byte': (100688, 100688), 'forced_token': (100832, 101008),
                'adaptive_latent': (100752, 100752), 'adaptive_codebook': (101008, 101008),
                'frozen_gate_weights': (100832, 101008), 'random_gate': (100832, 100832)}
    receipt = arm.parameter_receipt()
    assert (receipt['trainable_parameters'], receipt['deployed_parameters']) == expected[name]
    assert receipt['within_100k_band'] and not receipt['training_authorized']
    ids, targets = example()
    kwargs = {'random_draws': torch.linspace(0., .99, ids.shape[1]-1)[None]} if name == 'random_gate' else {}
    logits = arm(ids, mode='surrogate', **kwargs)
    F.cross_entropy(logits.reshape(-1, 256), targets.flatten()).backward()
    for field in ('byte_emb', 'gru_cell', 'byte_head'):
        parameters = list(getattr(arm.backbone, field).parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
        assert sum(float(p.grad.abs().sum()) for p in parameters) > 0
    if name in ('forced_token', 'frozen_gate_weights'):
        assert all(not p.requires_grad and p.grad is None for p in arm.backbone.mode_gate.parameters())


def test_seed_pairing_preserves_shared_initial_weights_and_caller_rng():
    before = torch.random.get_rng_state().clone()
    code = TokenComparisonArm('adaptive_codebook', seed=8)
    latent = TokenComparisonArm('adaptive_latent', seed=8)
    assert torch.equal(before, torch.random.get_rng_state())
    for name, weight in latent.backbone.named_parameters():
        assert torch.equal(weight, dict(code.backbone.named_parameters())[name])


def test_random_control_replays_and_isolates_examples_and_future_prefix():
    arm = TokenComparisonArm('random_gate', seed=9, random_byte_probability=.4).eval()
    ids, _ = example()
    batch = torch.cat((ids, ids.flip(1)), 0)
    batch[1, 0] = 256
    batch[1, 1:] %= 256
    draws = torch.linspace(0, .99, 2*(ids.shape[1]-1)).reshape(2, -1)
    with torch.no_grad():
        a = arm(batch, random_draws=draws)
        b = arm(batch.flip(0), random_draws=draws.flip(0)).flip(0)
        changed = batch.clone()
        changed[:, 12:] = 42
        c = arm(changed, random_draws=draws)
    assert torch.equal(a, b)
    assert torch.equal(a[:, :12], c[:, :12])
    with pytest.raises(ValueError, match='explicit'):
        arm.backbone(ids)
    with pytest.raises(ValueError, match='uniform'):
        arm(ids, random_draws=torch.full((1, ids.shape[1]-1), float('nan')))


@pytest.mark.parametrize('probability', [None, 0., 1., float('nan'), True])
def test_random_probability_must_be_frozen_and_non_degenerate(probability):
    with pytest.raises(ValueError):
        TokenComparisonArm('random_gate', seed=1, random_byte_probability=probability)


def test_code_values_intervention_freezes_routes_ids_and_original_state():
    torch.manual_seed(10)
    model = AdaptiveCodebookByteTokenGRU(max_token_bytes=3)
    with torch.no_grad():
        model.mode_gate.weight.zero_()
        model.mode_gate.bias.fill_(-2)
    ids, targets = example()
    before = copy.deepcopy(model.state_dict())
    report = codebook_interventions(model, ids, targets)
    assert model.training
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert report['token_events'] > 0
    assert report['interventions']['joint_row_permutation']['max_logit_change'] == 0
    for name in ('wrong_values_fixed_keys', 'zero_values_fixed_keys'):
        row = report['interventions'][name]
        assert row['route_schedule_preserved'] and row['selector_ids_preserved']
        assert row['max_logit_change'] > 0


def test_no_token_events_do_not_fake_a_codebook_effect():
    torch.manual_seed(11)
    model = AdaptiveCodebookByteTokenGRU()
    with torch.no_grad():
        model.mode_gate.weight.zero_()
        model.mode_gate.bias.fill_(2)
    ids, _ = example()
    report = codebook_interventions(model, ids)
    assert report['token_events'] == 0
    assert all(r['max_logit_change'] == 0 for r in report['interventions'].values())


def test_decision_regret_matches_independent_exact_window_and_targets():
    torch.manual_seed(12)
    model = AdaptiveCodebookByteTokenGRU(max_token_bytes=3).eval()
    ids, targets = example()
    report = decision_regret(model, ids, targets, decision_at=3, horizon=8)
    with torch.no_grad():
        for action in ('byte', 'token'):
            logits = model(ids, intervention_at=3, intervention_mode=action)
            direct = float(F.cross_entropy(logits[:, 3:11].reshape(-1, 256).double(),
                                            targets[:, 3:11].flatten()) / math.log(2))
            assert report['loss_bits_per_byte'][action] == pytest.approx(direct, abs=1e-12)
    assert report['realized_regret_bits_per_byte'] >= 0


def test_scoring_does_not_see_targets_outside_registered_horizon():
    torch.manual_seed(13)
    model = AdaptiveCodebookByteTokenGRU().eval()
    ids, targets = example()
    a = decision_regret(model, ids, targets, decision_at=2, horizon=4)
    changed_targets = targets.clone()
    changed_targets[:, 6:] = 99
    changed_ids = torch.cat((ids[:, :1], changed_targets[:, :-1]), 1)
    b = decision_regret(model, changed_ids, changed_targets, decision_at=2, horizon=4)
    assert a == b


@pytest.mark.parametrize('when,horizon', [(0, 1), (2, 0), (24, 8), (True, 1)])
def test_invalid_or_truncated_horizons_are_rejected(when, horizon):
    ids, targets = example()
    with pytest.raises(ValueError):
        decision_regret(AdaptiveCodebookByteTokenGRU(), ids, targets,
                        decision_at=when, horizon=horizon)


def test_target_shift_and_nonfinite_scores_are_rejected():
    ids, targets = example()
    with pytest.raises(ValueError, match='preceding'):
        codebook_interventions(AdaptiveCodebookByteTokenGRU(), ids, targets.roll(1, 1))
    with pytest.raises(ValueError, match='finite'):
        bits_per_byte(torch.full((*targets.shape, 256), float('nan')), targets)
    assert bits_per_byte(torch.zeros((*targets.shape, 256)), targets) == pytest.approx(8.)


def test_readiness_does_not_construct_an_optimizer_or_authorize_training(monkeypatch):
    from scripts.prepare_sense_tok_learning import readiness
    def forbidden(*args, **kwargs):
        raise AssertionError('optimizer must never be constructed')
    monkeypatch.setattr(torch.optim, 'AdamW', forbidden)
    monkeypatch.setattr(torch.optim, 'SGD', forbidden)
    report = readiness()
    assert report['status'] == 'BLOCKED_BEFORE_FIRST_FIT'
    assert report['optimizer_steps'] == report['model_fits'] == 0
    assert not report['test_accessed'] and not report['training_authorized']
    assert report['compute_plan']['state'] == 'READY_NO_FIT'
    assert not report['compute_plan']['approved_to_train']
    assert max(report['max_relative_parameter_gaps'].values()) <= .01


def test_training_readiness_cli_returns_failure_and_draft_cannot_be_a_training_manifest(capsys):
    import json
    from pathlib import Path
    from scripts.prepare_sense_tok_learning import main
    from scripts.validate_research_parity import check_manifest, ParityError
    assert main(['--check-training']) == 2
    assert json.loads(capsys.readouterr().out)['status'] == 'BLOCKED_BEFORE_FIRST_FIT'
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / 'research/sensory_tokenizers/SENSE-TOK-006-TRAINING-PROTOCOL.json').read_bytes())
    with pytest.raises(ParityError):
        check_manifest(protocol, 'preflight')
