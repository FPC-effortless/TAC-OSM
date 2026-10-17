"""Scientific integrity gates for actual MTSK and paired adaptation experiments."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tac_osm.development.adaptive_repair import (
    ARMS, NEURAL, HARD, EASY, AdaptiveModel, baseline_forecasts,
    rate_path, coupon, native_episode, collect_streams, tensors,
    fit_model, mixture_update, mixture_values, rollout_probe,
)
from tac_osm.development.full_plm_reference import SwiLAMTSK
from tac_osm.development.learned_belief_world import bellman
from tac_osm.development.research_statistics import paired_summary


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_protocol_and_state_match_actual_kernel_and_generic_control():
    proto = json.loads((Path(__file__).resolve().parents[1]/'research/dev053b/protocol.json').read_text())
    assert tuple(proto['arms']) == ARMS and tuple(proto['neural_arms']) == NEURAL
    model = AdaptiveModel(7301, 'mtsk')
    assert isinstance(model.kernel, SwiLAMTSK)
    assert model.state_scalars == AdaptiveModel(7301, 'gru128').state_scalars == 128
    assert proto['generator']['levels'] == [0, 4, 8, 16, 32]
    main = {r['seed'] for r in proto['splits'].values()}
    assert main.isdisjoint(seed+proto['smoke_seed_offset'] for seed in main)
    assert all(seed < 8000000 for seed in main)


def test_markov_endpoints_and_unseen_continuous_drift_are_real():
    markov, family = rate_path(8531004, 0, 'markov')
    drift, other = rate_path(8531005, 0, 'drift')
    assert family == 'markov' and other == 'drift'
    assert all(np.allclose(q, HARD) or np.allclose(q, EASY) for q in markov)
    assert any(not np.allclose(q, HARD) and not np.allclose(q, EASY) for q in drift)
    assert not np.allclose(drift[0], drift[8])


def test_coupon_success_is_public_verification_not_hidden_rate_metadata():
    for index in range(4):
        row = coupon(8531001, index, 0, [.4, .5, .6])
        assert row['success'] == int(row['verification']['state'] == 1)
        assert row['native_spent'] == 12 and len(row['calls']) == 2
        ep = native_episode(8531001, index, 0, [.4, .5, .6], 'coupon', row['attempt'])
        public = ep.public_payload()
        assert 'oracle' not in public and 'repair_rates' not in str(public)
        assert public['task']['metadata']['public_known_initial_fault'] == 1


def test_public_forecasts_are_invariant_to_hidden_target_tampering():
    streams = collect_streams({'seed': 8531001, 'worlds': 1, 'kind': 'train'})
    altered = copy.deepcopy(streams)
    altered[0]['targets'] = [[.99]*3]*33
    params = {'prior': [.5]*3, 'endpoints': [HARD.tolist(), EASY.tolist()], 'training_hazard': .08}
    for arm in ('prior', 'stationary_beta', 'discounted_beta', 'hmm_plugin'):
        left = baseline_forecasts(streams, arm, params, {'discount': .85})[0]
        right = baseline_forecasts(altered, arm, params, {'discount': .85})[0]
        assert np.array_equal(left, right)
    assert tensors(streams)[0].shape == (1, 32, 5)


@pytest.mark.parametrize('arm', NEURAL)
def test_forecasts_are_prefix_causal_and_incremental_api_matches(arm):
    model = AdaptiveModel(7301, arm)
    x = torch.randn(2, 12, 5)
    changed = x.clone(); changed[:, 6:] = torch.randn_like(changed[:, 6:])*3
    first, second = model.sequence(x), model.sequence(changed)
    assert torch.equal(first[:, :7], second[:, :7])
    state = model.initial(x[:, 0]); outputs = [model.predict(state)]
    for t in range(x.shape[1]):
        state = model.observe(state, x[:, t]); outputs.append(model.predict(state))
    assert torch.equal(torch.stack(outputs, 1), first)


def test_reset_ablates_carry_without_altering_last_token_or_parameters():
    a, b = torch.randn(1, 7, 5), torch.randn(1, 7, 5)
    b[:, -1] = a[:, -1]
    reset = AdaptiveModel(7301, 'mtsk_reset')
    normal = AdaptiveModel(7301, 'mtsk')
    assert all(torch.equal(v, normal.state_dict()[k]) for k, v in reset.state_dict().items())
    assert torch.equal(reset.sequence(a)[:, -1], reset.sequence(b)[:, -1])
    assert not torch.equal(normal.sequence(a)[:, -1], normal.sequence(b)[:, -1])


def test_frozen_control_keeps_kernel_but_trains_head():
    x, y = torch.randn(3, 4, 5), torch.rand(3, 5, 3)
    model = AdaptiveModel(7301, 'mtsk_frozen_kernel')
    kernel = {k: v.clone() for k, v in model.kernel.state_dict().items()}
    before = model.head.weight.clone()
    fit_model(model, (x, y), 7301, updates=2)
    assert all(torch.equal(v, model.kernel.state_dict()[k]) for k, v in kernel.items())
    assert not torch.equal(before, model.head.weight)


def test_mixture_joint_update_learns_regime_after_uncertain_service_and_sensor():
    j = np.array([[.25, .25], [.25, .25]])
    j = mixture_update(j, 'service', np.array([.25, .9]))
    negative = mixture_update(j, 'inspect', None, 0)
    positive = mixture_update(j, 'inspect', None, 1)
    assert negative.sum(1)[1] > .5 and positive.sum(1)[1] < .5
    assert np.allclose(positive.sum(), 1)


def test_mixture_dp_reduces_to_known_rate_dp_when_regime_is_known():
    j = np.array([[0., 0.], [.35, .65]])
    values = mixture_values(j, 0, 2, [HARD, EASY])
    known = bellman(.65, 0, 2, (.85, tuple(EASY)))
    assert min(values.values()) == pytest.approx(known[0], abs=1e-6)


def test_native_probe_charges_common_history_and_absorbs_healthy_state():
    rates, family = rate_path(8531004, 1, 'markov')
    row = {'seed': 8531004, 'index': 1, 'targets': rates, 'family': family}
    result = rollout_probe(row, 4, rates[4], 'oracle', {})
    assert result['cost'] == pytest.approx(result['native_spent']/10+4*result['failures'])
    assert result['total_cost_including_observations']-result['cost'] == pytest.approx(4*1.2)
    assert result['blocked'] == result['violations'] == 0


def test_crossed_bootstrap_retains_seed_uncertainty_even_for_constant_worlds():
    costs = np.array([[1.]*64, [2.]*64])
    result = paired_summary(costs, np.ones(64)*1.5)
    assert result['world_clusters'] == 64 and result['model_seeds'] == 2
    assert result['paired_delta_95pct'][0] < -.1
    assert result['paired_delta_95pct'][1] > .1


def test_zero_oracle_headroom_is_explicitly_undefined():
    result = paired_summary(np.ones((5, 8)), np.ones(8), np.ones(8))
    assert result['fraction_oracle_gap_closed'] is None
    assert result['fraction_invalid_denominator_rate'] == 1.
