"""Meaningful controls for matching, joint action updates and reset training."""
import numpy as np
import torch
from tac_osm.development.joint_repair import JointModel, calibration_bins
from tac_osm.development.adaptive_repair import mixture_update, mixture_values


def test_active_parameter_matching():
    models = {a: JointModel(8541001, a) for a in ('mtsk', 'gru_matched', 'gru_hmm')}
    counts = {a: sum(p.numel() for p in m.parameters()) for a,m in models.items()}
    assert counts == {'mtsk': 1104, 'gru_matched': 1100, 'gru_hmm': 1104}
    for a,m in models.items():
        x = torch.randn(2, 3, 7 if a == 'gru_hmm' else 5)
        m.sequence(x).square().mean().backward()
        if a.startswith('gru'): assert all(p.grad is not None for p in m.parameters())


def test_reset_is_history_independent_in_training():
    torch.set_num_threads(1)
    m = JointModel(8541002, 'mtsk_reset'); m.train()
    first, second = torch.randn(2, 5), torch.randn(2, 5)
    carried = m.observe(m.observe(m.initial(first), first), second)
    fresh = m.observe(m.initial(first), second)
    assert torch.equal(m.predict(carried), m.predict(fresh))
    m.predict(carried).sum().backward()
    assert m.head.weight.grad is not None


def test_joint_service_preserves_regime_fault_dependence():
    j = np.full((2, 2), .25)
    out = mixture_update(j, 'service', np.array([.1, .9]))
    assert np.allclose(out, [[.275, .225], [.475, .025]])
    updated = mixture_update(out, 'inspect', None, 1)
    assert updated[:, 1].sum() > out[:, 1].sum()
    assert not np.allclose(updated.sum(1), [.5, .5])
    v = mixture_values(j, 0, 0, [[.1,.2,.3],[.8,.9,.95]])
    assert set(v) == {'done', 'service', 'inspect'}
    assert all(np.isfinite(list(v.values())))


def test_expected_calibration_checks_probability_quality():
    q = np.array([.2,.8]); perfect = calibration_bins(q, q)
    bad = calibration_bins(1-q, q)
    assert perfect['expected_ece'] == 0
    assert abs(perfect['excess_nll']) < 1e-12
    assert bad['expected_ece'] > .5 and bad['excess_nll'] > 0
