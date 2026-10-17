"""DEV-040A boundaries: balanced supervision, frozen features and controls."""
import copy
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('dev040a_runner', ROOT / 'scripts/run_head_controls_dev040a.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    torch.manual_seed(40)
    y = torch.zeros(4, 8, 3, dtype=torch.long)
    y[:, 0, 0] = torch.tensor([0, 1, 0, 1])
    conditions = [dict(deadline='at', role='reissued_cancel', target_entity=0) for _ in range(4)]
    return dict(labels=y, conditions=conditions, features={'trained': torch.randn(4, 8, 3, 44)})


def test_uniform_weights_cover_every_label_equally():
    d = fixture()
    w = m.weights(d['labels'], d['conditions'], False)
    assert torch.all(w == w.flatten()[0])
    assert abs(float(w.sum()) - 1) < 1e-6


def test_balanced_loss_allocates_half_to_mature_target_wait():
    d = fixture()
    w = m.weights(d['labels'], d['conditions'], True)
    assert abs(float(w[:, 0, 0].sum()) - .5) < 1e-6
    assert abs(float(w.sum()) - 1) < 1e-6
    assert torch.all(w[:, 1:, :] > 0)


def test_balanced_loss_rejects_missing_focus():
    d = fixture()
    for c in d['conditions']:
        c['deadline'] = 'before'
    with pytest.raises(ValueError):
        m.weights(d['labels'], d['conditions'], True)


def test_physical_threshold_is_strict_and_action_conditioned():
    p = [[.5, .25, .8]] * 8
    assert m.services(p, .25) == ()
    assert m.services(p, .2) == tuple(range(8))
    assert m.services([[.5, .5, .8]] * 8, .25) == ()


def test_head_optimization_preserves_frozen_input_features():
    d = fixture()
    before = d['features']['trained'].clone()
    head = torch.nn.Linear(44, 2)
    m.fit_head(head, d, True, .002, 2)
    assert torch.equal(before, d['features']['trained'])
    assert d['features']['trained'].grad is None


def test_probe_statistics_use_training_features_only():
    d = fixture()
    x = m.probe_values(d, 'trained')
    p = m.fit_probe(d, 'trained', 4001, 2)
    torch.testing.assert_close(p['mean'], x.mean(0))
    torch.testing.assert_close(p['scale'], x.std(0, unbiased=False).clamp_min(1e-3))
    assert p['train_examples'] == 4 and p['dimensions'] == 44


def test_all_candidates_expose_exact_existing_44d_head_input():
    events = torch.randn(2, 3, 16)
    entities = torch.tensor([[0, 1, 2], [0, 1, 2]])
    values = torch.zeros(2, 3, dtype=torch.long)
    queries = torch.zeros(2, 8, 3, 14)
    for arm in m.ARMS:
        model = m.build(arm, 44001).eval()
        state = model.encode(events, entities, values)
        assert state[0].numel() == (41 if arm == 'schema41' else 768)
        assert m.feature_input(model, state, queries).shape == (2, 8, 3, 44)


def test_writer_reset_preserves_all_other_trained_components():
    model = m.build('bounded_mtsk', 44001)
    initial_memory = copy.deepcopy(model.memory.state_dict())
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(.01)
    before = copy.deepcopy(model.state_dict())
    model.memory.load_state_dict(initial_memory)
    for key, value in model.state_dict().items():
        assert torch.equal(value, initial_memory[key[len('memory.'):]] if key.startswith('memory.') else before[key])
