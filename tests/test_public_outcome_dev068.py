"""Information boundaries and Bellman accounting, not tests of fixed weights."""
import ast
import importlib.util
from pathlib import Path
import sys
import pytest
torch = pytest.importorskip('torch')
np = pytest.importorskip('numpy')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from tac_osm.development.public_outcome import PublicOutcomeAgent, configurations
from data_public_outcome_dev068 import (PublicMaintenance, measurements, graph_from_public,
                                        corpus, teacher_targets)
from run_public_outcome_dev068 import stage_parameters


def fixture_model():
    model = PublicOutcomeAgent().double()
    # Fixed test fixture: reject noise type, admit other types. Not a fit.
    for p in model.gate.parameters(): p.data.zero_()
    model.gate[0].weight.data[0, 5] = 1.
    model.gate[2].weight.data[0, 0] = -1.
    model.gate[2].bias.data.fill_(.1)
    return model


def test_measurements_are_paid_and_cannot_reach_private_state():
    class PublicOnly:
        def __init__(self): self.cost = 0.; self.actions = []
        def fees(self): return {'inspections': self.cost}
        def execute(self, action):
            assert action in (1, 2)
            self.actions.append(action); self.cost += .12
            return [int(action == 1)], np.zeros((2, 46))
        def __getattr__(self, name): raise AssertionError('Private field requested: '+name)
    env = PublicOnly(); label, trace = measurements(env)
    assert label == 2 and env.actions == [1, 2]
    assert trace['fee'] == pytest.approx(.24)


def test_collector_shadow_excludes_label_probes_and_preserves_accounting():
    model = fixture_model(); data, episodes, coverage = corpus(11681001, 30, model.gate)
    assert coverage['label_probe_actions'] == 2*coverage['states']
    assert len(data['state']) == coverage['transitions'] >= 30
    # Probe actions add rows to the REAL public log, but never to shadow histories.
    for episode in episodes:
        env = PublicMaintenance(episode['seed'], episode['index'], episode['history'], episode['error'], episode['inspection'])
        initial_signals = sum(int(e[5] == 0) for e in env.observations())
        added = 0
        for i, state_id in enumerate(episode['state_ids']):
            if i: added += len(episode['main'][i-1]['events'])
            assert int(data['present'][state_id].sum()) == min(64, initial_signals+added)
    teacher = teacher_targets(episodes, len(data['keys']))
    assert torch.allclose(teacher.sum(-1), torch.ones(len(teacher), dtype=torch.float64))
    terminal = data['done']; nonterminal = ~terminal
    assert torch.all(data['future_label'][terminal] >= 12)
    assert torch.all(data['future_label'][nonterminal] < 12)
    assert torch.allclose(data['physical_cost'][nonterminal], data['train_cost'][nonterminal])
    assert float(data['physical_cost'].sum()) == pytest.approx(coverage['main_fees']+coverage['terminal_costs'], rel=1e-6)
    # Intermediate rewards are public fees, never a hidden-fault difference.
    assert float(data['physical_cost'][nonterminal].max()) <= .64


def test_primary_collector_never_calls_reference_or_reads_hidden_fields():
    source = (ROOT/'scripts/data_public_outcome_dev068.py').read_text()
    tree = ast.parse(source)
    collector = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'corpus')
    forbidden = {'bits', 'parents', 'effects', 'transitions', 'reference', '_world'}
    assert not {n.attr for n in ast.walk(collector) if isinstance(n, ast.Attribute)} & forbidden
    assert not {'teacher_targets', 'snapshot'} & {n.id for n in ast.walk(collector) if isinstance(n, ast.Name)}


def test_policy_has_no_latent_or_current_state_bypass_and_stages_freeze():
    model = fixture_model(); batch = 3
    keys = torch.randn(batch, 2, 16, dtype=torch.float64)
    choices = torch.randn(batch, 5, 19, dtype=torch.float64)
    q = torch.randn(batch, 12, dtype=torch.float64).softmax(-1).requires_grad_()
    output = model.world_from_state(q, keys, choices, torch.ones(batch, 2, dtype=torch.float64),
                                    torch.ones(batch, 1, dtype=torch.float64))
    assert output['features'].shape == (batch, 5, 30)
    # Override just the prediction heads: then changing q cannot change policy.
    with torch.no_grad():
        for head in (model.future, model.cost):
            for parameter in head.parameters(): parameter.zero_()
    other = model.world_from_state(torch.flip(q, [-1]), keys, choices,
                torch.ones(batch, 2, dtype=torch.float64), torch.ones(batch, 1, dtype=torch.float64))
    unchanged = model.world_from_state(q, keys, choices,
                torch.ones(batch, 2, dtype=torch.float64), torch.ones(batch, 1, dtype=torch.float64))
    assert torch.equal(other['features'], unchanged['features'])
    for stage, prefixes in [('encoder', ('local.', 'belief.')), ('world', ('actor.', 'future.', 'cost.')), ('value', ('policy.',))]:
        params = stage_parameters(model, stage)
        assert params
        trainable = [name for name, p in model.named_parameters() if p.requires_grad]
        if stage != 'encoder': assert all(n.startswith(prefixes) for n in trainable)
        else: assert not any(n.startswith(('actor.', 'future.', 'cost.', 'policy.', 'gate.')) for n in trainable)
    with pytest.raises(ValueError):
        model.world_from_state(q, keys, choices, torch.ones(batch, 2), torch.ones(batch))


def test_parameter_matching_and_clock_shift_invariance():
    specs = configurations(); target = specs['tac_outcome']['parameters']
    assert all(abs(s['parameters']/target-1) <= .05 for s in specs.values())
    model = fixture_model(); keys = torch.randn(2, 2, 16, dtype=torch.float64)
    events = torch.randn(2, 7, 46, dtype=torch.float64)
    present = torch.ones(2, 7, dtype=torch.bool)
    events[:, :, 42] = torch.arange(7, dtype=torch.float64)
    a = model.encode(events, present, keys)[1]
    shifted = events.clone(); shifted[:, :, 42] = 113+7*shifted[:, :, 42]
    assert torch.equal(a, model.encode(shifted, present, keys)[1])


def test_runtime_module_has_no_simulator_or_teacher_import():
    tree = ast.parse((ROOT/'src/tac_osm/development/public_outcome.py').read_text())
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not any('world' in (name or '') or 'data_' in (name or '') for name in imports)
