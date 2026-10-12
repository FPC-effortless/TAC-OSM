"""Protection mechanism, anti-freeze controls and input boundaries."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tac_osm.development.noisy_memory import DeltaMemory,MemoryConfig,parameter_hash  # noqa: E402
from tac_osm.development.protected_memory import ProtectedMemory,ProtectionModel  # noqa: E402
from tac_osm.development.protection_benchmark import make_pair,tensors,visible  # noqa: E402
from tac_osm.development.protection_experiment import matched_models,measure,train,training_batch  # noqa: E402


@pytest.fixture(autouse=True)
def cpu():
    torch.set_num_threads(2)
    torch.manual_seed(43)


def test_matched_arms_start_from_identical_tensors():
    models = matched_models(43)
    assert len(models)==3 and len({parameter_hash(m) for m in models.values()})==1
    assert all(p.requires_grad for p in models['learned_protection'].memory.protection.parameters())
    assert not any(p.requires_grad for p in models['always_update'].memory.protection.parameters())


def test_always_update_is_original_delta_transition():
    cfg = MemoryConfig()
    protected = ProtectedMemory(cfg,'always_update')
    reference = DeltaMemory(cfg)
    reference.load_state_dict({k:v for k,v in protected.state_dict().items() if not k.startswith('protection.')})
    state = torch.randn(2,4,8,24)
    a,v = torch.randn(2,12),torch.tensor([0,1])
    expected = reference.update(state,a,v)
    assert torch.allclose(protected.update(state,a,v),expected,atol=1e-6)


def test_fixed_gate_matches_initial_learned_gate():
    models = matched_models(43)
    a,v = torch.randn(2,12),torch.tensor([0,1])
    state = torch.randn(2,4,8,24)
    x = models['learned_protection'].memory.update(state,a,v)
    y = models['fixed_protection'].memory.update(state,a,v)
    assert torch.equal(x,y)


def test_zero_protection_preserves_entire_state_including_decay():
    memory = ProtectedMemory(MemoryConfig())
    with torch.no_grad():
        memory.protection[-1].weight.zero_()
        memory.protection[-1].bias.fill_(-1000)
    state = torch.randn(2,4,8,24)
    assert torch.equal(memory.update(state,torch.randn(2,12),torch.tensor([0,1])),state)


def test_external_rejection_preserves_protected_state():
    memory = ProtectedMemory(MemoryConfig())
    state = torch.randn(2,4,8,24)
    assert torch.equal(memory.update(state,torch.randn(2,12),torch.tensor([0,1]),torch.zeros(2)),state)
    with pytest.raises(ValueError):
        memory.update(state,torch.randn(2,12),torch.tensor([0,1]),torch.tensor([float('nan'),1]))


@pytest.mark.parametrize('task',('retention','acquisition'))
def test_pairs_have_same_present_opposite_past_without_exact_key(task):
    pair = make_pair(55,32,16,1.0,.08,task)
    assert pair.left.target!=pair.right.target and pair.left.query==pair.right.query
    assert pair.left.current_bit==pair.right.current_bit
    assert set(visible(pair.left))=={'addresses','values','query','current_bit'}
    assert pair.left.query not in visible(pair.left)['addresses']
    assert (pair.left.target_stream_slot<32)==(task=='retention')


def test_oracle_labels_role_and_slot_cannot_enter_model_inputs():
    pair = make_pair(8,32,4,0.0,.25,'retention')
    changed = replace(pair.left,target=1-pair.left.target,target_stream_slot=35,task='acquisition')
    assert all(torch.equal(a,b) for a,b in zip(tensors([pair.left,pair.right]),tensors([changed,pair.right])))


def test_training_batches_balance_roles_and_share_conditions():
    episodes = training_batch(43,0)
    assert len(episodes)==8
    assert sum(e.task=='retention' for e in episodes)==4
    assert sum(e.task=='acquisition' for e in episodes)==4
    assert len({len(e.history)+len(e.writes) for e in episodes})==1


def test_refusing_new_writes_has_exact_acquisition_ceiling():
    row = measure(ProtectionModel().eval(),89,32,16,1.0,.08,'acquisition')
    correct = sum(y==t for y,t in zip(row['predictions']['no_intervening_writes_oracle'],row['target']))
    assert correct==1
    assert sum(y==t for y,t in zip(row['predictions']['reset'],row['target']))==1


def test_actual_gate_trace_counts_every_write_and_contains_valid_gates():
    model = ProtectionModel().eval()
    state = model.encode(torch.randn(2,12,12),torch.randint(0,2,(2,12)))
    assert state.shape==(2,4,8,24)
    assert len(model.last_gate_trace)==12
    assert all(0<=minimum<=mean<=maximum<=1 for mean,minimum,maximum in model.last_gate_trace)


def test_gate_gets_task_gradients_while_all_controls_train():
    _,record = train(43,smoke=True)
    assert len({a['initial_hash'] for a in record['arms'].values()})==1
    for arm,result in record['arms'].items():
        assert all(result['gradient_seen'].values())
        assert result['gate_gradient_seen']==(arm=='learned_protection')
        assert result['initial_hash']!=result['final_hash']


def test_invalid_task_or_gate_mode_fails_closed():
    with pytest.raises(ValueError):
        make_pair(1,32,4,0.0,.08,'oracle')
    with pytest.raises(ValueError):
        ProtectionModel(mode='unknown')


@pytest.fixture(scope='module')
def smoke_evidence(tmp_path_factory):
    directory = tmp_path_factory.mktemp('protection-replay')
    subprocess.run([sys.executable,str(ROOT/'scripts/run_retention_protection_dev.py'),
                    '--smoke','--output',str(directory)],check=True,capture_output=True,text=True)
    return directory


def test_all_matched_checkpoints_replay_predictions_and_gate_traces(smoke_evidence):
    sys.path.insert(0,str(ROOT/'scripts'))
    from validate_retention_protection_dev import validate
    assert validate(smoke_evidence)['status']=='REPLAY_VALIDATED_DEVELOPMENT_ONLY'


@pytest.mark.parametrize('change',('prediction','gate_trace','initialization'))
def test_replay_rejects_altered_predictions_gates_and_initialization(smoke_evidence,change):
    sys.path.insert(0,str(ROOT/'scripts'))
    from validate_retention_protection_dev import validate
    path = smoke_evidence/'report.json'
    original = path.read_text()
    report = json.loads(original)
    if change=='prediction':
        report['records'][0]['rows']['learned_protection'][0]['predictions']['native'][0]^=1
    elif change=='gate_trace':
        report['records'][0]['rows']['learned_protection'][0]['gate_trace_mean_min_max'][0][0]=0.0
    else:
        report['training'][0]['arms']['learned_protection']['initial_hash']='incorrect'
    path.write_text(json.dumps(report))
    try:
        with pytest.raises(ValueError):
            validate(smoke_evidence)
    finally:
        path.write_text(original)
