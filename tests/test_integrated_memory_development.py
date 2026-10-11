"""Causal and gradient checks for the development successor."""
from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from tac_osm.development.conditional_compute import ConditionalExecutor  # noqa: E402
from tac_osm.development.full_plm import DevelopmentPLM, StreamingState, VerifiedMTSK  # noqa: E402
from tac_osm.development.full_plm_reference import FullPLMConfig  # noqa: E402
from tac_osm.development.memory_experiment import (  # noqa: E402
    data_seed, followup, measure_pair, tensors, train_model,
)
from tac_osm.development.noisy_memory import MemoryConfig, NoisyMemoryModel  # noqa: E402
from tac_osm.integration_002_benchmark import make_pair  # noqa: E402


@pytest.fixture(autouse=True)
def reproducible_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(17)
    yield
    torch.set_num_threads(previous)


def test_top_one_task_loss_reaches_selector_with_nonzero_gradient():
    executor = ConditionalExecutor(8, depth=2)
    x, q = torch.randn(4,8), torch.randn(4,8)
    result, _ = executor(x,q)
    result.square().mean().backward()
    gradient = executor.selector[0].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert float(gradient.abs().sum()) > 1e-7


def test_early_exit_makes_zero_operator_calls_and_preserves_input():
    executor = ConditionalExecutor(8).eval()
    with torch.no_grad():
        executor.halt[-1].weight.zero_()
        executor.halt[-1].bias.fill_(20)
    counts = []
    handles = [operator.register_forward_hook(lambda *args: counts.append(1)) for operator in executor.operators]
    x, q = torch.randn(4,8), torch.randn(4,8)
    output, _ = executor(x,q)
    for handle in handles:
        handle.remove()
    assert not counts and torch.equal(output,x)
    assert sum(trace.operator_sample_calls for trace in executor.last_trace) == 0


def test_nonhalting_trace_agrees_with_actual_dispatched_sample_count():
    executor = ConditionalExecutor(8, top_k=2, depth=3).eval()
    with torch.no_grad():
        executor.halt[-1].weight.zero_()
        executor.halt[-1].bias.fill_(-20)
    calls = []
    handles = [op.register_forward_hook(lambda op,args,out:calls.append(args[0].shape[0])) for op in executor.operators]
    executor(torch.randn(4,8),torch.randn(4,8))
    for h in handles:
        h.remove()
    assert sum(calls) == 4*2*3
    assert sum(t.operator_sample_calls for t in executor.last_trace) == sum(calls)


def test_mixed_batch_halts_only_finished_samples():
    class SignedHalt(torch.nn.Module):
        def forward(self,joint):
            return joint[:,:1]*30
    executor = ConditionalExecutor(8, depth=2).eval()
    executor.halt = SignedHalt()
    x = torch.zeros(2,8)
    x[0,0], x[1,0] = 1,-1
    result,_ = executor(x,torch.zeros_like(x))
    assert torch.equal(result[0],x[0])
    assert all(0 not in row.active_samples for row in executor.last_trace)
    assert executor.last_trace[0].operator_sample_calls == 1


def test_rejected_full_plm_commit_is_exact_identity_including_decay():
    cfg = FullPLMConfig(d_model=24,num_heads=4)
    kernel = VerifiedMTSK(cfg).eval()
    state = torch.randn(2,4,6,6,6)
    result = kernel.commit_verified(state,torch.randn(2,24),torch.tensor([0,1]),
                                    torch.zeros(2),torch.zeros(2))
    assert torch.equal(result,state)


def test_future_streaming_decision_trains_full_plm_write_projection():
    cfg = FullPLMConfig(d_model=24,num_heads=4)
    model = DevelopmentPLM(cfg)
    state = StreamingState(model.mtsk.initial_state(2,torch.device('cpu'),torch.float32),None)
    state = model.commit(state,torch.randn(2,24),torch.tensor([0,1]),torch.ones(2),torch.ones(2))
    next_state,read,experts = model.observe(torch.randint(0,32,(2,4)),
                                         torch.randn(2,1,16,16),torch.randn(2,1,32),state)
    model.decide(read,experts).square().mean().backward()
    grad = model.mtsk.write_proj.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert next_state.responsibility is not None


def test_streaming_uses_fresh_observation_only_and_preserves_temporal_responsibility():
    model = DevelopmentPLM(FullPLMConfig(d_model=24,num_heads=4)).eval()
    obs = (torch.randint(0,32,(2,4)),torch.randn(2,1,16,16),torch.randn(2,1,32))
    state,_,_ = model.observe(*obs)
    nxt,read,_ = model.observe(*obs,state)
    reset,reset_read,_ = model.observe(*obs)
    assert nxt.memory.shape == reset.memory.shape
    assert not torch.equal(read,reset_read)
    assert state.responsibility.shape == nxt.responsibility.shape


def test_reference_module_remains_byte_identical_to_frozen_source():
    import hashlib
    expected = 'c307667fc05590fd4e0729b1e748e8eeff3152cc52410e82c254b3ca103c931c'
    path = ROOT/'src/tac_osm/development/full_plm_reference.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_noisy_memory_state_capacity_is_fixed_as_history_grows():
    model = NoisyMemoryModel()
    a = model.encode(torch.randn(2,2,12),torch.randint(0,2,(2,2)))
    b = model.encode(torch.randn(2,40,12),torch.randint(0,2,(2,40)))
    assert a.shape == b.shape == (2,4,8,24)
    assert len(model.last_writes) == 40
    assert len(list(model.memory.buffers())) == 0  # no retained hidden history cache


def test_separate_erase_write_gates_and_zero_gate_identity():
    model = NoisyMemoryModel()
    assert model.memory.erase.weight is not model.memory.write.weight
    state = torch.randn(2,4,8,24)
    address,value = torch.randn(2,12),torch.tensor([0,1])
    assert torch.equal(model.memory.update(state,address,value,torch.zeros(2)),state)
    with pytest.raises(ValueError):
        model.memory.update(state,address,value,torch.tensor([float('nan'),0]))


def test_visible_tensor_api_never_receives_target_metadata():
    p = make_pair(42,32,16,1.0,.25)
    altered = replace(p.left,target=1-p.left.target,target_slot=31-p.left.target_slot)
    original = tensors((p.left,p.right))
    changed = tensors((altered,p.right))
    assert all(torch.equal(x,y) for x,y in zip(original,changed))


def test_followup_has_same_present_opposite_past_and_independent_evidence():
    f = followup(72)
    for key in ('address','query','current','evidence'):
        assert torch.equal(f[key][0],f[key][1])
    assert f['value'][0] != f['value'][1] and f['target'][0] != f['target'][1]
    assert not torch.equal(f['address'],f['query'])


def test_followup_commit_affects_future_prediction_and_gets_task_gradient():
    model = NoisyMemoryModel()
    f = followup(44)
    state = model.memory.initial(2,f['address'])
    carried,_ = model.verified_write(state,f['address'],f['value'],f['evidence'])
    logits = model.predict(carried,f['query'],f['current'])
    loss = torch.nn.functional.cross_entropy(logits,f['target'])
    loss.backward()
    for p in (model.memory.write.weight,model.compute.selector[0].weight,
              model.verifier[0].weight):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert not torch.equal(logits[0],logits[1])


def test_reliability_rejection_preserves_old_memory():
    model = NoisyMemoryModel().eval()
    with torch.no_grad():
        model.verifier[-1].weight.zero_()
        model.verifier[-1].bias.fill_(-20)
    state = torch.randn(2,4,8,24)
    result,_ = model.verified_write(state,torch.randn(2,12),torch.tensor([0,1]),torch.ones(2))
    assert torch.equal(result,state)
    assert model.last_writes[0].rejected_samples == 2


def test_history_independent_controls_have_exact_paired_ceiling():
    row = measure_pair(NoisyMemoryModel().eval(),1234,32,1,0.0,.08)
    for prediction in (row['predictions']['reset'],row['followup']['reset'],row['followup']['no_commit']):
        target = row['target'] if prediction is row['predictions']['reset'] else row['followup']['target']
        assert sum(y==t for y,t in zip(prediction,target)) == 1
    assert len(row['work']['history_writes']) == 33


def test_training_changes_parameters_and_reports_component_task_gradients():
    _,record = train_model(43,steps=2)
    assert record['initial_hash'] != record['final_hash']
    assert all(record['nonzero_finite_task_gradients'].values())
    assert record['future_decision_write_gradient']


def test_train_eval_namespaces_are_distinct_and_repeatable():
    assert data_seed('train',43,0,0) == data_seed('train',43,0,0)
    assert data_seed('train',43,0,0) != data_seed('eval',43,0,0)
    with pytest.raises(ValueError):
        followup(1,3)
    with pytest.raises(ValueError):
        train_model(43,steps=0)


def test_validator_rejects_wrong_experiment_identity(tmp_path):
    sys.path.insert(0,str(ROOT/'scripts'))
    spec = importlib.util.spec_from_file_location('development_validator',ROOT/'scripts/validate_integrated_memory_dev.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path/'report.json').write_text(json.dumps({'experiment_id':'wrong'}))
    with pytest.raises(ValueError):
        module.validate(tmp_path)


def test_invalid_conditional_compute_configuration_fails_closed():
    with pytest.raises(ValueError):
        ConditionalExecutor(8,operators=2,top_k=3)
    with pytest.raises(ValueError):
        NoisyMemoryModel(MemoryConfig(read_top_k=5))


@pytest.fixture(scope='module')
def smoke_evidence(tmp_path_factory):
    directory = tmp_path_factory.mktemp('memory-replay')
    subprocess.run([sys.executable,str(ROOT/'scripts/run_integrated_memory_dev.py'),
                    '--smoke','--output',str(directory)],check=True,capture_output=True,text=True)
    return directory


def test_saved_checkpoint_replays_complete_predictions_and_traces(smoke_evidence):
    sys.path.insert(0,str(ROOT/'scripts'))
    from validate_integrated_memory_dev import validate
    assert validate(smoke_evidence)['status'] == 'REPLAY_VALIDATED_DEVELOPMENT_ONLY'


@pytest.mark.parametrize('field', ('prediction','write_trace'))
def test_replay_validator_rejects_prediction_and_work_trace_tampering(smoke_evidence,field):
    sys.path.insert(0,str(ROOT/'scripts'))
    from validate_integrated_memory_dev import validate
    path = smoke_evidence/'report.json'
    original = path.read_text()
    report = json.loads(original)
    if field == 'prediction':
        report['rows'][0]['predictions']['learned'][0] ^= 1
    else:
        report['rows'][0]['work']['history_writes'].pop()
    path.write_text(json.dumps(report))
    try:
        with pytest.raises(ValueError,match='raw predictions/data/work traces'):
            validate(smoke_evidence)
    finally:
        path.write_text(original)
